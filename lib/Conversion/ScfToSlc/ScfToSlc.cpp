#include "lib/Conversion/ScfToSlc/ScfToSlc.h"

#include <memory>
#include <utility>

#include "lib/Dialect/SLC/SlcDialect.h"
#include "lib/Dialect/SLC/SlcOps.h"
#include "lib/Dialect/SLC/SlcTypes.h"
#include "mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/include/mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/lib/IR/AffineExprDetail.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"
#include "llvm/include/llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"

namespace mlir {
#define GEN_PASS_DEF_SCFTOSLC
#include "lib/Conversion/ScfToSlc/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::sparse_tensor;

Region::iterator getBlockIt(Region &region, unsigned index) {
  return std::next(region.begin(), index);
}

class ScfToSlcTypeConverter : public TypeConverter {
public:
  ScfToSlcTypeConverter(MLIRContext *ctx) {
    addConversion(
        [ctx](MemRefType memRefType) -> MemRefType { return memRefType; });
    addConversion([ctx](IndexType indexType) -> slc::SlcStreamType {
      return slc::SlcStreamType::get(indexType);
    });
    addConversion([ctx](Float64Type float64Type) -> slc::SlcStreamType {
      return slc::SlcStreamType::get(float64Type);
    });

    // Convert a SLC stream type to its element type
    addSourceMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<slc::SlcFromStreamOp>(loc, inputs[0]);
    });

    // Convert a type to a SLC stream type
    addTargetMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<slc::SlcToStreamOp>(loc, inputs[0]);
    });
  }
};

/// Conversion Patterns.
class ConvertForOp : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (not llvm::isa<slc::SlcAccelerateForOp>(op.getBody()->front())) {
      return failure();
    }

    rewriter.eraseOp(&op.getBody()->front());

    Location loc = op.getLoc();
    Value lowerBound = adaptor.getLowerBound();
    Value upperBound = adaptor.getUpperBound();
    Value step = adaptor.getStep();
    slc::SlcForOp slcForOp =
        rewriter.create<slc::SlcForOp>(loc, lowerBound, upperBound, step);
    rewriter.eraseBlock(slcForOp.getBody());

    // Move the blocks from the forOp into the slcForOp. This is the body of the
    // slcForOp.
    rewriter.inlineRegionBefore(op.getRegion(), slcForOp.getRegion(),
                                slcForOp.getRegion().end());

    // Create the new induction variable to use.
    Type newIndVarType =
        slc::SlcStreamType::get(slcForOp.getBody()->getArgument(0).getType());

    // Apply signature conversion to the body of the forOp. It has a single
    // block, with argument which is the induction variable. That has to be
    // replaced with the new induction variable.
    TypeConverter::SignatureConversion signatureConverter(
        slcForOp.getBody()->getNumArguments());
    signatureConverter.addInputs(0, newIndVarType);
    rewriter.applySignatureConversion(&slcForOp.getRegion().front(),
                                      signatureConverter, getTypeConverter());

    rewriter.replaceOp(op, slcForOp);
    return success();
  }
};

class ConvertLoadOp : public OpConversionPattern<memref::LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (op->getParentOfType<slc::SlcCallbackOp>() != nullptr) {
      return failure();
    }

    Type newResTy = getTypeConverter()->convertType(op.getType());
    if (!newResTy)
      return rewriter.notifyMatchFailure(
          op->getLoc(), llvm::formatv("failed to convert memref type: {0}",
                                      op.getMemRefType()));

    rewriter.replaceOpWithNewOp<slc::SlcMemStreamOp>(
        op, newResTy, adaptor.getMemref(), adaptor.getIndices(),
        op.getNontemporal());
    return success();
  }
};

class ConvertAddIOp : public OpConversionPattern<arith::AddIOp> {
public:
  ConvertAddIOp(mlir::MLIRContext *context)
      : OpConversionPattern<arith::AddIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (op->getParentOfType<slc::SlcCallbackOp>() != nullptr) {
      return failure();
    }

    slc::SlcAluStreamOp slcAluStreamOp = rewriter.create<slc::SlcAluStreamOp>(
        op.getLoc(), adaptor.getLhs(), adaptor.getRhs(), slc::OpType::add);
    rewriter.replaceOp(op, slcAluStreamOp);
    return success();
  }
};

class ConvertMulIOp : public OpConversionPattern<arith::MulIOp> {
public:
  ConvertMulIOp(mlir::MLIRContext *context)
      : OpConversionPattern<arith::MulIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (op->getParentOfType<slc::SlcCallbackOp>() != nullptr) {
      return failure();
    }

    slc::SlcAluStreamOp slcAluStreamOp = rewriter.create<slc::SlcAluStreamOp>(
        op.getLoc(), adaptor.getLhs(), adaptor.getRhs(), slc::OpType::mul);
    rewriter.replaceOp(op, slcAluStreamOp);
    return success();
  }
};

class ConvertYieldOp : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (not llvm::isa<slc::SlcForOp>(op->getParentOp())) {
      return failure();
    }

    Location loc = op.getLoc();
    slc::SlcYieldOp slcYieldOp = rewriter.create<slc::SlcYieldOp>(loc);
    rewriter.replaceOp(op, slcYieldOp);
    return success();
  }
};

class ScfToSlcPass : public mlir::impl::ScfToSlcBase<ScfToSlcPass> {
  void runOnOperation() override;

  /// Runs the convert scf-to-slc pass on each function.
  LogicalResult runOnFunction(func::FuncOp func);
};

void ScfToSlcPass::runOnOperation() {
  MLIRContext *context = &getContext();
  Operation *module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<slc::SlcDialect>();

  RewritePatternSet patterns(context);

  ScfToSlcTypeConverter typeConverter(context);
  patterns.add<ConvertForOp>(typeConverter, context);
  patterns.add<ConvertLoadOp>(typeConverter, context);
  patterns.add<ConvertAddIOp>(typeConverter, context);
  patterns.add<ConvertMulIOp>(typeConverter, context);
  patterns.add<ConvertYieldOp>(typeConverter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::createScfToSlcPass() {
  return std::make_unique<ScfToSlcPass>();
}