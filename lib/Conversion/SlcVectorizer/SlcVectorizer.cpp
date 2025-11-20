#include "lib/Conversion/SlcVectorizer/SlcVectorizer.h"

#include <memory>
#include <utility>

#include "lib/Dialect/SLC/SlcDialect.h"
#include "lib/Dialect/SLC/SlcOps.h"
#include "lib/Dialect/SLC/SlcTypes.h"
#include "lib/Dialect/SLCVEC/SlcVecDialect.h"
#include "lib/Dialect/SLCVEC/SlcVecOps.h"
#include "lib/Dialect/SLCVEC/SlcVecTypes.h"
#include "mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#define GEN_PASS_DEF_SLCVECTORIZER
#include "lib/Conversion/SlcVectorizer/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::slc;
using namespace mlir::slcvec;

bool isVectorizableBlock(Block *block) {
  if (SlcVecForOp slcVecForOp = dyn_cast<SlcVecForOp>(block->getParentOp())) {
    for (SlcVectorizeForOp slcVectorizeForOp :
         slcVecForOp.getOps<SlcVectorizeForOp>()) {
      return true;
    }
  }
  return false;
}

class SlcVectorizerTypeConverter : public TypeConverter {
public:
  SlcVectorizerTypeConverter(MLIRContext *ctx, long vectorLength) {
    addConversion([](Type type) -> Type { return type; });
    addConversion(
        [](MemRefType memRefType) -> MemRefType { return memRefType; });
    addConversion(
        [vectorLength](SlcStreamType slcStreamType) -> slc::SlcStreamType {
          return SlcStreamType::get(
              VectorType::get({vectorLength}, slcStreamType.getElementType()));
        });

    // Convert a SLC vector stream type to a stream type
    addSourceMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<SlcTmpVecStreamCastOp>(loc, type, inputs[0]);
    });
    // Convert a stream type to a vector stream type
    addTargetMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<SlcBroadcastStreamOp>(loc, type, inputs[0]);
    });
  }
};

long getVectorLength(const TypeConverter *typeConverter, MLIRContext *context) {
  SlcStreamType inStreamType = SlcStreamType::get(IndexType::get(context));
  SlcStreamType outStreamType =
      cast<SlcStreamType>(typeConverter->convertType(inStreamType));
  VectorType vectorType = cast<VectorType>(outStreamType.getElementType());
  return vectorType.getShape().front();
}

arith::ConstantOp getPassThruMask(SlcForOp slcForOp, long vectorLength,
                                  PatternRewriter &rewriter) {
  if (func::FuncOp funcOp = slcForOp->getParentOfType<func::FuncOp>()) {
    for (arith::ConstantOp constantOp : funcOp.getOps<arith::ConstantOp>()) {
      if (VectorType vectorType = dyn_cast<VectorType>(constantOp.getType())) {
        if (vectorType.getShape() == ArrayRef<long>({vectorLength})) {
          if (vectorType.getElementType() == rewriter.getI1Type()) {
            if (constantOp.getValue() == rewriter.getOneAttr(vectorType)) {
              return constantOp;
            }
          }
        }
      }
    }
    assert(false);
  } else {
    assert(false);
  }
}

/// Conversion Patterns.
class ConvertSlcForOp : public OpConversionPattern<SlcForOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SlcForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    bool shouldVectorize = false;
    for (SlcVectorizeForOp slcVectorizeForOp : op.getOps<SlcVectorizeForOp>()) {
      rewriter.eraseOp(slcVectorizeForOp);
      shouldVectorize = true;
    }

    if (shouldVectorize) {
      Location loc = op.getLoc();
      Value lowerBound = adaptor.getLowerBound();
      Value upperBound = adaptor.getUpperBound();
      Value step = adaptor.getStep();
      long vectorLength = getVectorLength(getTypeConverter(), getContext());
      LoopConfig loopConfig = LoopConfig::none;
      Value mask;
      if (SlcVecForOp slcVecForOp = dyn_cast<SlcVecForOp>(*op->getParentOp())) {
        loopConfig = LoopConfig::vec;
        mask = slcVecForOp.getOutMask();
      } else if (SlcForOp slcForOp = dyn_cast<SlcForOp>(*op->getParentOp())) {
        loopConfig = LoopConfig::bcast;
        mask = getPassThruMask(op, vectorLength, rewriter);
        assert(mask);
      } else {
        loopConfig = LoopConfig::bcast;
        mask = getPassThruMask(op, vectorLength, rewriter);
        assert(mask);
      }

      SlcVecForOp slcVecForOp = rewriter.create<SlcVecForOp>(
          loc, lowerBound, upperBound, step, mask, vectorLength, loopConfig);
      rewriter.eraseBlock(slcVecForOp.getBody());

      // Move the blocks from the forOp into the slcForOp. This is the body of
      // the slcForOp.
      rewriter.inlineRegionBefore(op.getRegion(), slcVecForOp.getRegion(),
                                  slcVecForOp.getRegion().end());

      // Create the new induction variable to use.
      Type elementType =
          mlir::cast<SlcStreamType>(slcVecForOp.getInductionVar().getType())
              .getElementType();
      Type newIndVarType =
          slc::SlcStreamType::get(VectorType::get({vectorLength}, elementType));
      Type newMaskType = slc::SlcStreamType::get(
          VectorType::get({vectorLength}, rewriter.getI1Type()));

      // Apply signature conversion to the body of the forOp. It has a single
      // block, with argument which is the induction variable. That has to be
      // replaced with the new induction variable.
      TypeConverter::SignatureConversion signatureConverter(
          slcVecForOp.getBody()->getNumArguments());
      signatureConverter.addInputs(0, newIndVarType);
      signatureConverter.addInputs(newMaskType);
      rewriter.applySignatureConversion(&slcVecForOp.getRegion().front(),
                                        signatureConverter, getTypeConverter());
      rewriter.replaceOp(op, slcVecForOp);
      return success();
    } else {
      return failure();
    }
  }
};

class ConvertSlcYieldOp : public OpConversionPattern<SlcYieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SlcYieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (isVectorizableBlock(op->getBlock())) {
      SlcVecYieldOp slcVecYieldOp =
          rewriter.replaceOpWithNewOp<SlcVecYieldOp>(op);
      return success();
    } else {
      return failure();
    }
  }
};

class ConvertSlcMemStreamOp : public OpConversionPattern<SlcMemStreamOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SlcMemStreamOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isVectorizableBlock(op->getBlock())) {
      long vectorLength = getVectorLength(getTypeConverter(), getContext());
      Type elementType =
          cast<MemRefType>(adaptor.getMemref().getType()).getElementType();
      Type newResTy =
          SlcStreamType::get(VectorType::get({vectorLength}, elementType));
      SlcVecMemStreamOp slcVecMemStreamOp =
          rewriter.replaceOpWithNewOp<SlcVecMemStreamOp>(
              op, newResTy, adaptor.getMemref(), adaptor.getIndices(),
              op.getNontemporal());
      return success();
    }
    return failure();
  }
};

/*
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
*/

/*class ConvertSlcFromStreamOp : public OpConversionPattern<SlcFromStreamOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SlcFromStreamOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (long vectorLength =
            isVectorizableBlock(op.getStream().getParentBlock())) {
      Type elementType = op.getType();
      Type newResTy = VectorType::get({vectorLength}, elementType);
      rewriter.replaceOpWithNewOp<slc::SlcFromVecStreamOp>(op, newResTy,
                                                           adaptor.getStream());
      return success();
    }
    return failure();
  }
};*/

class SlcVectorizerPass
    : public mlir::impl::SlcVectorizerBase<SlcVectorizerPass> {

public:
  long vectorLength = 0;
  SlcVectorizerPass(long vectorLength) : vectorLength(vectorLength) {}

  void runOnOperation() override;

  /// Runs the vectorizer pass on each function.
  LogicalResult runOnFunction(func::FuncOp func);
};

void SlcVectorizerPass::runOnOperation() {
  MLIRContext *context = &getContext();
  Operation *module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<slcvec::SlcVecDialect>();

  IRRewriter rewriter(getOperation()->getContext());
  VectorType vectorType = VectorType::get({vectorLength}, rewriter.getI1Type());
  rewriter.setInsertionPointToStart(&getOperation()->getRegion(0).front());
  arith::ConstantOp mask = rewriter.create<arith::ConstantOp>(
      getOperation()->getLoc(), vectorType, rewriter.getOneAttr(vectorType));
  rewriter.finalizeOpModification(getOperation());

  RewritePatternSet patterns(context);

  long vectorLength = 8;
  SlcVectorizerTypeConverter typeConverter(context, vectorLength);
  patterns.add<ConvertSlcForOp>(typeConverter, context);
  patterns.add<ConvertSlcMemStreamOp>(typeConverter, context);
  patterns.add<ConvertSlcYieldOp>(typeConverter, context);
  // patterns.add<ConvertSlcFromStreamOp>(typeConverter, context);
  // patterns.add<ConvertAddIOp>(typeConverter, context);
  // patterns.add<ConvertMulIOp>(typeConverter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::createSlcVectorizerPass(long vectorLength) {
  return std::make_unique<SlcVectorizerPass>(vectorLength);
}