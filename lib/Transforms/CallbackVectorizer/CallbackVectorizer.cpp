#include "lib/Transforms/CallbackVectorizer/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/include/mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/include/mlir/Pass/PassManager.h" // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"
#include "mlir/include/mlir/Support/LogicalResult.h" // from @llvm-project
#include <memory>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "lib/Dialect/SLC/SlcOps.h"
#include "lib/Dialect/SLCVEC/SlcVecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace slc {
using namespace mlir::slcvec;

#define GEN_PASS_DEF_CALLBACKVECTORIZER
#include "lib/Transforms/CallbackVectorizer/Passes.h.inc"

struct CallbackVectorizer
    : public impl::CallbackVectorizerBase<CallbackVectorizer> {
  using CallbackVectorizerBase::CallbackVectorizerBase;

  CallbackVectorizer() {}

  void runOnOperation() override;
};

template <class T>
void addOutTmpCast(T op, Value newOp, PatternRewriter &rewriter) {
  SlcTmpVecCastOp slcTmpVecCastOp =
      rewriter.create<SlcTmpVecCastOp>(op.getLoc(), newOp);
  op->replaceAllUsesWith(slcTmpVecCastOp);
  rewriter.eraseOp(op);
}

void eraseInTmpCast(ArrayRef<Operation *> ops, PatternRewriter &rewriter) {
  for (Operation *op : ops) {
    if (op->getUses().empty()) {
      rewriter.eraseOp(op);
    }
  }
}

Value getMask(SlcCallbackOp &slcCallbackOp, PatternRewriter &rewriter) {
  SlcVecForOp slcVecForOp = cast<SlcVecForOp>(slcCallbackOp->getParentOp());
  Value maskStream = slcVecForOp.getOutMask();
  for (OpOperand &use : maskStream.getUses()) {
    if (use.getOwner()->getParentOp() == slcCallbackOp.getOperation()) {
      if (SlcFromStreamOp slcFromStreamOp =
              dyn_cast<SlcFromStreamOp>(use.getOwner())) {
        return slcFromStreamOp.getValue();
      }
    }
  }
  RewriterBase::InsertPoint ip = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(slcCallbackOp.getBody());
  SlcFromStreamOp slcFromStreamOp =
      rewriter.create<SlcFromStreamOp>(slcCallbackOp->getLoc(), maskStream);
  rewriter.restoreInsertionPoint(ip);
  return slcFromStreamOp;
}

struct VectorizeSlcFromStreamOp : public OpRewritePattern<SlcFromStreamOp> {
  VectorizeSlcFromStreamOp(mlir::MLIRContext *context)
      : OpRewritePattern<SlcFromStreamOp>(context) {}

  LogicalResult matchAndRewrite(SlcFromStreamOp op,
                                PatternRewriter &rewriter) const override {

    if (SlcTmpVecStreamCastOp slcTmpVecStreamCastOp =
            op.getStream().getDefiningOp<SlcTmpVecStreamCastOp>()) {
      Value vectorStream = slcTmpVecStreamCastOp.getIn();
      SlcFromStreamOp newSlcFromStreamOp =
          rewriter.create<SlcFromStreamOp>(op.getLoc(), vectorStream);
      addOutTmpCast(op, newSlcFromStreamOp, rewriter);
      eraseInTmpCast({slcTmpVecStreamCastOp.getOperation()}, rewriter);
      return success();
    } else {
      return failure();
    }
  }
};

struct VectorizeMemRefLoadOp : public OpRewritePattern<memref::LoadOp> {
  VectorizeMemRefLoadOp(mlir::MLIRContext *context)
      : OpRewritePattern<memref::LoadOp>(context) {}

  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter &rewriter) const override {

    auto indices = op.getIndices();
    if (SlcTmpVecCastOp slcTmpVecCastOp =
            indices.back().getDefiningOp<SlcTmpVecCastOp>()) {
      Value idxVectorValue = slcTmpVecCastOp.getIn();
      VectorType idxVectorType = cast<VectorType>(idxVectorValue.getType());
      VectorType resVectorType =
          VectorType::get(idxVectorType.getShape(), op.getType());
      Value memRef = op.getMemRef();
      SmallVector<Value> scalarArgs(indices.begin(), indices.end());
      scalarArgs.back() =
          rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
      SlcCallbackOp slcCallbackOp = cast<SlcCallbackOp>(op->getParentOp());
      Value mask = getMask(slcCallbackOp, rewriter);
      Value pass = rewriter.create<arith::ConstantOp>(
          op.getLoc(), resVectorType, rewriter.getZeroAttr(resVectorType));
      rewriter.setInsertionPoint(op.getOperation());
      vector::GatherOp vectorGatherOp = rewriter.create<vector::GatherOp>(
          op.getLoc(), resVectorType, memRef, scalarArgs, idxVectorValue, mask,
          pass);
      addOutTmpCast(op, vectorGatherOp, rewriter);
      eraseInTmpCast({slcTmpVecCastOp}, rewriter);
      return success();
    } else {
      return failure();
    }
  }
};

struct VectorizeArithAddFOp : public OpRewritePattern<arith::AddFOp> {
  VectorizeArithAddFOp(mlir::MLIRContext *context)
      : OpRewritePattern<arith::AddFOp>(context) {}

  LogicalResult matchAndRewrite(arith::AddFOp op,
                                PatternRewriter &rewriter) const override {

    if (SlcTmpVecCastOp lhsSlcTmpVecCastOp =
            op.getLhs().getDefiningOp<SlcTmpVecCastOp>()) {
      if (SlcTmpVecCastOp rhsSlcTmpVecCastOp =
              op.getRhs().getDefiningOp<SlcTmpVecCastOp>()) {
        Value lhsVec = lhsSlcTmpVecCastOp.getIn();
        Value rhsVec = rhsSlcTmpVecCastOp.getIn();
        arith::AddFOp newAddFOp =
            rewriter.create<arith::AddFOp>(op.getLoc(), lhsVec, rhsVec);
        addOutTmpCast(op, newAddFOp, rewriter);
        eraseInTmpCast({lhsSlcTmpVecCastOp, rhsSlcTmpVecCastOp}, rewriter);
        return success();
      } else {
        return failure();
      }
    } else {
      return failure();
    }
  }
};

struct VectorizeMemRefStoreOp : public OpRewritePattern<memref::StoreOp> {
  VectorizeMemRefStoreOp(mlir::MLIRContext *context)
      : OpRewritePattern<memref::StoreOp>(context) {}

  LogicalResult matchAndRewrite(memref::StoreOp op,
                                PatternRewriter &rewriter) const override {

    auto indices = op.getIndices();
    if (SlcTmpVecCastOp idxSlcTmpVecCastOp =
            indices.back().getDefiningOp<SlcTmpVecCastOp>()) {
      if (SlcTmpVecCastOp valSlcTmpVecCastOp =
              op.getValueToStore().getDefiningOp<SlcTmpVecCastOp>()) {
        Value idxVectorValue = idxSlcTmpVecCastOp.getIn();
        VectorType idxVectorType = cast<VectorType>(idxVectorValue.getType());
        Value valVectorValue = valSlcTmpVecCastOp.getIn();
        VectorType valVectorType = cast<VectorType>(valVectorValue.getType());
        Value memRef = op.getMemRef();
        SmallVector<Value> scalarArgs(indices.begin(), indices.end());
        scalarArgs.back() =
            rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
        SlcCallbackOp slcCallbackOp = cast<SlcCallbackOp>(op->getParentOp());
        Value mask = getMask(slcCallbackOp, rewriter);
        rewriter.setInsertionPoint(op.getOperation());
        vector::ScatterOp vectorScatterOp = rewriter.create<vector::ScatterOp>(
            op.getLoc(), memRef, scalarArgs, idxVectorValue, mask,
            valVectorValue);
        rewriter.eraseOp(op.getOperation());
        eraseInTmpCast({idxSlcTmpVecCastOp, valSlcTmpVecCastOp}, rewriter);
        return success();
      } else {
        return failure();
      }
    } else {
      return failure();
    }
  }
};

void CallbackVectorizer::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<VectorizeSlcFromStreamOp>(&getContext());
  patterns.add<VectorizeMemRefLoadOp>(&getContext());
  patterns.add<VectorizeArithAddFOp>(&getContext());
  patterns.add<VectorizeMemRefStoreOp>(&getContext());

  GreedyRewriteConfig config = GreedyRewriteConfig();
  config.setStrictness(GreedyRewriteStrictness::ExistingOps);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
}

std::unique_ptr<mlir::Pass> createCallbackVectorizerPass() {
  return std::make_unique<mlir::slc::CallbackVectorizer>();
}

} // namespace slc
} // namespace mlir
