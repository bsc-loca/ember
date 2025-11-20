#include "lib/Transforms/SimplifyMemOps/Passes.h"
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

#define GEN_PASS_DEF_SIMPLIFYMEMOPS
#include "lib/Transforms/SimplifyMemOps/Passes.h.inc"

struct SimplifyMemOps : public impl::SimplifyMemOpsBase<SimplifyMemOps> {
  using SimplifyMemOpsBase::SimplifyMemOpsBase;

  SimplifyMemOps() {}

  void runOnOperation() override;
};

struct SimplifyGatherOp : public OpRewritePattern<vector::GatherOp> {
  SimplifyGatherOp(mlir::MLIRContext *context)
      : OpRewritePattern<vector::GatherOp>(context) {}

  LogicalResult matchAndRewrite(vector::GatherOp op,
                                PatternRewriter &rewriter) const override {
    if (SlcVecIndVarOp slcVecIndVarOp =
            op.getIndexVec().getDefiningOp<SlcVecIndVarOp>()) {
      SmallVector<Value> newIndices = op.getIndices();
      newIndices.back() = slcVecIndVarOp.getIndVar();
      vector::MaskedLoadOp loadOp = rewriter.create<vector::MaskedLoadOp>(
          op.getLoc(), op.getResult().getType(), op.getBase(), newIndices,
          op.getMask(), op.getPassThru());
      rewriter.replaceOp(op.getOperation(), loadOp.getOperation());
      return success();
    }
    return failure();
  }
};

struct SimplifyScatterOp : public OpRewritePattern<vector::ScatterOp> {
  SimplifyScatterOp(mlir::MLIRContext *context)
      : OpRewritePattern<vector::ScatterOp>(context) {}

  LogicalResult matchAndRewrite(vector::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    if (SlcVecIndVarOp slcVecIndVarOp =
            op.getIndexVec().getDefiningOp<SlcVecIndVarOp>()) {
      SmallVector<Value> newIndices = op.getIndices();
      newIndices.back() = slcVecIndVarOp.getIndVar();
      vector::MaskedStoreOp storeOp = rewriter.create<vector::MaskedStoreOp>(
          op.getLoc(), op.getBase(), newIndices, op.getMask(),
          op.getValueToStore());
      rewriter.replaceOp(op.getOperation(), storeOp.getOperation());
      return success();
    }
    return failure();
  }
};

struct SimplifySlcVecIndVarOp : public OpRewritePattern<SlcVecIndVarOp> {
  SimplifySlcVecIndVarOp(mlir::MLIRContext *context)
      : OpRewritePattern<SlcVecIndVarOp>(context) {}

  LogicalResult matchAndRewrite(SlcVecIndVarOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getUses().empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

void SimplifyMemOps::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<SimplifyGatherOp>(&getContext());
  patterns.add<SimplifyScatterOp>(&getContext());
  patterns.add<SimplifySlcVecIndVarOp>(&getContext());

  GreedyRewriteConfig config = GreedyRewriteConfig();
  config.setStrictness(GreedyRewriteStrictness::ExistingOps);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
}

std::unique_ptr<mlir::Pass> createSimplifyMemOpsPass() {
  return std::make_unique<mlir::slc::SimplifyMemOps>();
}

} // namespace slc
} // namespace mlir
