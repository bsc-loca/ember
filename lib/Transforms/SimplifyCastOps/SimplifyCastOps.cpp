#include "lib/Transforms/SimplifyCastOps/Passes.h"
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

#define GEN_PASS_DEF_SIMPLIFYCASTOPS
#include "lib/Transforms/SimplifyCastOps/Passes.h.inc"

struct SimplifyCastOps : public impl::SimplifyCastOpsBase<SimplifyCastOps> {
  using SimplifyCastOpsBase::SimplifyCastOpsBase;

  SimplifyCastOps() {}

  void runOnOperation() override;
};

Value stripCasts(mlir::Value value) {
  if (SlcBroadcastStreamOp slcBroadcastStreamOp =
          value.getDefiningOp<SlcBroadcastStreamOp>()) {
    TypedValue<SlcStreamType> stream = slcBroadcastStreamOp.getStream();
    return stripCasts(stream);
  } else if (SlcToStreamOp slcToStreamOp =
                 value.getDefiningOp<SlcToStreamOp>()) {
    return slcToStreamOp.getValue();
  } else {
    return value;
  }
}

struct SimplifySlcForOp : public OpRewritePattern<SlcForOp> {
  SimplifySlcForOp(mlir::MLIRContext *context)
      : OpRewritePattern<SlcForOp>(context) {}

  LogicalResult matchAndRewrite(SlcForOp op,
                                PatternRewriter &rewriter) const override {

    Value lb = stripCasts(op.getLowerBound());
    Value ub = stripCasts(op.getUpperBound());
    Value step = stripCasts(op.getStep());

    if (lb.getType() != ub.getType() or ub.getType() != step.getType() or
        lb.getType() != step.getType()) {

      return failure();
    } else {
      Operation::operand_range init_args = op.getInitArgs();

      SlcForOp newSlcForOp =
          rewriter.create<SlcForOp>(op.getLoc(), lb, ub, step, init_args);

      rewriter.eraseBlock(newSlcForOp.getBody());
      rewriter.inlineRegionBefore(op.getRegion(), newSlcForOp.getRegion(),
                                  newSlcForOp.getRegion().end());

      rewriter.replaceOp(op, newSlcForOp);

      return success();
    }
  }
};

struct SimplifySlcVecForOp : public OpRewritePattern<SlcVecForOp> {
  SimplifySlcVecForOp(mlir::MLIRContext *context)
      : OpRewritePattern<SlcVecForOp>(context) {}

  LogicalResult matchAndRewrite(SlcVecForOp op,
                                PatternRewriter &rewriter) const override {

    Value lb = stripCasts(op.getLowerBound());
    Value ub = stripCasts(op.getUpperBound());
    Value step = stripCasts(op.getStep());

    if (lb.getType() != ub.getType() or ub.getType() != step.getType() or
        lb.getType() != step.getType()) {

      return failure();
    } else {
      Value mask = op.getInMask();
      long vectorLength = op.getVectorLength().getLimitedValue();
      LoopConfig loopConfig = op.getLoopConfig();
      Operation::operand_range init_args = op.getInitArgs();

      SlcVecForOp newSlcVecForOp = rewriter.create<SlcVecForOp>(
          op.getLoc(), lb, ub, step, mask, vectorLength, loopConfig, init_args);

      rewriter.eraseBlock(newSlcVecForOp.getBody());
      rewriter.inlineRegionBefore(op.getRegion(), newSlcVecForOp.getRegion(),
                                  newSlcVecForOp.getRegion().end());

      rewriter.replaceOp(op, newSlcVecForOp);

      return success();
    }
  }
};

struct SimplifySlcMemStreamOp : public OpRewritePattern<SlcMemStreamOp> {
  SimplifySlcMemStreamOp(mlir::MLIRContext *context)
      : OpRewritePattern<SlcMemStreamOp>(context) {}

  LogicalResult matchAndRewrite(SlcMemStreamOp op,
                                PatternRewriter &rewriter) const override {

    Type returnType = op.getType();
    Value memRef = op.getMemref();
    SmallVector<Value> indices;
    for (Value idx : op.getIndices()) {
      indices.push_back(stripCasts(idx));
    }

    SlcMemStreamOp newSlcMemStreamOp = rewriter.create<SlcMemStreamOp>(
        op.getLoc(), returnType, memRef, indices);

    rewriter.replaceOp(op, newSlcMemStreamOp);

    return success();
  }
};

struct SimplifySlcAluStreamOp : public OpRewritePattern<SlcAluStreamOp> {
  SimplifySlcAluStreamOp(mlir::MLIRContext *context)
      : OpRewritePattern<SlcAluStreamOp>(context) {}

  LogicalResult matchAndRewrite(SlcAluStreamOp op,
                                PatternRewriter &rewriter) const override {

    Value lhs = stripCasts(op.getLhs());
    Value rhs = stripCasts(op.getRhs());
    OpType aluOp = op.getAluOp();

    SlcAluStreamOp newSlcAluStreamOp =
        rewriter.create<SlcAluStreamOp>(op.getLoc(), lhs, rhs, aluOp);

    rewriter.replaceOp(op, newSlcAluStreamOp);

    return success();
  }
};

struct SimplifySlcVecMemStreamOp : public OpRewritePattern<SlcVecMemStreamOp> {
  SimplifySlcVecMemStreamOp(mlir::MLIRContext *context)
      : OpRewritePattern<SlcVecMemStreamOp>(context) {}

  LogicalResult matchAndRewrite(SlcVecMemStreamOp op,
                                PatternRewriter &rewriter) const override {

    Type returnType = op.getType();
    Value memRef = op.getMemref();
    SmallVector<Value> indices;
    for (Value idx : op.getIndices()) {
      indices.push_back(stripCasts(idx));
    }

    SlcVecMemStreamOp newSlcVecMemStreamOp = rewriter.create<SlcVecMemStreamOp>(
        op.getLoc(), returnType, memRef, indices);

    rewriter.replaceOp(op, newSlcVecMemStreamOp);

    return success();
  }
};

void SimplifyCastOps::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<SimplifySlcForOp>(&getContext());
  patterns.add<SimplifySlcMemStreamOp>(&getContext());
  patterns.add<SimplifySlcAluStreamOp>(&getContext());
  patterns.add<SimplifySlcVecForOp>(&getContext());
  patterns.add<SimplifySlcVecMemStreamOp>(&getContext());

  GreedyRewriteConfig config = GreedyRewriteConfig();
  config.setStrictness(GreedyRewriteStrictness::ExistingOps);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);

  IRRewriter rewriter(getOperation()->getContext());

  // Look for SlcBroadcastStreamOps
  getOperation()->walk([&](slc::SlcBroadcastStreamOp slcBroadcastStreamOp) {
    if (slcBroadcastStreamOp->getUses().empty()) {
      rewriter.eraseOp(slcBroadcastStreamOp.getOperation());
    }
  });

  // Look for SlcToStreamOps
  getOperation()->walk([&](slc::SlcToStreamOp slcToStreamOp) {
    if (slcToStreamOp->getUses().empty()) {
      rewriter.eraseOp(slcToStreamOp.getOperation());
    }
  });

  /*
      // Add ite arguments to the parent loop of transform candidates, replace
    the
      // slc.to_value uses with ite args, and erase slc.to_value ops
      std::vector<Operation *> yieldOpsToUpdate;
      for (Operation *transformCandidate : transformCandidates) {
        SlcFromStreamOp slcFromStreamOp =
    cast<SlcFromStreamOp>(transformCandidate); BlockArgument streamBlockArgument
    = mlir::dyn_cast<BlockArgument>(slcFromStreamOp.getStream()); SlcForOp
    parentSlcForOp = mlir::dyn_cast<SlcForOp>(
            streamBlockArgument.getParentBlock()->getParentOp());

        // Add the 0 value at the beginning of the block
        rewriter.setInsertionPointToStart(parentSlcForOp->getBlock());
        arith::ConstantIndexOp oneConstOp =
    rewriter.create<arith::ConstantIndexOp>( streamBlockArgument.getLoc(), 0);

        FailureOr<LoopLikeOpInterface> maybeNewParentSlcForOp =
            parentSlcForOp.replaceWithAdditionalIterOperands(
                rewriter, ValueRange(oneConstOp), false);
        assert(succeeded(maybeNewParentSlcForOp));
        SlcForOp newParentSlcForOp =
    mlir::cast<SlcForOp>(*maybeNewParentSlcForOp);
        yieldOpsToUpdate.push_back(newParentSlcForOp.getBody()->getTerminator());
        slcFromStreamOp.replaceAllUsesWith(
            newParentSlcForOp.getRegionIterArgs().back());
        slcFromStreamOp.erase();
      }

      // Add an FEND callbacks, or modify the current ones, to increment
    iteration
      // variables of modified loops
      for (Operation *op : yieldOpsToUpdate) {
        SlcYieldOp slcYieldOp = cast<SlcYieldOp>(op);
        // SlcForOp slcForOp = cast<SlcForOp>(slcYieldOp->getParentOp());

        if (SlcCallbackOp slcCallbackOp =
                mlir::dyn_cast<SlcCallbackOp>(slcYieldOp->getPrevNode())) {

          // Apply signature conversion to the body of the forOp. It has a
    single
          // block, with argument which is the induction variable. That has to
    be
          // replaced with the new induction variable.
          assert(slcCallbackOp->getNumOperands() == 0);

          rewriter.setInsertionPointAfter(slcCallbackOp);
          SlcCallbackOp newSlcCallbackOp = rewriter.create<SlcCallbackOp>(
              slcYieldOp.getLoc(), TypeRange(rewriter.getIndexType()));

          rewriter.eraseBlock(newSlcCallbackOp.getBody());
          rewriter.inlineRegionBefore(slcCallbackOp.getRegion(),
                                      newSlcCallbackOp.getRegion(),
                                      newSlcCallbackOp.getRegion().end());
          newSlcCallbackOp.getBody()->getTerminator()->erase();
          slcCallbackOp.erase();

          // Add increment operations
          rewriter.setInsertionPointToEnd(newSlcCallbackOp.getBody());
          arith::ConstantIndexOp constantIndexOp =
              rewriter.create<arith::ConstantIndexOp>(newSlcCallbackOp.getLoc(),
    1); arith::AddIOp addIOp = rewriter.create<arith::AddIOp>(
              newSlcCallbackOp.getLoc(), slcYieldOp.getResults().front(),
              constantIndexOp);

          // Modify the callback's slc.yield operation
          rewriter.create<SlcYieldOp>(newSlcCallbackOp->getLoc(),
                                      ValueRange(addIOp));

          // Modify the loop's slc.yield operation
          assert(slcYieldOp.getNumOperands() == 1);
          rewriter.modifyOpInPlace(slcYieldOp.getOperation(), [&] {
            slcYieldOp.setOperand(0, newSlcCallbackOp.getResult(0));
          });
        } else {
          // Create a new callback
          rewriter.setInsertionPointAfter(slcYieldOp->getPrevNode());
          SlcCallbackOp newSlcCallbackOp = rewriter.create<SlcCallbackOp>(
              slcYieldOp.getLoc(), TypeRange(rewriter.getIndexType()));

          // Add increment operations
          rewriter.setInsertionPointToStart(newSlcCallbackOp.getBody());
          arith::ConstantIndexOp constantIndexOp =
              rewriter.create<arith::ConstantIndexOp>(newSlcCallbackOp.getLoc(),
    1); arith::AddIOp addIOp = rewriter.create<arith::AddIOp>(
              newSlcCallbackOp.getLoc(), slcYieldOp.getResults().front(),
              constantIndexOp);

          // Modify the callback's slc.yield operation
          assert(newSlcCallbackOp.getBody()->getTerminator()->getNumOperands()
    == 0); rewriter.replaceOpWithNewOp<SlcYieldOp>(
              newSlcCallbackOp.getBody()->getTerminator(), ValueRange(addIOp));

          // Modify the loop's slc.yield operation
          assert(slcYieldOp.getNumOperands() == 1);
          rewriter.modifyOpInPlace(slcYieldOp.getOperation(), [&] {
            slcYieldOp.setOperand(0, newSlcCallbackOp.getResult(0));
          });
        }
  */
}

std::unique_ptr<mlir::Pass> createSimplifyCastOpsPass() {
  return std::make_unique<mlir::slc::SimplifyCastOps>();
}

} // namespace slc
} // namespace mlir
