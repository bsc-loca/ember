#include "lib/Transforms/ReplaceToValOps/Passes.h"
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

#include "lib/Dialect/SLC/SlcOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace slc {
#define GEN_PASS_DEF_REPLACETOVALOPS
#include "lib/Transforms/ReplaceToValOps/Passes.h.inc"

struct ReplaceToValOps : public impl::ReplaceToValOpsBase<ReplaceToValOps> {
  using ReplaceToValOpsBase::ReplaceToValOpsBase;

  ReplaceToValOps() {}

  void runOnOperation() override;
};

void ReplaceToValOps::runOnOperation() {

  IRRewriter rewriter(getOperation()->getContext());

  // Look for transform candidates
  std::vector<slc::SlcFromStreamOp> transformCandidates;
  getOperation()->walk([&](slc::SlcFromStreamOp slcFromStreamOp) {
    if (BlockArgument streamBlockArgument =
            mlir::dyn_cast<BlockArgument>(slcFromStreamOp.getStream())) {
      if (streamBlockArgument.getArgNumber() == 0) {
        transformCandidates.push_back(slcFromStreamOp);
      }
    }
  });

  // Add ite arguments to the parent loop of transform candidates, replace the
  // slc.to_value uses with ite args, and erase slc.to_value ops
  std::vector<Operation *> yieldOpsToUpdate;
  for (slc::SlcFromStreamOp slcFromStreamOp : transformCandidates) {
    BlockArgument streamBlockArgument =
        mlir::dyn_cast<BlockArgument>(slcFromStreamOp.getStream());
    SlcForOp parentSlcForOp = mlir::dyn_cast<SlcForOp>(
        streamBlockArgument.getParentBlock()->getParentOp());

    // Add the 0 value at the beginning of the block
    rewriter.setInsertionPointToStart(parentSlcForOp->getBlock());
    arith::ConstantIndexOp oneConstOp = rewriter.create<arith::ConstantIndexOp>(
        streamBlockArgument.getLoc(), 0);

    FailureOr<LoopLikeOpInterface> maybeNewParentSlcForOp =
        parentSlcForOp.replaceWithAdditionalIterOperands(
            rewriter, ValueRange(oneConstOp), false);
    assert(succeeded(maybeNewParentSlcForOp));
    SlcForOp newParentSlcForOp = mlir::cast<SlcForOp>(*maybeNewParentSlcForOp);
    yieldOpsToUpdate.push_back(newParentSlcForOp.getBody()->getTerminator());
    slcFromStreamOp.replaceAllUsesWith(
        newParentSlcForOp.getRegionIterArgs().back());
    slcFromStreamOp.erase();
  }

  // Add an FEND callbacks, or modify the current ones, to increment iteration
  // variables of modified loops
  for (Operation *op : yieldOpsToUpdate) {
    SlcYieldOp slcYieldOp = cast<SlcYieldOp>(op);
    // SlcForOp slcForOp = cast<SlcForOp>(slcYieldOp->getParentOp());

    if (SlcCallbackOp slcCallbackOp =
            mlir::dyn_cast<SlcCallbackOp>(slcYieldOp->getPrevNode())) {

      // Apply signature conversion to the body of the forOp. It has a single
      // block, with argument which is the induction variable. That has to be
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
          rewriter.create<arith::ConstantIndexOp>(newSlcCallbackOp.getLoc(), 1);
      arith::AddIOp addIOp = rewriter.create<arith::AddIOp>(
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
          rewriter.create<arith::ConstantIndexOp>(newSlcCallbackOp.getLoc(), 1);
      arith::AddIOp addIOp = rewriter.create<arith::AddIOp>(
          newSlcCallbackOp.getLoc(), slcYieldOp.getResults().front(),
          constantIndexOp);

      // Modify the callback's slc.yield operation
      assert(newSlcCallbackOp.getBody()->getTerminator()->getNumOperands() ==
             0);
      rewriter.replaceOpWithNewOp<SlcYieldOp>(
          newSlcCallbackOp.getBody()->getTerminator(), ValueRange(addIOp));

      // Modify the loop's slc.yield operation
      assert(slcYieldOp.getNumOperands() == 1);
      rewriter.modifyOpInPlace(slcYieldOp.getOperation(), [&] {
        slcYieldOp.setOperand(0, newSlcCallbackOp.getResult(0));
      });
    }
  }
}

std::unique_ptr<mlir::Pass> createReplaceToValOpsPass() {
  return std::make_unique<mlir::slc::ReplaceToValOps>();
}

} // namespace slc
} // namespace mlir
