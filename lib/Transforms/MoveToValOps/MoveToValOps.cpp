#include "lib/Transforms/MoveToValOps/Passes.h"
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
#define GEN_PASS_DEF_MOVETOVALOPS
#include "lib/Transforms/MoveToValOps/Passes.h.inc"

struct MoveToValOps : public impl::MoveToValOpsBase<MoveToValOps> {
  using MoveToValOpsBase::MoveToValOpsBase;

  MoveToValOps() {}

  void runOnOperation() override;
};

void MoveToValOps::runOnOperation() {

  getOperation()->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](slc::SlcFromStreamOp slcFromStreamOp) {
        std::map<std::pair<Operation *, Operation *>, std::vector<OpOperand *>>
            usesInOpCallPairs;
        for (OpOperand &use : slcFromStreamOp->getUses()) {
          Operation *user = use.getOwner();
          Operation *parentOp = user->getParentOp();

          if (slc::SlcCallbackOp slcCallbackOp =
                  mlir::dyn_cast<slc::SlcCallbackOp>(parentOp)) {
            std::pair<Operation *, Operation *> key = std::make_pair(
                slcFromStreamOp.getOperation(), slcCallbackOp.getOperation());
            usesInOpCallPairs[key].push_back(&use);
          }
        }

        for (auto &it : usesInOpCallPairs) {
          const std::pair<Operation *, Operation *> opCallPair = it.first;
          std::vector<OpOperand *> uses = it.second;
          SlcFromStreamOp slcFromStreamOp =
              mlir::cast<SlcFromStreamOp>(opCallPair.first);
          SlcCallbackOp slcCallbackOp =
              mlir::cast<SlcCallbackOp>(opCallPair.second);

          slc::SlcFromStreamOp clonedSlcFromStreamOp =
              mlir::cast<slc::SlcFromStreamOp>(slcFromStreamOp->clone());
          slcCallbackOp->getRegion(0).getBlocks().front().push_front(
              clonedSlcFromStreamOp);
          for (OpOperand *use : uses) {
            use->assign(clonedSlcFromStreamOp);
          }

          if (slcFromStreamOp->getUses().empty()) {
            slcFromStreamOp->erase();
          }
        }
      });
}

std::unique_ptr<mlir::Pass> createMoveToValOpsPass() {
  return std::make_unique<mlir::slc::MoveToValOps>();
}

} // namespace slc
} // namespace mlir
