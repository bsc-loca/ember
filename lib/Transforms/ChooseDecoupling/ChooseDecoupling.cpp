#include "lib/Transforms/ChooseDecoupling/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
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
#include "lib/Dialect/SLCVEC/SlcVecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace slc {
#define GEN_PASS_DEF_CHOOSEDECOUPLING
#include "lib/Transforms/ChooseDecoupling/Passes.h.inc"

struct ChooseDecoupling : public impl::ChooseDecouplingBase<ChooseDecoupling> {
  using ChooseDecouplingBase::ChooseDecouplingBase;

  ChooseDecoupling() {}

  void runOnOperation() override;
};

scf::ForOp getForOp(Block &block, const unsigned int id) {
  unsigned i = 0;
  for (scf::ForOp forOp : block.getOps<scf::ForOp>()) {
    if (i == id) {
      return forOp;
    }
    i++;
  }
  assert(false);
  return scf::ForOp();
}

bool isReadOnlyLoad(Operation *op) {
  if (memref::LoadOp loadOp = mlir::dyn_cast<memref::LoadOp>(op)) {
    Value memRef = loadOp.getMemRef();
    for (OpOperand &use : memRef.getUses()) {
      Operation *user = use.getOwner();
      if (mlir::isa<memref::StoreOp>(user)) {
        assert(use.getOperandNumber() == 1);
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

// Returns true iff there's at least a loadOp to a read-only location
bool shouldOffloadToSlc(scf::ForOp &forOp) {

  for (memref::LoadOp loadOp : forOp.getOps<memref::LoadOp>()) {
    if (isReadOnlyLoad(loadOp.getOperation())) {
      return true;
    }
  }
  return false;
}

bool hasOp(std::vector<Operation *> &callbackOps, Operation *op) {
  if (op != nullptr) {
    for (Operation *callbackOp : callbackOps) {
      if (op == callbackOp) {
        return true;
      }
    }
    return false;
  } else {
    return false;
  }
}

bool canOffloadOp(Operation &op, std::vector<Operation *> &fbegOps,
                  std::vector<Operation *> &fiteOps,
                  std::vector<Operation *> &fendOps) {
  if (isReadOnlyLoad(&op) or mlir::isa<arith::AddIOp>(op)) {
    for (OpOperand &operand : op.getOpOperands()) {
      Operation *operandOp = operand.get().getDefiningOp();
      if (operandOp != nullptr) { // Value is an operation
        if ((operandOp->getParentOfType<SlcCallbackOp>() != nullptr) or
            hasOp(fbegOps, operandOp) or hasOp(fiteOps, operandOp) or
            hasOp(fendOps, operandOp)) {
          return false;
        } else { // Value is an argument
          // Nothing to do
        }
      }
    }
    return true;
  } else {
    return false;
  }
}

bool canOffloadOp(Operation &op, std::vector<Operation *> &fbegOps,
                  std::vector<Operation *> &fendOps) {
  std::vector<Operation *> fiteOps;
  return canOffloadOp(op, fbegOps, fiteOps, fendOps);
}

bool canOffloadOp(Operation &op, std::vector<Operation *> &fiteOps) {
  std::vector<Operation *> fbegOps;
  std::vector<Operation *> fendOps;
  return canOffloadOp(op, fbegOps, fiteOps, fendOps);
}

void recursiveTraversal(mlir::Region &region, OpBuilder &builder) {

  // Only regions with one block
  if (region.getBlocks().size() != 1) {
    return;
  }

  Block &block = region.getBlocks().front();

  // Look for ForOp offloading candidates
  std::vector<unsigned int> forOpsToOffload;
  {
    unsigned int forId = 0;
    for (scf::ForOp forOp : block.getOps<scf::ForOp>()) {
      if (shouldOffloadToSlc(forOp)) {
        mlir::Value iv = forOp.getInductionVar();
        llvm::errs() << "Found loop to offload. IndVar=";
        iv.print(llvm::errs());
        llvm::errs() << "\n";
        forOpsToOffload.push_back(forId);
      }
      forId++;
    }
  }

  if (forOpsToOffload.size() == 0) {
    if (scf::ForOp scfForOp =
            mlir::dyn_cast<scf::ForOp>(*region.getParentOp())) {
      // Insert fiber iter callback primitive after the accelerate for
      // primitive
      assert(mlir::isa<slc::SlcAccelerateForOp>(block.front()));
      builder.setInsertionPointAfter(&block.front());
      slc::SlcCallbackOp fiteCallbackOp = builder.create<slc::SlcCallbackOp>(
          block.front().getLoc(), TypeRange());

      std::vector<Operation *> opsToOffload;
      std::vector<Operation *> opsToMoveInFite;

      unsigned int state = 0;
      for (Operation &op : block.getOperations()) {
        if (state == 0) { // op == accelerateForOp
          assert(mlir::isa<slc::SlcAccelerateForOp>(op));
          state++;
        } else if (state == 1) { // op == fiteCallbackOp
          assert(mlir::isa<slc::SlcCallbackOp>(op));
          state++;
        } else if (state == 2) { // fiteCallbackOp < op < terminator
          if (op.hasTrait<OpTrait::IsTerminator>()) {
            state++;
          } else {
            if (canOffloadOp(op, opsToMoveInFite)) {
              opsToOffload.push_back(&op);
            } else {
              opsToMoveInFite.push_back(&op);
            }
          }
        } else {
          assert(false);
        }
      }
      assert(state == 3);
      for (Operation *op : opsToOffload) {
        op->moveBefore(fiteCallbackOp.getOperation());
      }

      for (Operation *op : opsToMoveInFite) {
        op->moveBefore(fiteCallbackOp.getBody()->getTerminator());
      }

      // Check vectorization opportunity
      // i.e. all uses of induction variable are last idx of ld/st ops
      bool canVectorize = true;
      for (OpOperand &use : scfForOp.getInductionVar().getUses()) {
        if (memref::LoadOp loadOp =
                mlir::dyn_cast<memref::LoadOp>(use.getOwner())) {
          if (use.get() != loadOp.getIndices().back()) {
            canVectorize = false;
            break;
          }
        } else if (memref::StoreOp storeOp =
                       mlir::dyn_cast<memref::StoreOp>(use.getOwner())) {
          if (use.get() != storeOp.getIndices().back()) {
            canVectorize = false;
            break;
          }
        } else {
          canVectorize = false;
          break;
        }
      }

      if (canVectorize) {
        builder.setInsertionPointAfter(&block.front());
        builder.create<slcvec::SlcVectorizeForOp>(scfForOp->getLoc());
      }
    }
  } else if (forOpsToOffload.size() == 1) {
    const unsigned int forOpId = forOpsToOffload.front();

    // Insert accelerate for primitive
    builder.setInsertionPointToStart(getForOp(block, forOpId).getBody());
    slc::SlcAccelerateForOp accelerateForOp =
        builder.create<slc::SlcAccelerateForOp>(
            getForOp(block, forOpId).getLoc());

    if (mlir::isa<scf::ForOp>(getForOp(block, forOpId)->getParentOp())) {
      // Insert fiber begin callback primitive after the accelerate for
      // primitive
      assert(mlir::isa<slc::SlcAccelerateForOp>(block.front()));
      builder.setInsertionPointAfter(&block.front());
      slc::SlcCallbackOp fbegCallbackOp = builder.create<slc::SlcCallbackOp>(
          getForOp(block, forOpId).getLoc(), TypeRange());

      // Insert fiber end callback primitive
      builder.setInsertionPointAfter(getForOp(block, forOpId).getOperation());
      slc::SlcCallbackOp fendCallbackOp = builder.create<slc::SlcCallbackOp>(
          getForOp(block, forOpId).getLoc(), TypeRange());

      std::vector<Operation *> opsToOffload;
      std::vector<Operation *> opsToMoveInFbeg;
      std::vector<Operation *> opsToMoveInFend;

      unsigned int state = 0;
      for (Operation &op : block.getOperations()) {
        if (state == 0) { // (op == accelerateForOp) + op == fbegCallbackOp
          if (mlir::isa<slc::SlcAccelerateForOp>(op)) {
            state = 1;
          } else if (mlir::isa<slc::SlcCallbackOp>(op)) {
            state = 2;
          } else {
            assert(false);
          }
        } else if (state == 1) { // op == fbegCallbackOp
          assert(mlir::isa<slc::SlcCallbackOp>(op));
          state++;
        } else if (state == 2) { // fbegCallbackOp < op < forOp
          if (&op == getForOp(block, forOpId).getOperation()) {
            state++;
          } else {
            if (canOffloadOp(op, opsToMoveInFbeg, opsToMoveInFend)) {
              opsToOffload.push_back(&op);
            } else {
              opsToMoveInFbeg.push_back(&op);
            }
          }
        } else if (state == 3) { // op == fendCallbackOp
          assert(mlir::isa<slc::SlcCallbackOp>(op));
          state++;
        } else if (state == 4) { // fendCallbackOp < op < terminator
          if (op.hasTrait<OpTrait::IsTerminator>()) {
            state++;
          } else {
            if (canOffloadOp(op, opsToMoveInFbeg, opsToMoveInFend)) {
              opsToOffload.push_back(&op);
            } else {
              opsToMoveInFend.push_back(&op);
            }
          }
        } else {
          assert(false);
        }
      }
      assert(state == 5);

      for (Operation *op : opsToOffload) {
        op->moveBefore(fbegCallbackOp.getOperation());
      }

      for (Operation *op : opsToMoveInFbeg) {
        op->moveBefore(fbegCallbackOp.getBody()->getTerminator());
      }

      for (Operation *op : opsToMoveInFend) {
        op->moveBefore(fendCallbackOp.getBody()->getTerminator());
      }

      if (opsToMoveInFend.empty()) {
        fendCallbackOp->erase();
      }

      if (opsToMoveInFbeg.empty()) {
        fbegCallbackOp->erase();
      }
    }

    recursiveTraversal(getForOp(block, forOpId).getRegion(), builder);
  } else {
    assert(false);
  }
}

void ChooseDecoupling::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  OpBuilder builder(&getContext());

  recursiveTraversal(funcOp.getFunctionBody(), builder);

  /*
    auto moduleOp = getOperation();
    moduleOp.walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *, LogicalResult>(op).Case<sam::SamArrayVal>(
          [&](auto op) {
            return processArrayVal(op, streamShape, enableBlockSparse);
          });
    });
  */
}

std::unique_ptr<mlir::Pass> createChooseDecouplingPass() {
  return std::make_unique<mlir::slc::ChooseDecoupling>();
}

/*
void registerChooseDecouplingPipeline() {
  PassPipelineRegistration<ChooseDecouplingOptions>(
      "choose-decoupling", "The ChooseDecoupling pipeline",
      [&](OpPassManager &pm,
          const ChooseDecouplingPipelineOptions &options) {
        pm.addPass(createChooseDecoupling());
      });
}
*/

} // namespace slc
} // namespace mlir
