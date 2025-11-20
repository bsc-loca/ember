#include "lib/Conversion/SlcToDlc/SlcToDlc.h"

#include <memory>
#include <utility>

#include "lib/Dialect/SLC/SlcDialect.h"
#include "lib/Dialect/SLC/SlcOps.h"
#include "lib/Dialect/SLC/SlcTypes.h"
#include "lib/Dialect/SLCVEC/SlcVecDialect.h"
#include "lib/Dialect/SLCVEC/SlcVecOps.h"
#include "lib/Dialect/SLCVEC/SlcVecTypes.h"
#include "lib/Dialect/DLC/DlcDialect.h"
#include "lib/Dialect/DLC/DlcOps.h"
#include "lib/Dialect/DLC/DlcTypes.h"
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
#define GEN_PASS_DEF_SLCTODLC
#include "lib/Conversion/SlcToDlc/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::sparse_tensor;

class SlcToDlcPass : public mlir::impl::SlcToDlcBase<SlcToDlcPass> {
  void runOnOperation() override;

  /// Runs the convert slc-to-dlc pass on each function.
  LogicalResult runOnFunction(func::FuncOp func);
};

void decouple_vector_loop(slcvec::SlcVecForOp &forOp,
                          dlc::DlcConfigOp &configOp, OpBuilder &builder,
                          int depth,
                          std::map<int, slc::SlcCallbackOp> &callback_map) {
  // Insert a new TU declaration and indvar stream
  Operation &yieldOp = configOp->getRegion(0).getBlocks().front().back();
  builder.setInsertionPoint(&yieldOp);
  dlc::DlcTuOp tuOp =
      builder.create<dlc::DlcTuOp>(configOp->getLoc(), forOp.getLowerBound(),
                                   forOp.getUpperBound(), forOp.getStep());
  dlc::DlcGetIVOp getIVOp = builder.create<dlc::DlcGetIVOp>(
      configOp->getLoc(), forOp.getStep().getType(), tuOp);
  forOp.getInductionVar().replaceAllUsesWith(getIVOp);
  std::vector<Operation *> to_move;
  Operation *to_buffer = nullptr;
  for (Operation &op : *forOp->getRegion(0).begin()) {
    if (auto memStr = dyn_cast<slcvec::SlcVecMemStreamOp>(&op)) {
      to_move.push_back(memStr);
    } else if (auto toBufferStr = dyn_cast<slcvec::SlcVecToBufferOp>(&op)) {
      to_buffer = toBufferStr;
    } else {
      break;
    }
  }
  for (Operation *op : to_move) {
    op->moveBefore(&yieldOp);
  }
  if (to_buffer) {
    auto toBufferStr = dyn_cast<slcvec::SlcVecToBufferOp>(to_buffer);
    dlc::DlcRegisterOperandOp registerOperandOp =
        builder.create<dlc::DlcRegisterOperandOp>(configOp->getLoc(), tuOp,
                                                  dlc::OpType::end,
                                                  toBufferStr.getStream());
  }

  auto it = forOp->getRegion(0).begin()->begin();
  bool has_inner_loop = false;
  if (auto innerForOp = dyn_cast<slc::SlcForOp>(*it)) {
    assert(false);
  }
  if (auto innerForOp = dyn_cast<slcvec::SlcVecForOp>(*it)) {
    decouple_vector_loop(innerForOp, configOp, builder, depth + 1, callback_map);
    it++;
    has_inner_loop = true;
  }
  if (auto endCallback = dyn_cast<slc::SlcCallbackOp>(*it)) {
    uint64_t id = depth * 3 + (has_inner_loop ? 1 : 2);
    dlc::DlcRegisterCallbackOp registerCallbackOp =
        builder.create<dlc::DlcRegisterCallbackOp>(
            configOp->getLoc(), tuOp, dlc::OpType::end,
            id); // TODO fix here and below, should not always be end
    callback_map[id] = endCallback;
  }
}

void decouple_scalar_loop(slc::SlcForOp &forOp, dlc::DlcConfigOp &configOp,
                          OpBuilder &builder, int depth,
                          std::map<int, slc::SlcCallbackOp> &callback_map) {
  // Insert a new TU declaration and indvar stream
  Operation &yieldOp = configOp->getRegion(0).getBlocks().front().back();
  builder.setInsertionPoint(&yieldOp);
  dlc::DlcTuOp tuOp =
      builder.create<dlc::DlcTuOp>(configOp->getLoc(), forOp.getLowerBound(),
                                   forOp.getUpperBound(), forOp.getStep());
  dlc::DlcGetIVOp getIVOp = builder.create<dlc::DlcGetIVOp>(
      configOp->getLoc(), forOp.getStep().getType(), tuOp);
  forOp.getInductionVar().replaceAllUsesWith(getIVOp);
  std::vector<Operation *> to_move;
  for (Operation &op : *forOp->getRegion(0).begin()) {
    if (auto memStr = dyn_cast<slc::SlcMemStreamOp>(&op)) {
      to_move.push_back(memStr);
    } else if (auto aluStr = dyn_cast<slc::SlcAluStreamOp>(&op)) {
      to_move.push_back(aluStr);
    } else if (auto bufferStr = dyn_cast<slcvec::SlcVecBufStreamOp>(&op)) {

    } else {
      break;
    }
  }
  for (Operation *op : to_move) {
    op->moveBefore(&yieldOp);
  }

  auto it = forOp->getRegion(0).begin()->begin();
  if (auto bufferStr = dyn_cast<slcvec::SlcVecBufStreamOp>(*it)) {
    it++;
  }
  bool has_inner_loop = false;
  if (auto innerForOp = dyn_cast<slc::SlcForOp>(*it)) {
    decouple_scalar_loop(innerForOp, configOp, builder, depth + 1, callback_map);
    it++;
    has_inner_loop = true;
  }
  if (auto innerForOp = dyn_cast<slcvec::SlcVecForOp>(*it)) {
    decouple_vector_loop(innerForOp, configOp, builder, depth + 1, callback_map);
    it++;
    has_inner_loop = true;
  }
  if (auto endCallback = dyn_cast<slc::SlcCallbackOp>(*it)) {
    uint64_t id = depth * 3 + (has_inner_loop ? 1 : 2);
    dlc::DlcRegisterCallbackOp registerCallbackOp =
        builder.create<dlc::DlcRegisterCallbackOp>(configOp->getLoc(), tuOp,
                                                   dlc::OpType::end, id);
    callback_map[id] = endCallback;
  }
}

void eraseOpRecVec(slcvec::SlcVecForOp forOp) {
  std::vector<Operation*> to_del;
  for (Operation &op : llvm::reverse(forOp->getRegion(0).getBlocks().front())) {

    if (auto innerForOp = dyn_cast<slcvec::SlcVecForOp>(op)) {
      eraseOpRecVec(innerForOp);
    }
    to_del.push_back(&op);
  }

  for (Operation *op : to_del) {
    op->erase();
  }
  //forOp.erase();
}
void eraseOpRec(slc::SlcForOp forOp) {
  std::vector<Operation*> to_del;
  for (Operation &op : llvm::reverse(forOp->getRegion(0).getBlocks().front())) {

    if (auto innerForOp = dyn_cast<slc::SlcForOp>(op)) {
      eraseOpRec(innerForOp);
    }
    if (auto innerForOp = dyn_cast<slcvec::SlcVecForOp>(op)) {
      eraseOpRecVec(innerForOp);
    }
    to_del.push_back(&op);
  }

  for (Operation *op : to_del) {
    op->erase();
  }
  //forOp.erase();
}

void SlcToDlcPass::runOnOperation() {
  //MLIRContext *context = &getContext();
  func::FuncOp funcOp = getOperation();
  OpBuilder builder(&getContext());

  std::vector<slc::SlcForOp*> to_delete;
  // go through all outer SLC loops
  for (Operation &op : funcOp.getBody().front()) {
    if (auto forOp = dyn_cast<slc::SlcForOp>(&op)) {
      // Insert a DlcConfigOp
      builder.setInsertionPoint(forOp);
      dlc::DlcConfigOp configOp =
          builder.create<dlc::DlcConfigOp>(forOp.getLoc(), TypeRange());
      //dlc::DlcComputeLoopOp computeLoop = builder.create<dlc::DlcComputeLoopOp>();

      // Traverse inner SCF loops. Transform loops and streams and move them in
      // the config region. Move callbacks in compute region
      std::map<int,slc::SlcCallbackOp> callback_map;
      decouple_scalar_loop(forOp, configOp, builder, 0, callback_map);

      slc::SlcYieldOp slcYieldOp = slc::SlcYieldOp();
      SmallVector<int64_t> cases;
      for(std::pair<const int, slc::SlcCallbackOp> p : callback_map) {
        if (slc::SlcYieldOp op = dyn_cast<slc::SlcYieldOp>(
                p.second->getRegions().front().front().back())) {
          if (op->getNumOperands() == 1) {
            slcYieldOp = op;
          }
        }
        cases.push_back(p.first);
      }
      if (arith::AddIOp addi =
              llvm::dyn_cast<arith::AddIOp>(slcYieldOp.getOperand(0).getDefiningOp<arith::AddIOp>())) {
        mlir::Value operand = addi.getOperand(0);
        std::vector<mlir::OpOperand *> uses;
        for (mlir::OpOperand &use : operand.getUses()) {
          uses.push_back(&use);
        }
        for (auto use : uses) {
          Operation *user = use->getOwner();
          builder.setInsertionPoint(user);
          if (arith::AddIOp addi = llvm::dyn_cast<arith::AddIOp>(user)) {
            dlc::DlcIncrementVarOp incrementVarOp =
                builder.create<dlc::DlcIncrementVarOp>(forOp->getLoc());
          } else {
            dlc::DlcGetVarOp getVarOp = builder.create<dlc::DlcGetVarOp>(forOp->getLoc(), use->get().getType());
            user->setOperand(use->getOperandNumber(), getVarOp);
          }
        }
      }
      builder.setInsertionPoint(forOp);
      dlc::DlcComputeLoopOp computeLoop = builder.create<dlc::DlcComputeLoopOp>(
          configOp->getLoc(), slcYieldOp.getOperand(0).getType(), cases,
          cases.size());
      int i=0;
      for (std::pair<const int, slc::SlcCallbackOp> p : callback_map) {
        auto &srcRegion = p.second->getRegion(0);
        auto &dstRegion = computeLoop->getRegion(i++);
        dstRegion.getBlocks().splice(dstRegion.begin(), srcRegion.getBlocks());
        slc::SlcYieldOp slcYieldOp =
            dyn_cast<slc::SlcYieldOp>(dstRegion.front().back());
        builder.setInsertionPointAfter(slcYieldOp);
        dlc::DlcYieldOp dclYieldOp =
            builder.create<dlc::DlcYieldOp>(forOp.getLoc());
        if (slcYieldOp.getNumOperands() > 0) {
          if (arith::AddIOp addi = llvm::dyn_cast<arith::AddIOp>(
                  slcYieldOp.getOperand(0).getDefiningOp<arith::AddIOp>())) {
            slcYieldOp.erase();
            addi.erase();
          } else {
            slcYieldOp.erase();
          }
        } else {
          slcYieldOp.erase();
        }

        dstRegion.walk([&](slc::SlcFromStreamOp fromStreamOp) {
          // Create a replacement value (for example, a constant 0)
          OpBuilder builder(fromStreamOp);
          auto loc = fromStreamOp.getLoc();
          auto popOperandOp = builder.create<dlc::DlcPopOperandOp>(
              loc, fromStreamOp.getResult().getType());
          
          // Replace all uses of addOp with zero
          fromStreamOp.getResult().replaceAllUsesWith(popOperandOp);

          // Erase the old op
          fromStreamOp.erase();
        });
      }
      eraseOpRec(forOp);
      to_delete.push_back(&forOp);
      builder.setInsertionPointToStart(forOp.getBody());
      slc::SlcYieldOp sclYieldOp = builder.create<slc::SlcYieldOp>(
          forOp.getLoc());
    }
  }
  //to_delete.front()->erase();
}

std::unique_ptr<mlir::Pass> mlir::createSlcToDlcPass() {
  return std::make_unique<SlcToDlcPass>();
}