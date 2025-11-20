#include "lib/Transforms/BufferCompoundTypes/Passes.h"
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
#include "lib/Dialect/SLCVEC/SlcVecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace slc {

using namespace slcvec;

#define GEN_PASS_DEF_BUFFERCOMPOUNDTYPES
#include "lib/Transforms/BufferCompoundTypes/Passes.h.inc"

struct BufferCompoundTypes
    : public impl::BufferCompoundTypesBase<BufferCompoundTypes> {
  using BufferCompoundTypesBase::BufferCompoundTypesBase;

  BufferCompoundTypes() {}

  void runOnOperation() override;
};

void BufferCompoundTypes::runOnOperation() {

  IRRewriter rewriter(getOperation()->getContext());

  getOperation()->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](slcvec::SlcVecForOp slcVecForOp) {
        // Can only buffer an inner loop
        if (slcVecForOp.getOps<SlcForOp>().empty()) {
          // Check if loop bounds are indexes in FuncOp
          Value lowerBound = slcVecForOp.getLowerBound();
          Value upperBound = slcVecForOp.getUpperBound();
          Value step = slcVecForOp.getStep();
          bool lbIsIdxInFunc = mlir::isa<IndexType>(lowerBound.getType()) and
                               mlir::isa<func::FuncOp>(
                                   lowerBound.getParentBlock()->getParentOp());
          bool ubIsIdxInFunc = mlir::isa<IndexType>(upperBound.getType()) and
                               mlir::isa<func::FuncOp>(
                                   upperBound.getParentBlock()->getParentOp());
          bool stIsIdxInFunc = mlir::isa<IndexType>(upperBound.getType()) and
                               mlir::isa<func::FuncOp>(
                                   upperBound.getParentBlock()->getParentOp());
          if (lbIsIdxInFunc and ubIsIdxInFunc and stIsIdxInFunc) {
            Value indVarSlcStream = slcVecForOp.getInductionVar();
            // Get the FITE callback within that loop
            if (SlcCallbackOp slcCallbackOp = dyn_cast<SlcCallbackOp>(
                    *slcVecForOp.getBody()->getTerminator()->getPrevNode())) {

              // Look for SlcFromStreamOps of SlcVecMemStreamOps from the same
              // loop
              SlcFromStreamOp indVarFromStreamOp;
              SlcFromStreamOp maskFromStreamOp;
              std::vector<SlcFromStreamOp> slcFromStreamOpsToBuffer;
              for (SlcFromStreamOp slcFromStreamOp :
                   slcCallbackOp.getBody()->getOps<SlcFromStreamOp>()) {
                bool isOp =
                    slcFromStreamOp.getStream().getDefiningOp() != nullptr;
                bool sameLoop =
                    slcFromStreamOp->getParentOfType<SlcVecForOp>() ==
                    slcFromStreamOp.getStream()
                        .getParentRegion()
                        ->getParentOfType<SlcVecForOp>();
                // If from the same loop
                if (isOp and sameLoop) {
                  slcFromStreamOpsToBuffer.push_back(slcFromStreamOp);
                }

                if (not isOp and sameLoop) {
                  if (BlockArgument blockArg = dyn_cast<BlockArgument>(
                          slcFromStreamOp.getStream())) {
                    switch (blockArg.getArgNumber()) {
                    case 0:
                      indVarFromStreamOp = slcFromStreamOp;
                      break;
                    case 1:
                      maskFromStreamOp = slcFromStreamOp;
                      break;
                    }
                  }
                }
              }

              // Insert scf::ForOp in callback
              rewriter.setInsertionPointToStart(slcCallbackOp.getBody());
              scf::ForOp scfForOp = rewriter.create<scf::ForOp>(
                  slcCallbackOp.getLoc(), lowerBound, upperBound, step);
              for (SlcFromStreamOp slcFromStreamOp : slcFromStreamOpsToBuffer) {
                // Insert SlcBufStreamOp before the SlcForOp
                rewriter.setInsertionPoint(slcVecForOp.getOperation());
                Type elementType = slcFromStreamOp.getValue().getType();
                if (VectorType vectorType = dyn_cast<VectorType>(elementType)) {
                  elementType = vectorType.getElementType();
                }
                SlcVecBufStreamOp slcVecBufStreamOp =
                    rewriter.create<SlcVecBufStreamOp>(
                        slcVecForOp->getLoc(), ArrayRef(ShapedType::kDynamic),
                        elementType);
                // Insert SlcToBufferOps before the SlcCallbackOp
                rewriter.setInsertionPoint(slcCallbackOp.getOperation());
                TypedValue<SlcStreamType> slcMemStreamOp =
                    slcFromStreamOp.getStream();
                rewriter.create<SlcVecToBufferOp>(
                    slcVecForOp->getLoc(), slcMemStreamOp, slcVecBufStreamOp,
                    ValueRange(indVarSlcStream));
                rewriter.setInsertionPoint(scfForOp.getOperation());
                // Insert SlcFromStreamOp before the ForOp to load the buffer
                SlcFromStreamOp newSlcFromStreamOp =
                    rewriter.create<SlcFromStreamOp>(slcFromStreamOp.getLoc(),
                                                     slcVecBufStreamOp);
                rewriter.setInsertionPoint(slcFromStreamOp.getOperation());
                SlcVecFromBufferOp slcFromBufferOp =
                    rewriter.create<SlcVecFromBufferOp>(
                        slcFromStreamOp.getLoc(),
                        slcMemStreamOp.getType().getElementType(),
                        newSlcFromStreamOp.getValue(),
                        ValueRange(scfForOp.getInductionVar()));
                slcFromStreamOp->replaceAllUsesWith(slcFromBufferOp);
                slcFromStreamOp.erase();
              }

              // Insert SlcToIte an SlcToMask
              rewriter.setInsertionPointToStart(scfForOp.getBody());
              VectorType vecIndVarType =
                  cast<VectorType>(indVarFromStreamOp.getType());
              SlcVecIndVarOp slcVecIndVarOp = rewriter.create<SlcVecIndVarOp>(
                  scfForOp.getLoc(), vecIndVarType, scfForOp.getInductionVar(),
                  scfForOp.getLowerBound(), scfForOp.getUpperBound(),
                  scfForOp.getStep());
              indVarFromStreamOp.replaceAllUsesWith(
                  slcVecIndVarOp.getVecIndVar());
              indVarFromStreamOp.erase();

              Type maskType = VectorType::get(vecIndVarType.getShape(),
                                              rewriter.getI1Type());
              SlcVecMaskOp slcMaskOp = rewriter.create<SlcVecMaskOp>(
                  scfForOp.getLoc(), maskType, scfForOp.getInductionVar(),
                  scfForOp.getLowerBound(), scfForOp.getUpperBound(),
                  scfForOp.getStep());
              maskFromStreamOp.replaceAllUsesWith(slcMaskOp.getMask());
              maskFromStreamOp.erase();

              // Move all ops into scf::ForOp
              Operation *currOp = scfForOp->getNextNode();
              while (currOp != slcCallbackOp.getBody()->getTerminator()) {
                Operation *nextOp = currOp->getNextNode();
                if (SlcFromStreamOp slcFromStreamOp =
                        dyn_cast<SlcFromStreamOp>(*currOp)) {
                  if (slcFromStreamOp.getStream() ==
                      slcVecForOp.getInductionVar()) {
                    slcFromStreamOp.replaceAllUsesWith(
                        scfForOp.getInductionVar());
                    slcFromStreamOp.erase();
                  } else if (slcFromStreamOp->getParentOfType<SlcForOp>() !=
                             slcFromStreamOp.getStream()
                                 .getParentRegion()
                                 ->getParentOfType<SlcForOp>()) {
                    currOp->moveBefore(scfForOp.getOperation());
                  } else {
                    currOp->moveBefore(scfForOp.getBody()->getTerminator());
                  }
                } else {
                  currOp->moveBefore(scfForOp.getBody()->getTerminator());
                }
                currOp = nextOp;
              }

              // Move callback after SlcForOp
              slcCallbackOp->moveAfter(slcVecForOp);

              // TODO check there's not another callback already
            } else {
              assert(false);
            }
            return;
          }
        }
      });
}

std::unique_ptr<mlir::Pass> createBufferCompoundTypesPass() {
  return std::make_unique<mlir::slc::BufferCompoundTypes>();
}

} // namespace slc
} // namespace mlir
