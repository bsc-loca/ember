#include "lib/Dialect/DLC/DlcOps.h"

using namespace mlir;
using namespace mlir::slc;
using namespace mlir::dlc;

//===----------------------------------------------------------------------===//
// DlcConfigOp
//===----------------------------------------------------------------------===//

void DlcConfigOp::build(OpBuilder &builder, OperationState &result,
                          TypeRange resultTypes) {
  result.addTypes(resultTypes);

  Block *block = new Block();
  Region *region = result.addRegion();
  region->push_back(block);
  ensureTerminator(*region, builder, result.location);
}

//===----------------------------------------------------------------------===//
// DlcComputeLoopOp
//===----------------------------------------------------------------------===//

/// Parse the case regions and values.
/*static ParseResult
parseSwitchCases(OpAsmParser &p, DenseI64ArrayAttr &cases,
                 SmallVectorImpl<std::unique_ptr<Region>> &caseRegions) {
  SmallVector<int64_t> caseValues;
  while (succeeded(p.parseOptionalKeyword("case"))) {
    int64_t value;
    Region &region = *caseRegions.emplace_back(std::make_unique<Region>());
    if (p.parseInteger(value) || p.parseRegion(region, {}))
      return failure();
    caseValues.push_back(value);
  }
  cases = p.getBuilder().getDenseI64ArrayAttr(caseValues);
  return success();
}

/// Print the case regions and values.
static void printSwitchCases(OpAsmPrinter &p, Operation *op,
                             DenseI64ArrayAttr cases, RegionRange caseRegions) {
  for (auto [value, region] : llvm::zip(cases.asArrayRef(), caseRegions)) {
    p.printNewline();
    p << "case " << value << ' ';
    p.printRegion(*region, false);
  }
}*/

/*
LogicalResult DlcComputeLoopOp::verify() {
  if (getCases().size() != getCaseRegions().size()) {
    return emitOpError("has ")
           << getCaseRegions().size() << " case regions but "
           << getCases().size() << " case values";
  }

  DenseSet<int64_t> valueSet;
  for (int64_t value : getCases())
    if (!valueSet.insert(value).second)
      return emitOpError("has duplicate case value: ") << value;
  auto verifyRegion = [&](Region &region, const Twine &name) -> LogicalResult {
    auto yield = dyn_cast<YieldOp>(region.front().back());
    if (!yield)
      return emitOpError("expected region to end with scf.yield, but got ")
             << region.front().back().getName();

    if (yield.getNumOperands() != getNumResults()) {
      return (emitOpError("expected each region to return ")
              << getNumResults() << " values, but " << name << " returns "
              << yield.getNumOperands())
                 .attachNote(yield.getLoc())
             << "see yield operation here";
    }
    for (auto [idx, result, operand] :
         llvm::zip(llvm::seq<unsigned>(0, getNumResults()), getResultTypes(),
                   yield.getOperandTypes())) {
      if (result == operand)
        continue;
      return (emitOpError("expected result #")
              << idx << " of each region to be " << result)
                 .attachNote(yield.getLoc())
             << name << " returns " << operand << " here";
    }
    return success();
  };

  if (failed(verifyRegion(getDefaultRegion(), "default region")))
    return failure();
  for (auto [idx, caseRegion] : llvm::enumerate(getCaseRegions()))
    if (failed(verifyRegion(caseRegion, "case region #" + Twine(idx))))
      return failure();
  return success();
}
*/

unsigned dlc::DlcComputeLoopOp::getNumCases() { return getCases().size(); }

// Block &sdlc::DlcComputeLoopOp::getDefaultBlock() {
//   return getDefaultRegion().front();
// }

Block &dlc::DlcComputeLoopOp::getCaseBlock(unsigned idx) {
  assert(idx < getNumCases() && "case index out-of-bounds");
  return getCaseRegions()[idx].front();
}

/*void DlcComputeLoopOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &successors) {
  // All regions branch back to the parent op.
  if (!point.isParent()) {
    successors.emplace_back(getResults());
    return;
  }

  llvm::append_range(successors, getRegions());
}

void DlcComputeLoopOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &successors) {
  FoldAdaptor adaptor(operands, *this);

  // If a constant was not provided, all regions are possible successors.
  auto arg = dyn_cast_or_null<IntegerAttr>(adaptor.getArg());
  if (!arg) {
    llvm::append_range(successors, getRegions());
    return;
  }

  // Otherwise, try to find a case with a matching value. If not, the
  // default region is the only successor.
  for (auto [caseValue, caseRegion] : llvm::zip(getCases(), getCaseRegions())) {
    if (caseValue == arg.getInt()) {
      successors.emplace_back(&caseRegion);
      return;
    }
  }
  successors.emplace_back(&getDefaultRegion());
}

void DlcComputeLoopOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands, SmallVectorImpl<InvocationBounds> &bounds) {
  auto operandValue = llvm::dyn_cast_or_null<IntegerAttr>(operands.front());
  if (!operandValue) {
    // All regions are invoked at most once.
    bounds.append(getNumRegions(), InvocationBounds(0, 1));
    return;
  }

  unsigned liveIndex = getNumRegions() - 1;
  const auto *it = llvm::find(getCases(), operandValue.getInt());
  if (it != getCases().end())
    liveIndex = std::distance(getCases().begin(), it);
  for (unsigned i = 0, e = getNumRegions(); i < e; ++i)
    bounds.emplace_back(0, i == liveIndex);
}

struct FoldConstantCase : OpRewritePattern<dlc::DlcComputeLoopOp> {
  using OpRewritePattern<dlc::DlcComputeLoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(dlc::DlcComputeLoopOp op,
                                PatternRewriter &rewriter) const override {
    // If `op.getArg()` is a constant, select the region that matches with
    // the constant value. Use the default region if no matche is found.
    std::optional<int64_t> maybeCst = getConstantIntValue(op.getArg());
    if (!maybeCst.has_value())
      return failure();
    int64_t cst = *maybeCst;
    int64_t caseIdx, e = op.getNumCases();
    for (caseIdx = 0; caseIdx < e; ++caseIdx) {
      if (cst == op.getCases()[caseIdx])
        break;
    }

    Region &r = (caseIdx < op.getNumCases()) ? op.getCaseRegions()[caseIdx]
                                             : op.getDefaultRegion();
    Block &source = r.front();
    Operation *terminator = source.getTerminator();
    SmallVector<Value> results = terminator->getOperands();

    rewriter.inlineBlockBefore(&source, op);
    rewriter.eraseOp(terminator);
    // Replace the operation with a potentially empty list of results.
    // Fold mechanism doesn't support the case where the result list is empty.
    rewriter.replaceOp(op, results);

    return success();
  }
};

void DlcComputeLoopOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.add<FoldConstantCase>(context);
}*/