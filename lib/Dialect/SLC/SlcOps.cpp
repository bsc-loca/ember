#include "lib/Dialect/SLC/SlcOps.h"

using namespace mlir;
using namespace mlir::slc;

//===----------------------------------------------------------------------===//
// SlcForOp
//===----------------------------------------------------------------------===//

MutableArrayRef<OpOperand> SlcForOp::getInitsMutable() {
  return getInitArgsMutable();
}

std::optional<ResultRange> SlcForOp::getLoopResults() { return getResults(); }

Block::BlockArgListType SlcForOp::getRegionIterArgs() {
  return getBody()->getArguments().drop_front(getNumInductionVars());
}
/*
std::optional<Value> SlcForOp::getSingleInductionVar() {
  return getInductionVar();
}

std::optional<OpFoldResult> SlcForOp::getSingleLowerBound() {
  return OpFoldResult(getLowerBound());
}

std::optional<OpFoldResult> SlcForOp::getSingleStep() {
  return OpFoldResult(getStep());
}

std::optional<OpFoldResult> SlcForOp::getSingleUpperBound() {
  return OpFoldResult(getUpperBound());
}
*/
std::optional<MutableArrayRef<OpOperand>> SlcForOp::getYieldedValuesMutable() {
  return cast<slc::SlcYieldOp>(getBody()->getTerminator()).getResultsMutable();
}

FailureOr<LoopLikeOpInterface> SlcForOp::replaceWithAdditionalYields(
    RewriterBase &rewriter, ValueRange newInitOperands,
    bool replaceInitOperandUsesInLoop,
    const NewYieldValuesFn &newYieldValuesFn) {
  // Create a new loop before the existing one, with the extra operands.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(getOperation());
  auto inits = llvm::to_vector(getInitArgs());
  inits.append(newInitOperands.begin(), newInitOperands.end());
  SlcForOp newLoop = rewriter.create<SlcForOp>(
      getLoc(), getLowerBound(), getUpperBound(), getStep(), inits,
      [](OpBuilder &, Location, Value, ValueRange) {});

  // Generate the new yield values and append them to the scf.yield operation.
  auto yieldOp = cast<SlcYieldOp>(getBody()->getTerminator());
  ArrayRef<BlockArgument> newIterArgs =
      newLoop.getBody()->getArguments().take_back(newInitOperands.size());
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(yieldOp);
    SmallVector<Value> newYieldedValues =
        newYieldValuesFn(rewriter, getLoc(), newIterArgs);
    assert(newInitOperands.size() == newYieldedValues.size() &&
           "expected as many new yield values as new iter operands");
    rewriter.modifyOpInPlace(yieldOp, [&]() {
      yieldOp.getResultsMutable().append(newYieldedValues);
    });
  }

  // Move the loop body to the new op.
  rewriter.mergeBlocks(getBody(), newLoop.getBody(),
                       newLoop.getBody()->getArguments().take_front(
                           getBody()->getNumArguments()));

  if (replaceInitOperandUsesInLoop) {
    // Replace all uses of `newInitOperands` with the corresponding basic block
    // arguments.
    for (auto it : llvm::zip(newInitOperands, newIterArgs)) {
      rewriter.replaceUsesWithIf(std::get<0>(it), std::get<1>(it),
                                 [&](OpOperand &use) {
                                   Operation *user = use.getOwner();
                                   return newLoop->isProperAncestor(user);
                                 });
    }
  }

  // Replace the old loop.
  rewriter.replaceOp(getOperation(),
                     newLoop->getResults().take_front(getNumResults()));
  return cast<LoopLikeOpInterface>(newLoop.getOperation());
}

SmallVector<Region *> SlcForOp::getLoopRegions() { return {&getRegion()}; }

void SlcForOp::build(OpBuilder &builder, OperationState &result, Value lb,
                     Value ub, Value step, ValueRange iterArgs,
                     BodyBuilderFn bodyBuilder) {
  result.addOperands({lb, ub, step});
  result.addOperands(iterArgs);

  for (Value v : iterArgs)
    result.addTypes(v.getType());
  Type t = lb.getType();
  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  bodyBlock.addArgument(t, result.location);
  for (Value v : iterArgs)
    bodyBlock.addArgument(v.getType(), v.getLoc());

  // Create the default terminator if the builder is not provided and if the
  // iteration arguments are not provided. Otherwise, leave this to the caller
  // because we don't know which values to return from the loop.
  if (iterArgs.empty() && !bodyBuilder) {
    SlcForOp::ensureTerminator(*bodyRegion, builder, result.location);
  } else if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock.getArgument(0),
                bodyBlock.getArguments().drop_front());
  }
}

LogicalResult SlcForOp::verify() {
  // Check that the number of init args and op results is the same.
  if (getInitArgs().size() != getNumResults())
    return emitOpError(
        "mismatch in number of loop-carried values and defined values");

  return success();
}

LogicalResult SlcForOp::verifyRegions() {
  Type lbType = getLowerBound().getType();
  Type ubType = getUpperBound().getType();
  Type stepType = getStep().getType();
  if (lbType != ubType or ubType != stepType or lbType != stepType) {
    return emitOpError("expected induction variable to have same element "
                       "type as bounds and step");
  }

  if (getNumRegionIterArgs() != getNumResults())
    return emitOpError(
        "mismatch in number of basic block args and defined values");

  auto initArgs = getInitArgs();
  auto iterArgs = getRegionIterArgs();
  auto opResults = getResults();
  unsigned i = 0;
  for (auto e : llvm::zip(initArgs, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return emitOpError() << "types mismatch between " << i
                           << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return emitOpError() << "types mismatch between " << i
                           << "th iter region arg and defined value";

    ++i;
  }
  return success();
}

/// Prints the initialization list in the form of
///   <prefix>(%inner = %outer, %inner2 = %outer2, <...>)
/// where 'inner' values are assumed to be region arguments and 'outer' values
/// are regular SSA values.
static void SlcForOpPrintInitializationList(OpAsmPrinter &p,
                                            Block::BlockArgListType blocksArgs,
                                            ValueRange initializers,
                                            StringRef prefix = "") {
  assert(blocksArgs.size() == initializers.size() &&
         "expected same length of arguments and initializers");
  if (initializers.empty())
    return;

  p << prefix << '(';
  llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p, [&](auto it) {
    p << std::get<0>(it) << " = " << std::get<1>(it);
  });
  p << ")";
}

void SlcForOp::print(OpAsmPrinter &p) {
  p << " " << getInductionVar() << " = " << getLowerBound() << " to "
    << getUpperBound() << " step " << getStep();
  SlcForOpPrintInitializationList(p, getRegionIterArgs(), getInitArgs(),
                                  " init_args");
  if (!getInitArgs().empty())
    p << " -> (" << getInitArgs().getTypes() << ')';
  p << ' ';
  p << " : " << getLowerBound().getType() << " -> "
    << getInductionVar().getType() << ' ';
  p.printRegion(getRegion(), true, true);
  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult SlcForOp::parse(OpAsmParser &parser, OperationState &result) {
  // %3 = slc.for %arg5 = %c0 to %2 step %c1 init_args(%arg6 = %c0) -> (index)
  // : index -> index {
  auto &builder = parser.getBuilder();

  OpAsmParser::Argument inductionVariable;
  OpAsmParser::UnresolvedOperand lb, ub, step;

  Type lb_type = builder.getType<SlcStreamType>(builder.getIndexType());
  Type ub_type = builder.getType<SlcStreamType>(builder.getIndexType());
  Type step_type = builder.getType<SlcStreamType>(builder.getIndexType());
  Type indvar_type = builder.getType<SlcStreamType>(builder.getIndexType());

  // Parse the induction variable followed by '=', loop bounds and step.
  if (parser.parseOperand(inductionVariable.ssaName) || parser.parseEqual() ||
      parser.parseOperand(lb) || parser.parseKeyword("to") ||
      parser.parseOperand(ub) || parser.parseKeyword("step") ||
      parser.parseOperand(step))
    return failure();

  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  regionArgs.push_back(inductionVariable);

  bool hasIterArgs = succeeded(parser.parseOptionalKeyword("init_args"));
  if (hasIterArgs) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return failure();
  }

  if (regionArgs.size() != result.types.size() + 1)
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");

  // Parse types
  if (parser.parseColon() || parser.parseType(lb_type) || parser.parseArrow() ||
      parser.parseType(ub_type))
    return failure();

  assert(lb_type == builder.getType<SlcStreamType>(builder.getIndexType()) or
         lb_type == builder.getIndexType());
  assert(ub_type == builder.getType<SlcStreamType>(builder.getIndexType()) or
         ub_type == builder.getIndexType());

  // Resolve input operands.
  regionArgs.front().type = indvar_type;
  if (parser.resolveOperand(lb, lb_type, result.operands) ||
      parser.resolveOperand(ub, ub_type, result.operands) ||
      parser.resolveOperand(step, step_type, result.operands))
    return failure();
  if (hasIterArgs) {
    for (auto argOperandType :
         llvm::zip(llvm::drop_begin(regionArgs), operands, result.types)) {
      Type type = std::get<2>(argOperandType);
      std::get<0>(argOperandType).type = type;
      if (parser.resolveOperand(std::get<1>(argOperandType), type,
                                result.operands))
        return failure();
    }
  }

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  SlcForOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// SlcCallbackOp
//===----------------------------------------------------------------------===//

void SlcCallbackOp::build(OpBuilder &builder, OperationState &result,
                          TypeRange resultTypes) {
  result.addTypes(resultTypes);

  Block *block = new Block();
  Region *region = result.addRegion();
  region->push_back(block);
  ensureTerminator(*region, builder, result.location);
}