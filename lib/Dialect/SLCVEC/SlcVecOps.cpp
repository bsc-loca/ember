#include "lib/Dialect/SLCVEC/SlcVecOps.h"

using namespace mlir;
using namespace mlir::slc;
using namespace mlir::slcvec;

//===----------------------------------------------------------------------===//
// SlcVecForOp
//===----------------------------------------------------------------------===//

MutableArrayRef<OpOperand> SlcVecForOp::getInitsMutable() {
  return getInitArgsMutable();
}

std::optional<ResultRange> SlcVecForOp::getLoopResults() {
  return getResults();
}

Block::BlockArgListType SlcVecForOp::getRegionIterArgs() {
  return getBody()->getArguments().drop_front(getNumInductionVars());
}
/*
std::optional<Value> SlcVecForOp::getSingleInductionVar() {
  return std::nullopt;
}

std::optional<OpFoldResult> SlcVecForOp::getSingleLowerBound() {
  return OpFoldResult(getLowerBound());
}

std::optional<OpFoldResult> SlcVecForOp::getSingleStep() {
  return OpFoldResult(getStep());
}

std::optional<OpFoldResult> SlcVecForOp::getSingleUpperBound() {
  return OpFoldResult(getUpperBound());
}
*/

std::optional<MutableArrayRef<OpOperand>>
SlcVecForOp::getYieldedValuesMutable() {
  return cast<SlcVecYieldOp>(getBody()->getTerminator()).getResultsMutable();
}

FailureOr<LoopLikeOpInterface> SlcVecForOp::replaceWithAdditionalYields(
    RewriterBase &rewriter, ValueRange newInitOperands,
    bool replaceInitOperandUsesInLoop,
    const NewYieldValuesFn &newYieldValuesFn) {
  assert(false);
  // Create a new loop before the existing one, with the extra operands.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(getOperation());
  auto inits = llvm::to_vector(getInitArgs());
  inits.append(newInitOperands.begin(), newInitOperands.end());
  SlcVecForOp newLoop = rewriter.create<SlcVecForOp>(
      getLoc(), getLowerBound(), getUpperBound(), getStep(), getInMask(),
      getVectorLength().getLimitedValue(), getLoopConfig(), inits,
      [](OpBuilder &, Location, Value, ValueRange) {});

  // Generate the new yield values and append them to the scf.yield operation.
  SlcVecYieldOp yieldOp = cast<SlcVecYieldOp>(getBody()->getTerminator());
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

SmallVector<Region *> SlcVecForOp::getLoopRegions() { return {&getRegion()}; }

void SlcVecForOp::build(OpBuilder &builder, OperationState &result, Value lb,
                        Value ub, Value step, Value inMask, long vectorLength,
                        LoopConfig loopConfig, ValueRange iterArgs,
                        BodyBuilderFn bodyBuilder) {
  // Add operands and attributes
  result.addOperands({lb, ub, step, inMask});
  result.addOperands(iterArgs);
  result.addAttribute(getStaticVectorLengthAttrName(),
                      builder.getIndexAttr(vectorLength));
  result.addAttribute(getStaticLoopConfigAttrName(),
                      LoopConfigAttr::get(builder.getContext(), loopConfig));
  // Add return types
  for (Value v : iterArgs)
    result.addTypes(v.getType());

  // Add body
  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();

  // Add induction variable to body
  SlcStreamType indVarType = SlcStreamType::get(
      VectorType::get({vectorLength}, builder.getIndexType()));
  bodyBlock.addArgument(indVarType, result.location);

  // Add mask to body
  SlcStreamType maskType =
      SlcStreamType::get(VectorType::get({vectorLength}, builder.getI1Type()));
  bodyBlock.addArgument(maskType, result.location);

  // Add iterArgs to body
  for (Value v : iterArgs)
    bodyBlock.addArgument(v.getType(), v.getLoc());

  // Create the default terminator if the builder is not provided and if the
  // iteration arguments are not provided. Otherwise, leave this to the caller
  // because we don't know which values to return from the loop.
  if (iterArgs.empty() && !bodyBuilder) {
    SlcVecForOp::ensureTerminator(*bodyRegion, builder, result.location);
  } else if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock.getArgument(0),
                bodyBlock.getArguments().drop_front());
  }
}

LogicalResult SlcVecForOp::verify() {
  // Check that the number of init args and op results is the same.
  if (getInitArgs().size() != getNumResults())
    return emitOpError(
        "mismatch in number of loop-carried values and defined values");

  return success();
}

LogicalResult SlcVecForOp::verifyRegions() {
  // Check that the body defines as single block argument for the induction
  // variable.
  Type lbType = getLowerBound().getType();
  Type ubType = getUpperBound().getType();
  Type stepType = getStep().getType();
  if (lbType != ubType or ubType != stepType or lbType != stepType) {
    return emitOpError("expected bounds and step to have the same type");
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
static void SlcVecForOpPrintInitializationList(
    OpAsmPrinter &p, Block::BlockArgListType blocksArgs,
    ValueRange initializers, StringRef prefix = "") {
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

void SlcVecForOp::print(OpAsmPrinter &p) {
  p << " <" << stringifyLoopConfig(getLoopConfig()) << "x" << getVectorLength()
    << "> (" << getInductionVar() << " && " << getOutMask()
    << ") = " << getLowerBound() << " to " << getUpperBound() << " step "
    << getStep() << " mask " << getInMask();
  SlcVecForOpPrintInitializationList(p, getRegionIterArgs(), getInitArgs(),
                                     " init_args");
  if (!getInitArgs().empty())
    p << " -> (" << getInitArgs().getTypes() << ')';
  p << ' ';
  p << " : (" << getLowerBound().getType() << " && " << getInMask().getType()
    << ") -> (" << getInductionVar().getType() << " && "
    << getOutMask().getType() << ") ";
  p.printRegion(getRegion(), true, true);
  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult SlcVecForOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  OpAsmParser::Argument inductionVariable, mask_out;
  OpAsmParser::UnresolvedOperand lb, ub, step, mask_in;

  // Parse vector length
  IntegerAttr vectorLength;
  if (parser.parseLess() ||
      parser.parseAttribute(vectorLength, getStaticVectorLengthAttrName(),
                            result.attributes) ||
      parser.parseGreater())
    return failure();

  ArrayRef<int64_t> shape =
      ArrayRef<int64_t>(vectorLength.getValue().getLimitedValue());
  Type indvar_type =
      SlcStreamType::get(VectorType::get(shape, builder.getIndexType()));
  Type bound_type, in_mask_type, ind_type, out_mask_type;
  // Parse the induction variable followed by '=', bounds, step, and their type.
  if (parser.parseLParen() || parser.parseOperand(inductionVariable.ssaName) ||
      parser.parseKeyword("xx") || parser.parseOperand(mask_out.ssaName) ||
      parser.parseRParen() || parser.parseEqual() ||
      // Parse loop bounds.
      parser.parseOperand(lb) || parser.parseKeyword("to") ||
      parser.parseOperand(ub) || parser.parseKeyword("step") ||
      parser.parseOperand(step) || parser.parseKeyword("mask") ||
      parser.parseOperand(mask_in))
    return failure();

  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  regionArgs.push_back(inductionVariable);
  regionArgs.push_back(mask_out);

  if (parser.parseColon())
    return failure();

  // Parse in types
  if (parser.parseLParen() || parser.parseType(bound_type) ||
      parser.parseKeyword("xx") || parser.parseType(in_mask_type) ||
      parser.parseRParen())
    return failure();

  // Parse types
  if (parser.parseArrow())
    return failure();

  // Parse in types
  if (parser.parseLParen() || parser.parseType(ind_type) ||
      parser.parseKeyword("xx") || parser.parseType(out_mask_type) ||
      parser.parseRParen())
    return failure();

  bool hasIterArgs = succeeded(parser.parseOptionalKeyword("init_args"));
  if (hasIterArgs) {
    assert(false);
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return failure();
  }

  // if (regionArgs.size() != result.types.size() + 1)
  //   return parser.emitError(
  //       parser.getNameLoc(),
  //       "mismatch in number of loop-carried values and defined values");

  // Resolve input operands.
  regionArgs.front().type = indvar_type;
  regionArgs.back().type = in_mask_type;
  if (parser.resolveOperand(lb, bound_type, result.operands) ||
      parser.resolveOperand(ub, bound_type, result.operands) ||
      parser.resolveOperand(step, bound_type, result.operands) ||
      parser.resolveOperand(mask_in, in_mask_type, result.operands))
    return failure();

  if (hasIterArgs) {
    assert(false);
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

  SlcVecForOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}