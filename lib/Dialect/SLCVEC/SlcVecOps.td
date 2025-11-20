#ifndef LIB_DIALECT_SLCVEC_SLCVECOPS_TD
#define LIB_DIALECT_SLCVEC_SLCVECOPS_TD

include "mlir/IR/OpBase.td"

include "SlcVecDialect.td"
include "SlcVecTypes.td"
include "SlcVecAttr.td"
include "SlcTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class SlcVecOp<string mnemonic, list<Trait> traits = []> : Op<SlcVec_Dialect, mnemonic, traits>;


def SlcVectorizeForOp : SlcVecOp<"vectorize_for", []> {
    let summary = "A SLC for vectorization operation.";
    let description = [{
        Example:

        ```mlir
        slc.vectorize_for
        ```
    }];

    let assemblyFormat = "attr-dict";
}

class SlcVectorOf<list<Type> type> :
    FixedVectorOfLengthAndType<[2, 3, 4, 8, 16], type>;

def SlcVecForOp : SlcVecOp<"for", 
                        [
                            AutomaticAllocationScope,
                            DeclareOpInterfaceMethods<LoopLikeOpInterface,
                                ["getInitsMutable", "getLoopResults", "getRegionIterArgs",
                                    "getSingleInductionVar", "getSingleLowerBound", "getSingleStep",
                                    "getSingleUpperBound", "getYieldedValuesMutable",
                                    "replaceWithAdditionalYields"]>,
                            AllTypesMatch<["lowerBound", "upperBound", "step"]>,
                            SingleBlockImplicitTerminator<"slcvec::SlcVecYieldOp">,
                        ]> {
    let summary = "A SLC for loop.";
    let description = [{
        Example:

        ```mlir
        slc.for %ivar = %lb to %ub step %step : !slc.stream<index> { ... }
        ```
    }];

    let arguments = (ins Arg<AnyTypeOf<[Index,SlcStreamOf<[AnyTypeOf<[Index,SlcVectorOf<[Index]>]>]>]>>:$lowerBound,
                       Arg<AnyTypeOf<[Index,SlcStreamOf<[AnyTypeOf<[Index,SlcVectorOf<[Index]>]>]>]>>:$upperBound,
                       Arg<AnyTypeOf<[Index,SlcStreamOf<[AnyTypeOf<[Index,SlcVectorOf<[Index]>]>]>]>>:$step,
                       Arg<AnyTypeOf<[SlcVectorOf<[I1]>,SlcStreamOf<[SlcVectorOf<[I1]>]>]>>:$inMask,
                       Arg<APIntAttr>:$vectorLength,
                       Arg<LoopConfigAttr>:$loopConfig,
                       Variadic<Index>:$initArgs);
    
    let results = (outs Variadic<Index>:$results);
    let regions = (region SizedRegion<1>:$region);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$lowerBound, "Value":$upperBound, "Value":$step, "Value":$inMask,
        CArg<"long", "1">:$vectorLength,
        CArg<"LoopConfig", "LoopConfig::none">:$loopConfig,
        CArg<"ValueRange", "std::nullopt">:$iterArgs,
        CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
            "nullptr">)>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasRegionVerifier = 1;

    let extraClassDeclaration = [{
        using BodyBuilderFn =
            function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

        Value getInductionVar() { return getBody()->getArgument(0); }
        Value getOutMask() { return getBody()->getArgument(1); }

        /// Return the `index`-th region iteration argument.
        BlockArgument getRegionIterArg(unsigned index) {
            assert(index < getNumRegionIterArgs() &&
                "expected an index less than the number of region iter args");
            return getBody()->getArguments().drop_front(getNumInductionVars())[index];
        }

        void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
        void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
        void setStep(Value step) { getOperation()->setOperand(2, step); }

        /// Number of induction variables (indVar+mask)
        unsigned getNumInductionVars() { return 2; }
        /// Number of region arguments for loop-carried values
        unsigned getNumRegionIterArgs() {
            return getBody()->getNumArguments() - getNumInductionVars();
        }
        /// Number of operands controlling the loop: lb, ub, step
        unsigned getNumControlOperands() { return 3; }

        static constexpr llvm::StringRef getStaticVectorLengthAttrName() { return "vectorLength"; }
        static constexpr llvm::StringRef getStaticLoopConfigAttrName() { return "loopConfig"; }
        //long getVectorLengthValue() { getAttr(getVectorLengthAttrName()); }
    }];
}


def SlcVecYieldOp : SlcVecOp<"yield", [Pure, ReturnLike, Terminator,
        ParentOneOf<["SlcVecForOp"]>]> {
    let summary = "SLC vec loop yield and termination operation";
    let description = [{"slcvec.yield" terminates a regions.}];

    let arguments = (ins Variadic<SlcStreamOf<[SlcVectorOf<[Index]>]>>:$results);
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

    let assemblyFormat = [{ attr-dict ($results^ `:` type($results))? }];
}

def SlcVecMemStreamOp : SlcVecOp<"mem_stream",
                    [
                        //AllElementTypesMatch<["memref", "result"]>,
                    ]> {
    let summary = "SLC mem stream";
    let description = [{
        Example:

        ```mlir
        %ptb = slcvec.mem_stream %arg0[%ite] : memref<?xindex> to !slc.stream<index>
        ```
    }];
    
    let arguments = (ins Arg<MemRefOf<[AnyType]>, "the reference to load from", [MemRead]>:$memref,
                        Variadic<SlcStreamOf<[AnyTypeOf<[Index,SlcVectorOf<[Index]>]>]>>:$indices,
                        DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal);
    let results = (outs SlcStreamOf<[SlcVectorOf<[AnyType]>]>:$result);

    let assemblyFormat = "$memref `[` $indices `]` attr-dict `:` type($memref) `[` type($indices) `]` `into` type($result) ";
}

/*
def SlcVecAluStreamOp : SlcVecOp<"vec_alu_stream",
                        [
                            AllTypesMatch<["lhs", "rhs", "result"]>
                        ]
                        > {
    let summary = "SLC ALU stream";
    let description = [{
        Example:

        ```mlir
        %ite = slc.alu_stream<add> %lhs, %rhs : !slc.stream<index>
        ```
    }];

    let arguments = (ins Arg<SlcStreamOf<[AnyTypeOf<[Index,SlcVectorOf<[Index]>]>]>>:$lhs,
                            Arg<SlcStreamOf<[AnyTypeOf<[Index,SlcVectorOf<[Index]>]>]>>:$rhs, 
                            OpTypeAttr:$aluOp);
    let results = (outs SlcStreamOf<[SlcVectorOf<[Index]>]>:$result);

    let assemblyFormat = "`<` $aluOp `>` $lhs `,` $rhs attr-dict `:` type($result)";
}
*/
def SlcBroadcastStreamOp : SlcVecOp<"broadcast",
                    [
                        //TypesMatchWith<"stream has same element type of vecStream", "vecStream", "stream", "SlcStreamType::get(::llvm::cast<SlcVecStreamType>($_self).getElementType())">,
                    ]
                    > {
    let summary = "Broadcast a stream to a vector stream";
    let description = [{
        Example:

        ```mlir
        %ptb_val = slc.broadcast %ptb : !slc.vec_stream<8xindex>
        ```
    }];

    let arguments = (ins SlcStreamOf<[AnyType]>:$stream);
    let results = (outs SlcStreamOf<[SlcVectorOf<[AnyType]>]>:$vecStream);

    let assemblyFormat = "$stream attr-dict `:` type($stream) `to` type($vecStream)";
}

def SlcTmpVecStreamCastOp : SlcVecOp<"tmp_vec_stream_cast",
                    [
                        //TypesMatchWith<"in and out element types match", "in", "out", "mlir::slc::SlcStreamType::get(cast<SlcVecStreamType>($_self).getElementType())">,
                    ]
                    > {
    let summary = "Temporary vector stream to scalar stream cast";
    let description = [{
        Example:

        ```mlir
        %ptb_val = slc.tmp_vec_stream_cast %ptb : index
        ```
    }];

    let arguments = (ins SlcStreamOf<[SlcVectorOf<[AnyType]>]>:$in);
    let results = (outs SlcStreamOf<[AnyType]>:$out);

    let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";
}

def SlcTmpVecCastOp : SlcVecOp<"tmp_vec_cast",
                    [
                        TypesMatchWith<"in and out element types match", "in", "out", "cast<VectorType>($_self).getElementType()">,
                    ]
                    > {
    let summary = "Temporary vector to scalar cast op";
    let description = [{
        Example:

        ```mlir
        %ptb_val = slc.tmp_vec_cast %ptb : index
        ```
    }];

    let arguments = (ins SlcVectorOf<[AnyType]>:$in);
    let results = (outs AnyType:$out);

    let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";
}


def SlcVecBufStreamOp : SlcVecOp<"buf_stream", []> {
    let summary = "Allocates a SLC buffer";
    let description = [{
        Example:

        ```mlir
        %buff = slc.buf_stream : !slc.stream<memref<?xf32>>
        ```
    }];

    let builders = [
        OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType), [{
            odsState.addTypes(mlir::slc::SlcStreamType::get(MemRefType::get(shape, elementType)));
        }]>
    ];

    let results = (outs SlcStream<MemRefOf<[AnyType]>>:$buffer);

    let assemblyFormat = "attr-dict `:` type($buffer)";
}

def SlcVecToBufferOp : SlcVecOp<"to_buffer",
                    [
                        //TypesMatchWith<"buffer and stream element types match", "buffer", "stream", "SlcStreamType::get(::llvm::cast<MemRefType>(::llvm::cast<SlcStreamType>($_self).getElementType()).getElementType())">,
                    ]
                    > {
    let summary = "Buffer stream elements.";
    let description = [{
        Example:

        ```mlir
        slc.to_bufffer %val, %buff[%idx1] : memref<?xf32>
        ```
    }];

    let arguments = (ins SlcStream<AnyType>:$stream,
                        Arg<SlcStream<MemRefOf<[AnyType]>>, "the buffer to store to",
                            [MemWrite]>:$buffer,
                        Variadic<SlcStream<AnyTypeOf<[Index, SlcVectorOf<[Index]>]>>>:$indices);

    let assemblyFormat = [{
        $stream `,` $buffer `[` $indices `]` attr-dict `:` type($stream) `->` type($buffer) `[` type($indices) `]`
    }];
}

def SlcVecFromBufferOp : SlcVecOp<"from_buffer",
     [
     ]> {
  let summary = "load-from-SLC-buffer operation";
  let description = [{
    The `load` op reads an element from a buffer specified by an index list. The
    output of load is a new value with the same type as the elements of the
    memref. The arity of indices is the rank of the memref (i.e., if the memref
    loaded from is of rank 3, then 3 indices are required for the load following
    the memref identifier).
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       Variadic<Index>:$indices,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal);
  let results = (outs AnyType:$result);

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getMemRef().getType());
    }
  }];

  //let hasFolder = 1;
  //let hasVerifier = 1;

  let assemblyFormat = "$memref `[` $indices `]` attr-dict `:` type($memref) `[` type($indices) `]` `->` type($result)";
}

def SlcVecExtractOp : SlcVecOp<"extract",
                    [
                        //TypesMatchWith<"buffer and stream element types match", "buffer", "stream", "SlcStreamType::get(::llvm::cast<MemRefType>(::llvm::cast<SlcStreamType>($_self).getElementType()).getElementType())">,
                    ]
                    > {
    let summary = "Extract a stream from a vector stream.";
    let description = [{
        Example:

        ```mlir
        slcvec.extract %vstr[0] : !slc.stream<vector<8xindex>> -> !slc.stream<index>
        ```
    }];

    let arguments = (ins SlcStreamOf<[SlcVectorOf<[AnyType]>]>:$vectorStream,
                        Arg<APIntAttr>:$index);
    let results = (outs SlcStreamOf<[AnyType]>:$result);

    let assemblyFormat = [{
       $vectorStream `[` $index `]` attr-dict `:` type($vectorStream) `->` type($result)
    }];
}

def SlcVecIndVarOp : SlcVecOp<"vec_indvar",
                    [
                        //TypesMatchWith<"buffer and stream element types match", "buffer", "stream", "SlcStreamType::get(::llvm::cast<MemRefType>(::llvm::cast<SlcStreamType>($_self).getElementType()).getElementType())">,
                    ]
                    > {
    let summary = "Generates a vector induction variable";
    let description = [{
        Example:

        ```mlir
        slcvec.vec_indvar %indVar, %lb, %ub, %step : vector<8xindex>
        ```
    }];

    let arguments = (ins Index:$indVar,
                        Index:$lb,
                        Index:$ub,
                        Index:$step);
    let results = (outs SlcVectorOf<[Index]>:$vecIndVar);

    let assemblyFormat = [{
       $indVar $lb $ub $step attr-dict `:` type($vecIndVar)
    }];
}

def SlcVecMaskOp : SlcVecOp<"mask",
                    [
                        //TypesMatchWith<"buffer and stream element types match", "buffer", "stream", "SlcStreamType::get(::llvm::cast<MemRefType>(::llvm::cast<SlcStreamType>($_self).getElementType()).getElementType())">,
                    ]
                    > {
    let summary = "Generates a mask";
    let description = [{
        Example:

        ```mlir
        slcvec.mask %indVar, %lb, %ub, %step : vector<8xindex>
        ```
    }];

    let arguments = (ins Index:$indVar,
                        Index:$lb,
                        Index:$ub,
                        Index:$step);
    let results = (outs SlcVectorOf<[I1]>:$mask);

    let assemblyFormat = [{
       $indVar $lb $ub $step attr-dict `:` type($mask)
    }];
}

#endif /* LIB_DIALECT_SLCVEC_SLCVECOPS_TD */
 