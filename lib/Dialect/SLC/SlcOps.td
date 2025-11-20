#ifndef LIB_DIALECT_SLC_SLCOPS_TD
#define LIB_DIALECT_SLC_SLCOPS_TD

include "mlir/IR/OpBase.td"

include "SlcDialect.td"
include "SlcTypes.td"
include "SlcAttr.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class SlcOp<string mnemonic, list<Trait> traits = []> : Op<Slc_Dialect, mnemonic, traits>;

def SlcConfigureOp : SlcOp<"configure", []> {
    let summary = "A SLC configuration operation.";
    let description = [{
        Example:

        ```mlir
        slc.configure
        ```
    }];

    let assemblyFormat = "attr-dict";
}

def SlcForOp : SlcOp<"for", 
                        [
                            AutomaticAllocationScope,
                            DeclareOpInterfaceMethods<LoopLikeOpInterface,
                                ["getInitsMutable", "getLoopResults", "getRegionIterArgs",
                                    "getSingleInductionVar", "getSingleLowerBound", "getSingleStep",
                                    "getSingleUpperBound", "getYieldedValuesMutable",
                                    "replaceWithAdditionalYields"]>,
                            AllTypesMatch<["lowerBound", "upperBound", "step"]>,
                            SingleBlockImplicitTerminator<"slc::SlcYieldOp">,
                        ]> {
    let summary = "A SLC for loop.";
    let description = [{
        Example:

        ```mlir
        slc.for %ivar = %lb to %ub step %step : !slc.stream<index> { ... }
        ```
    }];

    let arguments = (ins Arg<AnyTypeOf<[Index,SlcStreamOf<[Index]>]>>:$lowerBound,
                       Arg<AnyTypeOf<[Index,SlcStreamOf<[Index]>]>>:$upperBound,
                       Arg<AnyTypeOf<[Index,SlcStreamOf<[Index]>]>>:$step,
                       Variadic<Index>:$initArgs);
    
    let results = (outs Variadic<Index>:$results);
    let regions = (region SizedRegion<1>:$region);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$lowerBound, "Value":$upperBound, "Value":$step,
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

        /// Return the `index`-th region iteration argument.
        BlockArgument getRegionIterArg(unsigned index) {
            assert(index < getNumRegionIterArgs() &&
                "expected an index less than the number of region iter args");
            return getBody()->getArguments().drop_front(getNumInductionVars())[index];
        }

        void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
        void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
        void setStep(Value step) { getOperation()->setOperand(2, step); }

        /// Number of induction variables, always 1 for scf::ForOp.
        unsigned getNumInductionVars() { return 1; }
        /// Number of region arguments for loop-carried values
        unsigned getNumRegionIterArgs() {
            return getBody()->getNumArguments() - getNumInductionVars();
        }
        /// Number of operands controlling the loop: lb, ub, step
        unsigned getNumControlOperands() { return 3; }
    }];
}

def SlcMemStreamOp : SlcOp<"mem_stream",
                    [
                        TypesMatchWith<"result and 'memref' element types match", "memref", "result", "SlcStreamType::get(::llvm::cast<MemRefType>($_self).getElementType())">,
                    ]> {
    let summary = "SLC mem stream";
    let description = [{
        Example:

        ```mlir
        %ptb = slc.mem_stream %arg0[%ite] : memref<?xindex> to !slc.stream<index>
        ```
    }];
    
    let arguments = (ins Arg<MemRefOf<[AnyType]>, "the reference to load from", [MemRead]>:$memref,
                        Variadic<AnyTypeOf<[Index,SlcStreamOf<[Index]>]>>:$indices,
                        DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal);
    let results = (outs SlcStream<AnyType>:$result);

    let assemblyFormat = "$memref `[` $indices `]` attr-dict `:` type($memref) `[` type($indices) `]` `into` type($result) ";
}

def SlcAluStreamOp : SlcOp<"alu_stream",
                        [
                            //AllTypesMatch<["lhs", "rhs", "result"]>
                        ]
                        > {
    let summary = "SLC ALU stream";
    let description = [{
        Example:

        ```mlir
        %ite = slc.alu_stream<add> %lhs, %rhs : !slc.stream<index>
        ```
    }];

    let arguments = (ins Arg<AnyTypeOf<[Index,SlcStreamOf<[Index]>]>>:$lhs,
                            Arg<AnyTypeOf<[Index,SlcStreamOf<[Index]>]>>:$rhs, 
                            OpTypeAttr:$aluOp);
    let results = (outs SlcStream<Index>:$result);

    let assemblyFormat = "`<` $aluOp `>` $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SlcFromStreamOp : SlcOp<"from_stream",
                    [
                        TypesMatchWith<"value and 'stream' element types match", "stream", "value", "::llvm::cast<SlcStreamType>($_self).getElementType()">,
                    ]
                    > {
    let summary = "Reads the head of a stream";
    let description = [{
        Example:

        ```mlir
        %ptb_val = slc.from_stream %ptb : index
        ```
    }];

    let arguments = (ins SlcStream<AnyType>:$stream);
    let results = (outs AnyType:$value);

    let assemblyFormat = "$stream attr-dict `:` type($stream) `to` type($value)";
}

def SlcToStreamOp : SlcOp<"to_stream",
                    [
                        TypesMatchWith<"value and 'stream' element types match", "value", "stream", "SlcStreamType::get($_self)">,
                    ]
                    > {
    let summary = "Write the head of a stream";
    let description = [{
        Example:

        ```mlir
        %ptb_val = slc.to_stream %ptb : !slc.stream<index>
        ```
    }];

    let arguments = (ins AnyType:$value);
    let results = (outs SlcStream<AnyType>:$stream);

    let assemblyFormat = "$value attr-dict `:` type($value) `to` type($stream)";
}

def SlcAccelerateForOp : SlcOp<"accelerate_for", []> {
    let summary = "A SLC for acceleration operation.";
    let description = [{
        Example:

        ```mlir
        slc.accelerate_for
        ```
    }];

    let assemblyFormat = "attr-dict";
}

def SlcCallbackOp : SlcOp<"callback", [
                                        AffineScope, 
                                        AutomaticAllocationScope,
                                        SingleBlockImplicitTerminator<"slc::SlcYieldOp">
                                      ]> {
    let summary = "A SLC callback containing operations to trigger upon traversal events.";
    let description = [{
        Example:

        ```mlir
        slc.callback { ... }
        ```
    }];

    let results = (outs Variadic<Index>:$results);
    let regions = (region SizedRegion<1>:$region);

    let skipDefaultBuilders = 1;
    let builders = [OpBuilder<(ins "TypeRange":$resultTypes)>];

    let assemblyFormat = "$region `:` type($results) attr-dict";
}

def SlcYieldOp : SlcOp<"yield", [Pure, ReturnLike, Terminator,
        ParentOneOf<["SlcForOp", "SlcCallbackOp"]>]> {
    let summary = "SLC loop yield and termination operation";
    let description = [{"slc.yield" terminates a regions.}];

    let arguments = (ins Variadic<Index>:$results);
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

    let assemblyFormat = [{ attr-dict ($results^ `:` type($results))? }];
}

#endif /* LIB_DIALECT_SLC_SLCOPS_TD */
 