#ifndef LIB_DIALECT_DLC_DLCOPS_TD
#define LIB_DIALECT_DLC_DLCOPS_TD

include "mlir/IR/OpBase.td"

include "DlcDialect.td"
include "DlcTypes.td"
include "DlcAttr.td"
include "SlcTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class DlcOp<string mnemonic, list<Trait> traits = []> : Op<Dlc_Dialect, mnemonic, traits>;

def DlcConfigureOp : DlcOp<"configure", []> {
    let summary = "A DLC configuration operation.";
    let description = [{
        Example:

        ```mlir
        dlc.configure
        ```
    }];

    let assemblyFormat = "attr-dict";
}


def DlcTuOp : DlcOp<"new_tu",
                    [
                        AllTypesMatch<["lowerBound", "upperBound", "step"]>,
                    ]> {
    let summary = "Declare a new DLC Traversal Unit";
    let description = [{
        Example:

        ```mlir
        %ptb = dlc.new_tu(%beg to %end step %step) : tu
        ```
    }];
    
    let arguments = (ins AnyType:$lowerBound,
                        AnyType:$upperBound,
                        AnyType:$step);
    let results = (outs DlcTuType:$result);

    let assemblyFormat = "$lowerBound `to` $upperBound `step` $step attr-dict `:` type($step) `into` type($result) ";
}


def DlcGetIVOp : DlcOp<"get_ind_var", []> {
    let summary = "Get induction variable from traversal unit";
    let description = [{
        Example:

        ```mlir
        %ptb = dlc.get_ind_var(%tu) : !dlc.stream<index>
        ```
    }];
    
    let arguments = (ins DlcTuType:$tu);
    let results = (outs SlcStreamOf<[Index]>:$result);

    let assemblyFormat = "$tu attr-dict `:` type($result) ";
}

def DlcRegisterCallbackOp : DlcOp<"register_callback", []> {
    let summary = "Register callback to push control token on TU event";
    let description = [{
        Example:

        ```mlir
        dlc.register_callback(%tu, "end", 3)
        ```
    }];
    
    let arguments = (ins DlcTuType:$tu, OpTypeAttr:$event, I64Attr:$id);
    //let results = (outs SlcStreamOf<[Index]>:$result);

    let assemblyFormat = "$tu `,` $event attr-dict `:` type($tu)";
}

def DlcRegisterOperandOp : DlcOp<"register_operand", []> {
    let summary = "Register operand for callback on TU event";
    let description = [{
        Example:

        ```mlir
        dlc.register_operand(%tu, "end", %operand)
        ```
    }];
    
    let arguments = (ins DlcTuType:$tu, OpTypeAttr:$event, SlcStreamOf<[AnyType]>:$value);
    //let results = (outs SlcStreamOf<[Index]>:$result);

    let assemblyFormat = "$tu `,` $event `,` $value attr-dict `:` type($tu) `,` type($value)";
}


def DlcPopOperandOp : DlcOp<"pop_operand", []> {
    let summary = "Pop operand from queue";
    let description = [{
        Example:

        ```mlir
        dlc.pop_operand()
        ```
    }];
    
    let arguments = (ins );
    let results = (outs AnyType:$result);

    let assemblyFormat = "attr-dict `:` type($result)";
}

def DlcConfigOp : DlcOp<"configure_access_engine", [
                                        AffineScope, 
                                        AutomaticAllocationScope,
                                        SingleBlockImplicitTerminator<"dlc::DlcYieldOp">
                                      ]> {
    let summary = "A DLC region storing the configuration commands for the access unit.";
    let description = [{
        Example:

        ```mlir
        dlc.configure_access_engine { ... }
        ```
    }];

    let results = (outs Variadic<Index>:$results);
    let regions = (region SizedRegion<1>:$region);

    let skipDefaultBuilders = 1;
    let builders = [OpBuilder<(ins "TypeRange":$resultTypes)>];

    let assemblyFormat = "$region `:` type($results) attr-dict";
}

def DlcYieldOp : DlcOp<"yield", [Pure, ReturnLike, Terminator,
        ParentOneOf<["DlcConfigOp", "DlcComputeLoopOp"]>]> {
    let summary = "DLC loop yield and termination operation";
    let description = [{"dlc.yield" terminates a regions.}];

    let arguments = (ins Variadic<Index>:$results);
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

    let assemblyFormat = [{ attr-dict ($results^ `:` type($results))? }];
}

def DlcComputeLoopOp : DlcOp<"compute_loop", [//RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"dlc::DlcYieldOp">,
    /*DeclareOpInterfaceMethods<RegionBranchOpInterface,
                              ["getRegionInvocationBounds",
                               "getEntrySuccessorRegions"]>*/]> {
  let summary = "DLC compute loop";
  let description = [{
    TODO
    ```
  }];

  let arguments = (ins DenseI64ArrayAttr:$cases);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$caseRegions);

  let assemblyFormat = [{
    attr-dict (`->` type($results)^)?
    custom<SwitchCases>($cases, $caseRegions)
  }];

  let extraClassDeclaration = [{
    /// Get the number of cases.
    unsigned getNumCases();

    /// Get the default region body.
    /// Block &getDefaultBlock();

    /// Get the body of a case region.
    Block &getCaseBlock(unsigned idx);
  }];

  let hasCanonicalizer = 0;
  let hasVerifier = 0;
}


def DlcGetVarOp : DlcOp<"get_var", []> {
    let summary = "Get induction variable by ID";
    let description = [{
        Example:

        ```mlir
        TODO
        ```
    }];
    
    let arguments = (ins );
    let results = (outs Index:$result);

    let assemblyFormat = "attr-dict `:` type($result)";
}


def DlcIncrementVarOp : DlcOp<"increment_var", []> {
    let summary = "Increment induction variable by ID";
    let description = [{
        Example:

        ```mlir
        TODO
        ```
    }];
    
    let arguments = (ins );
    let results = (outs );

    let assemblyFormat = "attr-dict";
}

#endif /* LIB_DIALECT_DLC_DLCOPS_TD */
 