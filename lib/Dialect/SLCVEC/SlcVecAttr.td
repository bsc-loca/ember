#ifndef LIB_DIALECT_SLCVEC_SLCVECATTR_TD
#define LIB_DIALECT_SLCVEC_SLCVECATTR_TD

include "SlcVecDialect.td"
include "SlcVecEnums.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/TensorEncoding.td"

class SlcVec_Attr<string name, list<Trait> traits = []> : AttrDef<SlcVec_Dialect, name, traits>;


def LoopConfigAttr
    : EnumAttr<SlcVec_Dialect, LoopConfigEnum,
               "LoopConfig"> {
   let mnemonic = "loop_config";
   let assemblyFormat = "qualified($value)";
}


#endif /* LIB_DIALECT_SLCVEC_SLCVECATTR_TD */
