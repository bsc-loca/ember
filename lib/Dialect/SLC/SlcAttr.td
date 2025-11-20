#ifndef LIB_DIALECT_SLC_SLCATTR_TD
#define LIB_DIALECT_SLC_SLCATTR_TD

include "SlcDialect.td"
include "SlcEnums.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/TensorEncoding.td"



class Slc_Attr<string name, list<Trait> traits = []> : AttrDef<Slc_Dialect, name, traits>;


def OpTypeAttr
    : EnumAttr<Slc_Dialect, OpTypeEnum,
               "OpType"> {
   let mnemonic = "op_type";
   let assemblyFormat = "qualified($value)";
}


#endif /* LIB_DIALECT_SLC_SLCATTR_TD */
