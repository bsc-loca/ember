#ifndef LIB_DIALECT_DLC_DLCATTR_TD
#define LIB_DIALECT_DLC_DLCATTR_TD

include "DlcDialect.td"
include "DlcEnums.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/TensorEncoding.td"

class Dlc_Attr<string name, list<Trait> traits = []> : AttrDef<Dlc_Dialect, name, traits>;


def OpTypeAttr
    : EnumAttr<Dlc_Dialect, OpTypeEnum,
               "OpType"> {
   let mnemonic = "op_type";
   let assemblyFormat = "qualified($value)";
}


#endif /* LIB_DIALECT_DLC_DLCATTR_TD */
