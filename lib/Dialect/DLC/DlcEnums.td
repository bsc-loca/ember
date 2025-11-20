#ifndef LIB_DIALECT_DLC_DLCENUM_TD
#define LIB_DIALECT_DLC_DLCENUM_TD

include "mlir/IR/EnumAttr.td"
include "DlcDialect.td"

def OpTypeEnum : I32EnumAttr<"OpType", "stuff", [
    I32EnumAttrCase<"beg", 0>,
    I32EnumAttrCase<"ite", 1>,
    I32EnumAttrCase<"end", 2>,
]>{
    let genSpecializedAttr = 0;
    let cppNamespace = Dlc_Dialect.cppNamespace;
}

#endif /* LIB_DIALECT_DLC_DLCENUM_TD */
