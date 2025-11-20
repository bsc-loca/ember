#ifndef LIB_DIALECT_SLC_SLCENUM_TD
#define LIB_DIALECT_SLC_SLCENUM_TD

include "mlir/IR/EnumAttr.td"
include "SlcDialect.td"

def OpTypeEnum : I32EnumAttr<"OpType", "stuff", [
    I32EnumAttrCase<"add", 0>,
    I32EnumAttrCase<"sub", 1>,
    I32EnumAttrCase<"mul", 2>,
    I32EnumAttrCase<"div", 3>,
]>{
    let genSpecializedAttr = 0;
    let cppNamespace = Slc_Dialect.cppNamespace;
}


#endif /* LIB_DIALECT_SLC_SLCENUM_TD */
