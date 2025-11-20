#ifndef LIB_DIALECT_SLCVEC_SLCVECENUM_TD
#define LIB_DIALECT_SLCVEC_SLCVECENUM_TD

include "mlir/IR/EnumAttr.td"
include "SlcVecDialect.td"

def LoopConfigEnum : I32EnumAttr<"LoopConfig", "stuff", [
    I32EnumAttrCase<"none",  0>,
    I32EnumAttrCase<"scl",   1>,
    I32EnumAttrCase<"bcast", 2>,
    I32EnumAttrCase<"vec",   3>,
]>{
    let genSpecializedAttr = 0;
    let cppNamespace = SlcVec_Dialect.cppNamespace;
}


#endif /* LIB_DIALECT_SLCVEC_SLCVECENUM_TD */
