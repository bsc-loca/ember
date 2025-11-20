#ifndef LIB_DIALECT_DLC_DLCTYPES_TD
#define LIB_DIALECT_DLC_DLCTYPES_TD

include "DlcDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//class Dlc_Type<string name, string typeMnemonic> : TypeDef<Dlc_Dialect, name> {
//    let mnemonic = typeMnemonic;
//}

// Base class for DLC dialect types.
class Dlc_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Dlc_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let typeName = "dlc." # typeMnemonic;
}

def DlcTuType : Dlc_Type<"DlcTuType", "tu"> {
    let summary = "DLC Traversal Unit";
    let description = [{
        
        Syntax: TODO

        examples: TODO
    }];

    let assemblyFormat = "";
    let skipDefaultBuilders=0;
}

#endif /* LIB_DIALECT_DLC_DLCTYPES_TD */
