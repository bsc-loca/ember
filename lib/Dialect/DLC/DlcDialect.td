#ifndef LIB_DIALECT_DLC_DLCDIALECT_TD
#define LIB_DIALECT_DLC_DLCDIALECT_TD

include "mlir/IR/DialectBase.td"

def Dlc_Dialect : Dialect {
    let name = "dlc";
    let summary = "A dialect for DLC Extended";
    let description = [{}];

    let cppNamespace = "::mlir::dlc";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;

  let dependentDialects = [
    "mlir::slc::SlcDialect",
    // "mlir::dlc::DlcDialect",
    // "mlir::sparse_tensor::SparseTensorDialect",
  ];
}

#endif /* LIB_DIALECT_DLC_DLCDIALECT_TD */
