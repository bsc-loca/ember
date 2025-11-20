#ifndef LIB_DIALECT_SLC_SLCDIALECT_TD
#define LIB_DIALECT_SLC_SLCDIALECT_TD

include "mlir/IR/DialectBase.td"

def Slc_Dialect : Dialect {
    let name = "slc";
    let summary = "A dialect for SLC Extended";
    let description = [{}];

    let cppNamespace = "::mlir::slc";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;

  let dependentDialects = [
    // "mlir::slc::SlcDialect",
    // "mlir::sparse_tensor::SparseTensorDialect",
  ];
}

#endif /* LIB_DIALECT_SLC_SLCDIALECT_TD */
