#ifndef LIB_DIALECT_SLCVEC_SLCVECDIALECT_TD
#define LIB_DIALECT_SLCVEC_SLCVECDIALECT_TD

include "mlir/IR/DialectBase.td"

def SlcVec_Dialect : Dialect {
    let name = "slcvec";
    let summary = "A dialect for SLCVEC Extended";
    let description = [{}];

    let cppNamespace = "::mlir::slcvec";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;

  let dependentDialects = [
    "mlir::slc::SlcDialect",
    "mlir::vector::VectorDialect",
    // "mlir::sparse_tensor::SparseTensorDialect",
  ];
}

#endif /* LIB_DIALECT_SLCVEC_SLCVECDIALECT_TD */
