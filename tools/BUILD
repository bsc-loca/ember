package(
    default_visibility = ["//visibility:public"],
)

cc_binary(
    name = "ember-opt",
    srcs = ["ember-opt.cpp"],
    includes = [],
    deps = [
        "//lib/Dialect/SLC:Slc",
        "//lib/Dialect/SLCVEC:SlcVec",
        "//lib/Dialect/DLC:Dlc",
        "//lib/Conversion/ScfToSlc",
        "//lib/Conversion/SlcToDlc",
        "//lib/Conversion/SlcVectorizer",
        "//lib/Transforms/CallbackVectorizer",
        "//lib/Transforms/SimplifyMemOps",
        "//lib/Transforms/SimplifyCastOps",
        "//lib/Transforms/ChooseDecoupling:ChooseDecoupling",
        "//lib/Transforms/MoveToValOps:MoveToValOps",
        "//lib/Transforms/ReplaceToValOps:ReplaceToValOps",
        "//lib/Transforms/BufferCompoundTypes:BufferCompoundTypes",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:SparseTensorTransforms",
        "@llvm-project//mlir:SparseTensorUtils",
    ],
)