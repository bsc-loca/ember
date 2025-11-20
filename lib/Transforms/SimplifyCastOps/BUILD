load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = ["//visibility:public"],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=SimplifyCastOps",
            ],
            "Passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "Passes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "SimplifyCastOps",
    srcs = ["SimplifyCastOps.cpp"],
    hdrs = [
        "Passes.h",
    ],
    deps = [
        "//lib/Transforms/SimplifyCastOps:pass_inc_gen",
        "//lib/Dialect/SLC:Slc",
        "//lib/Dialect/SLCVEC:SlcVec",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:SparseTensorTransforms",
        "@llvm-project//mlir:SparseTensorUtils",
        # "@llvm-project//mlir:AffineDialect",
        # "@llvm-project//mlir:ArithDialect",
        # "@llvm-project//mlir:FuncDialect",
        # "@llvm-project//mlir:FuncTransforms",
        # "@llvm-project//mlir:IR",
        # "@llvm-project//mlir:IndexDialect",
        # "@llvm-project//mlir:MathDialect",
        # "@llvm-project//mlir:MemRefDialect",
        # "@llvm-project//mlir:Support",
        # "@llvm-project//mlir:SparseTensorDialect",
        # "@llvm-project//mlir:TensorDialect",
        # "@llvm-project//mlir:LinalgDialect",
        # "@llvm-project//mlir:TranslateLib",
    ],
)

cc_library(
    name = "Passes",
    hdrs = ["Passes.h"],
    deps = [
        # ":SamPasses",
        ":pass_inc_gen",
    ],
)
