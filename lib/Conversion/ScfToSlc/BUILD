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
                "-name=ScfToSlc",
            ],
            "Passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "ScfToSlc.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
        # "@llvm-project//mlir:SparseTensorDialect",
    ],
)

cc_library(
    name = "ScfToSlc",
    # srcs = ["ScfToSlc.cpp"],
    srcs = ["ScfToSlc.cpp"],
    # hdrs = ["//include/Conversion/ScfToSlc/ScfToSlc.h"],
    hdrs = [
        # "ScfToSlc.h",
        "ScfToSlc.h",
        "Passes.h",
    ],
    deps = [
        "//lib/Dialect/SLC:Slc",
        "//lib/Conversion/ScfToSlc:pass_inc_gen",
        # "//lib/Target/Common:common_emitter",
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
        ":ScfToSlc",
        ":pass_inc_gen",
    ],
)
