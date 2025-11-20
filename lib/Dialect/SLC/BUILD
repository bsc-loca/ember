load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "td_files",
    srcs = [
        "SlcDialect.td",
        "SlcAttr.td",
        "SlcTypes.td",
        "SlcEnums.td",
    ],
    includes = ["@slc//include", "//lib/Dialect/SLC",],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:DestinationStyleOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:TilingInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "SlcDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "SlcDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SlcDialect.td",
    deps = [
        ":td_files",
        # "@llvm-project//mlir:SparseTensorDialect",
        # "@llvm-project//mlir:SparseTensorEnums",
        # "@llvm-project//mlir:SparseTensorPipelines",
        # "@llvm-project//mlir:SparseTensorTransforms",
        # "@llvm-project//mlir:SparseTensorTypesIncGen",
        # "@llvm-project//mlir:SparseTensorUtils",
    ],
)

gentbl_cc_library(
    name = "types_inc_gen",
    tbl_outs = [
        (
            ["-gen-typedef-decls"],
            "SlcTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "SlcTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SlcTypes.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "attrs_inc_gen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "SlcAttrDefs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "SlcAttrDefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SlcAttr.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "SlcOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "SlcOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SlcOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
    ],
)

gentbl_cc_library(
    name = "enums_inc_gen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "SlcEnumDefs.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "SlcEnumDefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SlcEnums.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

cc_library(
    name = "Slc",
    srcs = [
        "SlcDialect.cpp",
        "SlcOps.cpp",
        "SlcTypes.cpp"
        ],
    hdrs = [
        "SlcDialect.h",
        "SlcOps.h",
        "SlcTypes.h",
    ],
    deps = [
        ":attrs_inc_gen",
        ":dialect_inc_gen",
        ":enums_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:SparseTensorEnums",
        "@llvm-project//mlir:SparseTensorPipelines",
        "@llvm-project//mlir:SparseTensorTransforms",
        "@llvm-project//mlir:SparseTensorTypesIncGen",
        "@llvm-project//mlir:SparseTensorUtils",
    ],
)
