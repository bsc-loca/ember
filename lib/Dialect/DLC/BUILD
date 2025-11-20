load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "td_files",
    srcs = [
        "DlcDialect.td",
        "DlcAttr.td",
        "DlcTypes.td",
        "DlcEnums.td",
    ],
    includes = ["@dlc//include", "//lib/Dialect/DLC",],
    deps = [
        "//lib/Dialect/SLC:td_files",
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
            "DlcDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "DlcDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "DlcDialect.td",
    deps = [
        ":td_files",
        "//lib/Dialect/SLC:dialect_inc_gen",
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
            "DlcTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "DlcTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "DlcTypes.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        "//lib/Dialect/SLC:types_inc_gen",
    ],
)

gentbl_cc_library(
    name = "attrs_inc_gen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "DlcAttrDefs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "DlcAttrDefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "DlcAttr.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        "//lib/Dialect/SLC:attrs_inc_gen",
    ],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "DlcOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "DlcOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "DlcOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        ":types_inc_gen",
        "//lib/Dialect/SLC:ops_inc_gen",
    ],
)

gentbl_cc_library(
    name = "enums_inc_gen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "DlcEnumDefs.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "DlcEnumDefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "DlcEnums.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        "//lib/Dialect/SLC:enums_inc_gen",
    ],
)

cc_library(
    name = "Dlc",
    srcs = [
        "DlcDialect.cpp",
        "DlcOps.cpp",
        "DlcTypes.cpp"
        ],
    hdrs = [
        "DlcDialect.h",
        "DlcOps.h",
        "DlcTypes.h",
    ],
    deps = [
        ":attrs_inc_gen",
        ":dialect_inc_gen",
        ":enums_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "//lib/Dialect/SLC:Slc",
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
