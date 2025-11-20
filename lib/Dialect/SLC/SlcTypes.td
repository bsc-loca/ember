#ifndef LIB_DIALECT_SLC_SLCTYPES_TD
#define LIB_DIALECT_SLC_SLCTYPES_TD

include "SlcDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//class Slc_Type<string name, string typeMnemonic> : TypeDef<Slc_Dialect, name> {
//    let mnemonic = typeMnemonic;
//}

// Base class for SLC dialect types.
class Slc_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Slc_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let typeName = "slc." # typeMnemonic;
}

def IsSlcStreamTypePred : CPred<"::mlir::isa<slc::SlcStreamType>($_self)">;

class SlcStreamOf<list<Type> allowedTypes> : 
    ContainerType<AnyTypeOf<allowedTypes>, IsSlcStreamTypePred, 
            "::llvm::cast<::mlir::slc::SlcStreamType>($_self).getElementType()",
            "stream", "::mlir::slc::SlcStreamType">;

def Slc_Stream : Slc_Type<"SlcStream", "stream"> {
    let summary = "Streams produced on the SLC";
    let description = [{
        Values with slc_stream types represent streams produced by the SLC.
        It currently holds the elements of the stream and need to be converted back 
        to CPU value types before being used in the callback. 

        Syntax: TODO

        examples: TODO
    }];

    let parameters = (ins "Type":$elementType);
    let builders = [
        TypeBuilder<(ins "Type":$elementType)>,
        TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
            return $_get(elementType.getContext(), elementType);
        }]>
    ];
    let extraClassDeclaration = [{
    /// This is a builder type that keeps local references to arguments.
    /// Arguments that are passed into the builder must outlive the builder.
    class Builder;
    }];
    let assemblyFormat = "`<` $elementType `>`";
    //let skipDefaultBuilders=1;
}

class SlcStream<Type type>
    : ConfinedType<SlcStreamOf<[AnyType]>, [
          SubstLeaves<"$_self",
                      "::llvm::cast<::mlir::slc::SlcStreamType>($_self).getElementType()",
           type.predicate>],
           "SLC stream type with " # type.summary # " elements",
           "::mlir::slc::SlcStreamType">,
      SameBuildabilityAs<type, "::mlir::slc::SlcStreamType::get($_builder.get" # type #
                               "Type())"> {
  Type elementType = type;
}
/*
def IsSlcVecStreamTypePred : CPred<"::mlir::isa<slc::SlcVecStreamType>($_self)">;

class SlcVecStreamOf<list<Type> allowedTypes> : 
    ContainerType<AnyTypeOf<allowedTypes>, IsSlcVecStreamTypePred, 
            "::llvm::cast<::mlir::slc::SlcVecStreamType>($_self).getElementType()",
            "stream_vec", "::mlir::slc::SlcVecStreamType">;

def Slc_VecStream : Slc_Type<"SlcVecStream", "vec_stream", [ShapedTypeInterface]> {
  let summary = "Shaped vector stream for SLC vector types.";
  let description = [{}];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );
  let builders = [
    TypeBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType)>,
    TypeBuilderWithInferredContext<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType), [{
        return $_get(elementType.getContext(), shape, elementType);
    }]>,
    TypeBuilderWithInferredContext<(ins "VectorType":$vectorType), [{
        return $_get(vectorType.getContext(), vectorType.getShape(), vectorType.getElementType());
    }]>
  ];
  let extraClassDeclaration = [{
    using ShapedType::Trait<SlcVecStreamType>::getElementTypeBitWidth;
    using ShapedType::Trait<SlcVecStreamType>::getRank;
    using ShapedType::Trait<SlcVecStreamType>::getNumElements;
    using ShapedType::Trait<SlcVecStreamType>::isDynamicDim;
    using ShapedType::Trait<SlcVecStreamType>::hasStaticShape;
    using ShapedType::Trait<SlcVecStreamType>::getNumDynamicDims;
    using ShapedType::Trait<SlcVecStreamType>::getDimSize;
    using ShapedType::Trait<SlcVecStreamType>::getDynamicDimIndex;

    /// This is a builder type that keeps local references to arguments.
    /// Arguments that are passed into the builder must outlive the builder.
    class Builder;

    /// Returns true if the given type can be used as an element of a vector
    /// type. In particular, vectors can consist of integer, index, or float
    /// primitives.
    static bool isValidElementType(Type t) {
      return ::llvm::isa<IntegerType, IndexType, FloatType>(t);
    }

    SlcVecStreamType cloneWith(std::optional<ArrayRef<int64_t>> shape, Type elementType) const;

    bool hasRank() const {
        return true;
    }

  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}
*/
/*
def IsSlcBufferTypePred : CPred<"::mlir::isa<slc::SlcBufferType>($_self)">;

class SlcBufferOf<list<Type> allowedTypes> : 
    ContainerType<AnyTypeOf<allowedTypes>, IsSlcBufferTypePred, 
            "::llvm::cast<::mlir::slc::SlcBufferType>($_self).getElementType()",
            "buffer", "::mlir::slc::SlcBufferType">;

def Slc_Buffer : Slc_Type<"SlcBuffer", "buffer", [ShapedTypeInterface]> {
  let summary = "Shaped buffer for SLC compound types.";
  let description = [{}];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );
  let builders = [
    TypeBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType)>,
    TypeBuilderWithInferredContext<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType), [{
        return $_get(elementType.getContext(), shape, elementType);
    }]>
  ];
  let extraClassDeclaration = [{
    using ShapedType::Trait<SlcBufferType>::getElementTypeBitWidth;
    using ShapedType::Trait<SlcBufferType>::getRank;
    using ShapedType::Trait<SlcBufferType>::getNumElements;
    using ShapedType::Trait<SlcBufferType>::isDynamicDim;
    using ShapedType::Trait<SlcBufferType>::hasStaticShape;
    using ShapedType::Trait<SlcBufferType>::getNumDynamicDims;
    using ShapedType::Trait<SlcBufferType>::getDimSize;
    using ShapedType::Trait<SlcBufferType>::getDynamicDimIndex;

    /// This is a builder type that keeps local references to arguments.
    /// Arguments that are passed into the builder must outlive the builder.
    class Builder;

    /// Returns true if the given type can be used as an element of a buffer
    /// type. In particular, buffers can consist of integer, index, or float
    /// primitives.
    static bool isValidElementType(Type t) {
      return ::llvm::isa<IntegerType, IndexType, FloatType>(t);
    }

    SlcBufferType cloneWith(std::optional<ArrayRef<int64_t>> shape, Type elementType) const;

    bool hasRank() const {
        return true;
    }

  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}
*/
#endif /* LIB_DIALECT_SLC_SLCTYPES_TD */
