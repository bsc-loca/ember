#ifndef LIB_DIALECT_SLCVEC_SLCVECTYPES_TD
#define LIB_DIALECT_SLCVEC_SLCVECTYPES_TD

include "SlcVecDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//include "SlcTypes.td"

// Base class for SLCVEC dialect types.
class SlcVec_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<SlcVec_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let typeName = "slcvec." # typeMnemonic;
}

def IsSlcStreamVecTypePred : CPred<"::mlir::isa<slcvec::SlcStreamVecType>($_self)">;

class SlcStreamVecOf<list<Type> allowedTypes> : 
    ContainerType<AnyTypeOf<allowedTypes>, IsSlcStreamVecTypePred, 
            "::llvm::cast<::mlir::slcvec::SlcStreamVecType>($_self).getElementType()",
            "stream_vec", "::mlir::slcvec::SlcStreamVecType">;

def SlcVec_StreamVec : SlcVec_Type<"SlcStreamVec", "buffer", [ShapedTypeInterface]> {
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
    }]>
  ];
  let extraClassDeclaration = [{
    using ShapedType::Trait<SlcStreamVecType>::getElementTypeBitWidth;
    using ShapedType::Trait<SlcStreamVecType>::getRank;
    using ShapedType::Trait<SlcStreamVecType>::getNumElements;
    using ShapedType::Trait<SlcStreamVecType>::isDynamicDim;
    using ShapedType::Trait<SlcStreamVecType>::hasStaticShape;
    using ShapedType::Trait<SlcStreamVecType>::getNumDynamicDims;
    using ShapedType::Trait<SlcStreamVecType>::getDimSize;
    using ShapedType::Trait<SlcStreamVecType>::getDynamicDimIndex;

    /// This is a builder type that keeps local references to arguments.
    /// Arguments that are passed into the builder must outlive the builder.
    class Builder;

    /// Returns true if the given type can be used as an element of a vector
    /// type. In particular, vectors can consist of integer, index, or float
    /// primitives.
    static bool isValidElementType(Type t) {
      return ::llvm::isa<IntegerType, IndexType, FloatType>(t);
    }

    SlcStreamVecType cloneWith(std::optional<ArrayRef<int64_t>> shape, Type elementType) const;

    bool hasRank() const {
        return true;
    }

  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

#endif /* LIB_DIALECT_SLCVEC_SLCVECTYPES_TD */
