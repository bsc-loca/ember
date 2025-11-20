#ifndef LIB_TRANSFORMS_CHOOSEDECOUPLING_PASSES_H_
#define LIB_TRANSFORMS_CHOOSEDECOUPLING_PASSES_H_

#include "mlir/include/mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace slc {

std::unique_ptr<Pass> createChooseDecoupling();

#define GEN_PASS_DECL
#include "lib/Transforms/ChooseDecoupling/Passes.h.inc"

/*struct ChooseDecouplingPipelineOptions
    : public PassPipelineOptions<ChooseDecouplingPipelineOptions> {
  PassOptions::Option<int32_t> streamShape{
      *this, "stream-shape",
      llvm::cl::desc(
          "Set the vector/block size (use 0 to use default scalar stream)."),
      llvm::cl::init(0)};
  PassOptions::Option<bool> enableBlockSparse{
      *this, "enable-block-sparse",
      llvm::cl::desc(
          "Set to block sparse mode (use false to use default vector stream)."),
      llvm::cl::init(false)};
};*/

std::unique_ptr<mlir::Pass> createChooseDecouplingPass();

// void registerChooseDecouplingPipeline();

} // namespace slc
} // namespace mlir

#endif // LIB_TRANSFORMS_CHOOSEDECOUPLING_PASSES_H_