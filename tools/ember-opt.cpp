
// #include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h" // from
// @llvm-project #include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      //
// from @llvm-project #include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h" //
// from @llvm-project #include
// "mlir/include/mlir/Dialect/Index/IR/IndexDialect.h" // from @llvm-project
// #include "mlir/include/mlir/Dialect/Index/IR/IndexOps.h" // from
// @llvm-project #include "mlir/include/mlir/Dialect/Math/IR/Math.h"      //
// from @llvm-project #include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h" //
// from @llvm-project #include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h" //
// from @llvm-project

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/include/mlir/InitAllDialects.h"
#include "mlir/include/mlir/InitAllPasses.h"
#include "mlir/include/mlir/Pass/PassManager.h"
#include "mlir/include/mlir/Pass/PassRegistry.h"
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"

#include "lib/Conversion/ScfToSlc/ScfToSlc.h"
#include "lib/Conversion/SlcToDlc/SlcToDlc.h"
#include "lib/Conversion/SlcVectorizer/SlcVectorizer.h"
#include "lib/Dialect/SLC/SlcDialect.h"
#include "lib/Dialect/SLCVEC/SlcVecDialect.h"
#include "lib/Transforms/BufferCompoundTypes/Passes.h"
#include "lib/Transforms/CallbackVectorizer/Passes.h"
#include "lib/Transforms/ChooseDecoupling/Passes.h"
#include "lib/Transforms/MoveToValOps/Passes.h"
#include "lib/Transforms/ReplaceToValOps/Passes.h"
#include "lib/Transforms/SimplifyCastOps/Passes.h"
#include "lib/Transforms/SimplifyMemOps/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::slc::SlcDialect>();
  registry.insert<mlir::slcvec::SlcVecDialect>();

  mlir::registerAllPasses();

  mlir::PassPipelineRegistration<> pipeline_scf_to_slc(
      "scf-to-slc", "Converts SCF to SLC", [](mlir::OpPassManager &pm) {
        // Choose operation decoupling
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::slc::createChooseDecouplingPass());
        // Converts
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createScfToSlcPass());
        // Move slc.to_val materialization operations into callbacks
        pm.addNestedPass<mlir::func::FuncOp>(mlir::slc::createMoveToValOps());
      });

  mlir::PassPipelineRegistration<> pipeline_optimize(
      "optimize", "Optimizes SLC", [](mlir::OpPassManager &pm) {
        // Vectorize SLC loops and streams
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::createSlcVectorizerPass(/*vectorLength=*/8));
        // Simplify cast operations
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::slc::createSimplifyCastOpsPass());
        // Vectorize SLC callbacks
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::slc::createCallbackVectorizerPass());
        // Buffers compound-types data marshaling
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::slc::createBufferCompoundTypes());
        // Replace simple slc.to_val materialization operations with
        // induction variables
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::slc::createReplaceToValOps());
        // Replace simple slc.to_val materialization operations with
        // induction variables
        pm.addNestedPass<mlir::func::FuncOp>(mlir::slc::createSimplifyMemOps());
      });

  mlir::PassPipelineRegistration<> pipeline_slc_to_dlc(
      "slc-to-dlc", "Converts SLC to DLC", [](mlir::OpPassManager &pm) {
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createSlcToDlcPass());
      });

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SLC Pass Driver", registry));
}
