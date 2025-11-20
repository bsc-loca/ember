# The Ember compiler

This repo contains an MLIR implementation of **Ember**, a compiler to lower PyTorch and TensorFlow embedding operations to **Decoupled Access-Execute (DAE)** architectures. This page discusses how we implemented the main components of the Ember compiler, and is structured as follow:
- [MLIR terminology](#mlir-terminology)  
- [Structured DAE dialects](#structured-dae-dialects)  
  - [SLC dialect (`lib\Dialect\SLC`)](#slc-dialect-libdialectslc)  
  - [SLCV dialect (`lib\Dialect\SLCV`)](#slcv-dialect-libdialectslcv)  
  - [DLC dialect (`lib\Dialect\DLC`)](#dlc-dialect-libdialectdlc)  
- [SCF → SLC lowering pipeline (`tools\ember-opt.cpp` with `--scf-to-slc` flag)](#scf--slc-lowering-pipeline-toolsember-optcpp-with---scf-to-slc-flag)  
- [SLC optimization pipeline (`tools\ember-opt.cpp` with `--optimize` flag)](#slc-optimization-pipeline-toolsember-optcpp-with---optimize-flag)
- [SLC → DLC lowering pipeline (`tools\ember-opt.cpp` with `--slc-to-dlc` flag)](#slc--dlc-lowering-pipeline-toolsember-optcpp-with---slc-to-dlc-flag)  
- [Notes on generating code for specific DAE architectures (e.g. CPU+TMU)](#notes-on-generating-code-for-specific-dae-architectures-eg-cputmu)  
- [Build & run instructions](#build--run-instructions)  
- [Artifact Evaluation](#artifact-evaluation)   

For deeper design discussions, please refer to the **Ember paper**.

---

## MLIR terminology

This is the **MLIR terminology** we used across the project:
- **Dialect**: a set of custom types and operations that define one level of abstraction. In this repo, all dialects are placed in the `lib\Dialect` folder.
- **Target dialects**: dialects meant to generate machine code.
- **Conversion pass**: converts operations from one dialect into operations in another dialect, possibly transforming types. In this repo, all conversion passes are placed in the `lib\Conversion` folder.
- **Transformation pass**: optimizes operations within the same dialect. In this repo, all transformation passes are placed in the `lib\Transforms` folder.

By chaining multiple conversion and transformation passes, MLIR progressively lowers high-level embedding operations to DAE code that matches the performance of handoptimized code.

## Structured DAE dialects

The key feature of Ember is to lower embedding operations through multiple IRs/dialects to optimize DAE code at different abstractions levels. In this repo, all dialects are placed in the `lib\Dialect` folder, and are briefly introduced here.

### SLC dialect (`lib\Dialect\SLC`)

The **Structured Lookup-Compute (SLC) dialect** is a natural extension of the MLIR Structured Control FLow (SCF) dialect for DAE code. The table below summarizes the main operations in the SLC dialect; the complete grammar can be found in the *Ember* paper.

| Op Name      | ins | outs | Regions | Description |
|--------------|-----|------|---------|-------------|
| `for` | `[Index,Stream<Index>]:$lowerBound`<br>`[Index,Stream<Index>]:$upperBound`<br>`Index:$step`<br>`Variadic<Index>:$iterArgs` | `Variadic<Index>:$results` | One | TMU `for` loop operation.<br>Provides induction variable and iteration count<br>in `Stream<Index>` variables in its loop body. |
| `mem_str` | `MemRef<AnyType>:$memref`<br>`Variadic<[Index,Stream<Index>]>:$indices` | `Stream<AnyType>:$result` |  | A stream that loads data from a `memref` location given<br>a set of static/dynamic `indices` to index every `memref` dimension. |
| `alu_str` | `OpcodeAttr:$opcode`<br>`[Index,Stream<Index>]:$op1`<br>`[Index,Stream<Index>]:$op2` | `Stream<Index>:$result` |  | A stream that performs integer arithmetic operations over<br>two static or dynamic `op`erands. |
| `fwd_str` | `[AnyType,Stream<AnyType>]:$op` | `Stream<Index>:$result` |  | A stream that repeats constant variables or<br>forwards streams from the previous layer. |
| `str_to_val` | `Stream<AnyType>:$stream` | `AnyType:$value` |  | Converts a TMU stream to a value to be used in a callback. |
| `callback` |  | `Variadic<Index>:$indVar` | One | Callback operation containing stream-to-value ops and CPU code. |

`tests/eb_slc_O0.mlir` contains the SLC representation of the EmbeddingBag function. 

As described in the *Ember* paper, the SLC IR introduces the concept of **compute callbacks** on top of SCF. This separation allows:
- **Tensor traversal operations** (i.e., iteration and memory access) to be mapped to the **access unit**, expressed with SCF-like constructs and streams.
- **Compute operations** (e.g., embedding vector accumulation) to be mapped to the **compute unit**, encapsulated within compute callbacks.

**Stream-to-value conversion** is handled using a dedicated `str_to_val` operation. In this way, the SLC dialect explicitly defines which operations belong to the access unit and which to the compute unit, while preserving the overall dataflow.

### SLCV dialect (`lib\Dialect\SLCV`)

The **SLCV dialect** is the **vectorized extension** of the MLIR SLC dialect. The table below summarizes the main operations in the SLCV dialect.

| Op Name    | ins | outs | Regions | Description |
|------------|-----|------|---------|-------------|
| `vec_for` | `[Index,Stream<[Index,Vector<Index>]>]:$lowerBound`<br>`[Index,Stream<[Index,Vector<Index>]>]:$upperBound`<br>`Index:$step`<br>`[Vector<I1>,Stream<Vector<I1>>]:$inMask`<br>`APIntAttr:$vectorLength`<br>`LoopConfigAttr:$loopConfig`<br>`Variadic<Index>:$iterArgs` | `Variadic<Index>:$results` | One | TMU vectorized `for` loop operation.<br>Besides induction variable and iteration count as `Stream<Vector<Index>>`, it also provides an output mask as `Stream<Vector<I1>>` variable in its loop body. |
| `vec_mem_str` | `MemRef<AnyType>:$memref`<br>`Variadic<[Index,Stream<[Index,Vector<Index>]>]>:$indices` | `Stream<Vector<AnyType>>:$result` |  | A stream that loads vectors from a `memref`<br>location given a set of static/dynamic vector<br>`indices` to index every `memref` dimension. |
| `vec_alu_str` | `Variadic<[Index,Stream<[Index,Vector<Index>]>]>:$op1`<br>`Variadic<[Index,Stream<[Index,Vector<Index>]>]>:$op2` | `Stream<Vector<Index>>:$result` |  | Registers a stream that performs integer arithmetic<br>operations over two static or dynamic vector operands. |

`tests/eb_slc_O3.mlir` contains the SLCV representation of the EmbeddingBag function.

Compared to the `slc.for` operation, the `slcv.for` operation introduces several vector-specific features:
1. **Vector length attribute**
2. **Loop configuration attribute** ∈ {scalar, bcast, vector}
3. Instantiation of **vectorized induction variables** and **iteration counters**
4. Introduction of **masks** to handle loop boundaries that do not align with the vector length

These vectorized induction variables, iteration counters, and masks enable **vectorized index computation** and **data loading** through the vector stream (e.g. `slcv.mem_str`).

### DLC dialect (`lib\Dialect\DLC`)

The **Decoupled Lookup-Compute (DLC) dialect** is a lower level representation of DAE embedding operations. The table below summarizes the main operations in the DLC dialect.

| Op Name      | ins | outs | Regions| Description |
|--------------|-----|------|--------|-------------|
| `compute_loop` | `DenseI64ArrayAttr:$cases` | `Variadic<AnyType>:$results` | Many | Switch-case like construct to wrap all callbacks triggered from the access unit to run on the compute unit. |
| `pop_operand` |  | `AnyType:$result` | | Within a callback, pops operand from the data queue. |
| `config_access` |  |  | | Wraps the following primitives to configure the access unit. |
| `new_tu` | `[Index,Stream<Index>]:$lowerBound`<br>`[Index,Stream<Index>]:$upperBound`<br>`Index:$step` | `DlcTuType:$results` | | Low-level TMU `for` loop operation.<br>Provides induction variable and iteration count<br>through custom operations reported below. |
| `get_ind_var` | `DlcTuType:$tu` | `Stream<AnyType>:$result` | | Return the induction variable of a TU. |
| `register_callback` | `DlcTuType:$tu`<br>`OpAttrType:$event`<br>`I64Attr:$id` | `Stream<Index>:$result` | | Registers callback to push control tokens over TU events. |
| `register_operand` | `DlcTuType:$tu`<br>`OpAttrType:$event`<br>`Stream<AnyType>:$value` | | | Registers callback to push operands over TU events. |

`tests/eb_dlc.mlir` contains the DLC representation of the EmbeddingBag function.

As described in the *Ember* paper, the DLC IR actually separates access and compute code in the `config_access` and `compute_loop` operations, respectively. Specifically:
- `config_access` configures the access unit (with dataflow code) to traverse loops in `new_tu`s (where a TU is a Traversal Unit), which control data loading from the previously defined SLC streams, and push data and control operands into the queues with the `register_operand` and `register_callback` operations. The stream containin the loop induction variable can be accessed through the `get_ind_var` operation.
- `compute_loop` just wraps all the callbacks in a switch-like construct that cases over the callback IDs read from the control queue. The `pop_operand` pops an operand from the data queue.

In this way, the DLC IR faclitates lowering of access and execute code to target dialects like LLVM. 

## SCF → SLC lowering pipeline (`tools\ember-opt.cpp` with `--scf-to-slc` flag)

Ember **lowers PyTorch/TensorFlow code to MLIR SCF code** through state-of-the-art tools like [torch-mlir](https://github.com/llvm/torch-mlir) or [MPACT](https://developers.google.com/mlir-sparsifier). Then, thourgh the `tools\ember-opt.cpp` pipeline, Ember lowers SCF code to SLC code to perform global optimizations before lowering to decoupled code. All the passes within such pipeline are either placed in the `lib\Conversion` or `lib\Transforms` folder.

In particular, Ember **lowers SCF to SLC** in three main steps:
1. **Recursive traversal with `lib\Transforms\ChooseDecoupling`**
   This transformation pass scans the SCF IR and:
   - Marks loop offloading candidates (by inserting special operations in their bodies)
   - Marks loop vectorization candidates (also via special operations)
   - Places `beg/ite/end` callbacks around offloading candidates
   - Keeps non-offloaded operations outside callbacks
   - Moves offloaded operations inside callbacks
2. **Conversion with `lib\Conversion\ScfToSlc`**
   A match-and-rewrite pass that lowers the marked SCF constructs into SLC operations, specifically:
   - All SCF loops marked for offloading
   - All `load`, `add`, and `mul` operations outside callbacks
3. **Canonicalization with `lib\Transforms\MoveToValOps` pass**
   Match and rewrite pass that moves `to_val` ops into callbacks
     
When in the prject folder, this pipeline can be run with `bazel run //tools:ember-opt -- --debug --debug-only=dialect_conversion --mlir-print-ir-after-all --allow-unregistered-dialect --scf-to-slc $PWD/tests/eb_scf.mlir`

## SLC optimization pipeline (`tools\ember-opt.cpp` with `--optimize` flag)

At first, Ember **vectorizes code** using a two-step process:
1. **Access code vectorization** with the `lib\Conversion\SlcVectorizer` pass
2. **Execute code vectorization** with the `lib\Transforms\CallbackVectorizer` pass

To ensure type correctness, the first conversion pass inserts **temporary vector-to-scalar cast operations** before each stream-to-value cast inside callbacks. The second pattern-rewriting pass then iteratively transforms the IR by:
- Using the temporary cast operations to track the *frontier* of vectorized instructions
- Replacing each scalar operation with a vectorized counterpart, followed by a new temporary cast
- Advancing the frontier until all eligible operations are vectorized

All the **other optimizations** are implemented as match-and-rewrite passes that walk the IR to detect and transform specific patterns.

Overall, the full lowering and optimization `tools\ember-opt.cpp` pipeline consists of, in order:
1. **`lib\Transforms\ChooseDecoupling` pass**
   - Marks loop offloading/vectorization candidates
   - Places `beg/ite/end` callbacks
   - Moves ops inside/outside callbacks
2. **`lib\Conversion\ScfToSlc` pass**
   - Converts SCF loops/loads/add/mul into SLC ops
3. **`lib\Conversion\SlcVectorizer` pass**
   - Vectorizes marked loops (not callbacks)
4. **`lib\Transforms\SimplifyCastOps` pass**
   - Cleans up temporary cast ops
5. **`lib\Transforms\CallbackVectorizer` pass**
   - Vectorizes callback contents
6. **`lib\Transforms\BufferCompoundTypes` pass**
   - Bufferizes compound types (details in paper)
7. **`lib\Transforms\ReplaceToValOps` pass**
   - Queue alignment (details in paper)
8. **`lib\Transforms\SimplifyMemOps` pass**
   - Optimizes vector loads with sequential indices

When in the prject folder, this pipeline can be run with `bazel-bin/tools/ember-opt --debug '--debug-only=dialect_conversion' --mlir-print-ir-after-all --allow-unregistered-dialect --optimize $PWD/tests/eb_slc_O0.mlir`


## SLC → DLC lowering pipeline (`tools\ember-opt.cpp` with `--slc-to-dlc` flag)

Ember lowers SLC to DLC code through a custom decoupling pass, which performs the following steps:
- Finds the outermost SLC(V) loop(s).
- Places a `config_accss` and `compute_loop` right before.
- Traverses the SLC(V) hierarchy and
  - Moves all streams into the `config_access` operation.
  - Appends all callbacks to the `compute_loop` function and adds `register_callback` operations accordingly.
  - Adds the marshaling logic (`register_operand` and `pop_operand`) according to the `to_stream` operations within callbacks.
 On CPUs, the `compute_loop` operation can later be lowered to a while loop containing a if-then-else chain. 

When in the prject folder, this pipeline can be run with `bazel run //tools:ember-opt -- --debug --debug-only=dialect_conversion --mlir-print-ir-after-all --allow-unregistered-dialect --slc-to-dlc $PWD/tests/eb_slc_O3.mlir`.

## Notes on generating code for specific DAE architectures (e.g. CPU+TMU) 

As a final step, Ember lower access and execute code in the DLC IR to the **target dialect** of a given DAE architecture. We did not include such last step in this repo as entirely target dependent. However, we discuss the key aspects of lowering to the **TMU-CPU DAE architecture** discussed in the Ember paper to help users understand the high-level process.

For a CPU+TMU target, Ember lowers optimized SLC operations to the **TMU** and **LLVM dialects**. The LLVM dialect also provides the necessary boilerplate for the TMU-CPU queuing interface.

The table below illustrates a simplified version of the TMU dialect.

| Op Name | ins | outs | Description |
|---------|-----|------|-------------|
| `set_dns_tu` | `Index:$lowerBound`<br>`Index:$upperBound`<br>`Index:$step` | `Tu:$tu`<br>`Stream<Index>:$indVar`<br>`Stream<Index>:$counter` | Registers a dense `for` loop with static `Index` bounds on a new TU.<br>Returns the TU reference, induction variable, iteration count. |
| `set_rng_tu` | `Stream<Index>:$lowerBound`<br>`Stream<Index>:$upperBound`<br>`Index:$step` | `Tu:$tu`<br>`Stream<Index>:$indVar`<br>`Stream<Index>:$counter` | Registers a dense `for` loop with static `Index` bounds on a new TU.<br>Returns the TU reference, induction variable, iteration count. |
| `set_mem_str` | `Tu:$tu`<br>`MemRef<AnyType>:$memref`<br>`[Index,Stream<Index>]:$indices` | `Stream<AnyType>:$result` | Registers a stream that loads data from a `memref` location given<br>a set of static/dynamic `indices` to index every `memref` dimension. |
| `set_alu_str` | `Tu:$tu`<br>`OpcodeAttr:$opcode`<br>`[Index,Stream<Index>]:$op1`<br>`[Index,Stream<Index>]:$op2` | `Stream<Index>:$result` | Registers a stream that performs integer arithmetic operations over<br>two static or dynamic `op`erands. |
| `set_fwd_str` | `Tu:$tu`<br>`[AnyType,Stream<AnyType>]:$op` | `Stream<Index>:$result` | Registers a stream that repeats constant variables or<br>forwards streams from the previous layer. |
| `set_single_layer` | `Tu:$tu` | `Layer:$layer` | Registers a layer with a single TU. |
| `set_lockstep_layer` | `Variadic<Tu>:$tus` | `Layer:$layer` | Registers a layer to iterate multiple TUs in lockstep. |
| `set_bcast_layer` | `Tu:$tu` | `Layer:$layer` | Registers a layer to `broadcast` the content of a `single` layer<br>to a `lockstep` layer. |
| `set_operand` | `Stream<AnyType>:$stream`<br>`Layer:$layer`<br>`EventAttr:$event` | | Registers a marshaling operation to push the head of a `stream`<br>over a given `event` of a given `layer`. |
| `set_control_token` | `Layer:$layer`<br>`EventAttr:$event` | `Stream<Index>:$result` | Registers a marshaling operation to push a control token<br>over a given `event` of a given `layer`. |

Overall, the **TMU dialect** defines low-level SSA **dataflow operations** to initialize and connect TMU components, assuming unconstrained (logical) resources. Resource allocation to physical operations is automatically performed during code generation.

**Similarly to the DLC IR**, the TMU dialect defines
- Traversal Units (TUs) to iterate loops
- streams to load data and perform arithmetic computation.
- primitives to push control tokens and (vector) operands into the TMU output queue, defining both the control path and data path of computations executed on the core.
**Conversely from the DLC IR**, the TMU IR requires explicit
- index computation
- coordination across traversal units to implement
  - vectorization across TMU lanes
  - nested traversal across TMU layers.

The TMU IR allows Ember to optimize TMU resource usage and generate **resource-constrained TMU code**. Ember then lowers the TMU and LLVM dialects to machine code:
- **CPU code** is generated via the Clang compiler  
- **TMU primitives** are generated via a TMU code generator  

Finally, Ember maps TMU operations to physical TMU components, transforming SSA-based code into **dataflow code with constrained resources**. The Ember paper demonstrates that by lowering embedding operations through multiple IRs, Ember can implement all optimizations to match the performance of hand-optimized code.

## Build & run instructions
We build and run the optimization and lowering pipelines using Bazel. Installation instructions are available at the [official Bazel documentation](https://bazel.build/install). This section lists the commands to lower the embedding bag function from SCF to DLC IR. All commands should be run from the main folder. For convenience, all intermediate input and output files of the compilation pipeline can be found in the `tests` folder.

### Installation
1. Download and install `bazel` 7 or older at [bazel.build](https://bazel.build/install/) , e.g., `brew install bazel@7`
2. You may have to add the bazel install to your path. An example command is `echo 'export PATH="/opt/homebrew/opt/bazel@7/bin:$PATH"' >> ~/.zshrc` or to shell rc of choice. Then, open a new terminal or run `source .zshrc`.
3. To confirm, run `bazel run //tools:ember-opt -- --debug --debug-only=dialect_conversion --mlir-print-ir-after-all --allow-unregistered-dialect --scf-to-slc $PWD/tests/eb_scf.mlir`
4. You also need Python and a version of matplotlib installed

Note: we are having issues with M-series Macbooks. Please use an Intel Mac or linux machine to build bazel. 

### Run commands

The following command **lowers SCF to SLC**: `bazel run //tools:ember-opt -- --debug --debug-only=dialect_conversion --mlir-print-ir-after-all --allow-unregistered-dialect --scf-to-slc $PWD/tests/eb_scf.mlir`

The following command **optimizes SLC**: `bazel-bin/tools/ember-opt --debug '--debug-only=dialect_conversion' --mlir-print-ir-after-all --allow-unregistered-dialect --optimize $PWD/tests/eb_slc_O0.mlir`

The following command **lowers SLC to DLC**: `bazel run //tools:ember-opt -- --debug --debug-only=dialect_conversion --mlir-print-ir-after-all --allow-unregistered-dialect --slc-to-dlc $PWD/tests/eb_slc_O3.mlir`

## Artifact Evaluation

The following results from the *Ember* paper are made reproducible through this artifact:

- **Figure 10:** Demonstrates the lowering from SCF to SLC. Figure 10(b) can be generated using the *“lowers SCF to SLC”* command described in the [Build & Run Instructions](#build--run-instructions) section above.

- **Figure 12:** Illustrates the SLC optimization pipeline, starting from the code shown in Figure 10(b) (corresponding to Figure 12(a)). The fully optimized version (Figure 12(d)) can be reproduced using the *“optimizes SLC”* command from the [Build & Run Instructions](#build--run-instructions) section above. Intermediate optimization stages (Figures 12(b) and 12(c)) can be generated by modifying the optimization pipeline in `tools/ember-opt.cpp`.

- **Figures 13–16:** Demonstrate the performance potential of *Ember*. Each figure can be generated by navigating to the AE folder (`cd AE`) and running its corresponding Pyton script (e.g., `python fig_13.py`). Each script produces a corresponding PDF file (e.g., `fig_13.pdf`) derived from raw simulation statistics (such as cycles, throughput, and cache hits/misses), originally collected using the gem5 simulator. Since the gem5 simulator cannot be open-sourced, these statistics are embedded within the provided scripts.
