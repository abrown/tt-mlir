// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNINLINELINKEDKERNELS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Returns true if `func` is an injected external kernel, identified by the
// presence of a `ttnn.generic` op in its body. This restricts inlining to the
// kernels merged in by `ttir-link-external-functions` and leaves every other
// `func.call` (e.g. const-eval functions, which are invoked via the non-call
// `ttcore.load_cached` op) untouched.
static bool isLinkedKernel(func::FuncOp func) {
  bool hasGeneric = false;
  func.walk([&](GenericOp) { hasGeneric = true; });
  return hasGeneric;
}

class TTNNInlineLinkedKernels
    : public impl::TTNNInlineLinkedKernelsBase<TTNNInlineLinkedKernels> {
public:
  using impl::TTNNInlineLinkedKernelsBase<
      TTNNInlineLinkedKernels>::TTNNInlineLinkedKernelsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Collect call sites before mutating the IR.
    SmallVector<func::CallOp> kernelCalls;
    moduleOp.walk([&](func::CallOp callOp) {
      auto calleeFunc = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr()));
      if (calleeFunc && isLinkedKernel(calleeFunc)) {
        kernelCalls.push_back(callOp);
      }
    });

    SmallVector<func::FuncOp> inlinedFuncs;
    for (func::CallOp callOp : kernelCalls) {
      auto calleeFunc = cast<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr()));
      if (failed(inlineCall(callOp, calleeFunc))) {
        return signalPassFailure();
      }
      inlinedFuncs.push_back(calleeFunc);
    }

    // Erase the now-dead kernel entry functions. Helper kernel functions
    // referenced by the program attribute (via symbol refs, not func.call)
    // remain live and are retained.
    for (func::FuncOp func : inlinedFuncs) {
      if (SymbolTable::symbolKnownUseEmpty(func, moduleOp)) {
        func.erase();
      }
    }
  }

private:
  // Inlines `calleeFunc`'s single-block body at `callOp` by cloning its ops and
  // mapping the function arguments to the call operands. This mirrors the
  // approach used by `TTNNCollaspeD2M`; injected kernel entry functions have a
  // single block terminated by `func.return`.
  LogicalResult inlineCall(func::CallOp callOp, func::FuncOp calleeFunc) {
    if (!calleeFunc.getCallableRegion() ||
        !calleeFunc.getCallableRegion()->hasOneBlock()) {
      return callOp.emitOpError()
             << "cannot inline linked kernel '" << calleeFunc.getSymName()
             << "': expected a single-block body";
    }

    OpBuilder builder(callOp);
    IRMapping mapping;
    for (auto [arg, operand] :
         llvm::zip(calleeFunc.getArguments(), callOp.getOperands())) {
      mapping.map(arg, operand);
    }

    Block &calleeBody = calleeFunc.getCallableRegion()->front();
    for (Operation &op : calleeBody.without_terminator()) {
      builder.clone(op, mapping);
    }

    auto returnOp = dyn_cast<func::ReturnOp>(calleeBody.getTerminator());
    if (!returnOp) {
      return calleeFunc.emitOpError()
             << "linked kernel must end with func.return";
    }

    SmallVector<Value> results;
    results.reserve(returnOp.getNumOperands());
    for (Value operand : returnOp.getOperands()) {
      results.push_back(mapping.lookupOrDefault(operand));
    }

    if (results.size() != callOp.getNumResults()) {
      return callOp.emitOpError()
             << "result count mismatch when inlining linked kernel";
    }

    callOp.replaceAllUsesWith(results);
    callOp.erase();
    return success();
  }
};

} // namespace
} // namespace mlir::tt::ttnn
