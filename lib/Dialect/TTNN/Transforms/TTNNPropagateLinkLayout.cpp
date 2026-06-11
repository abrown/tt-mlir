// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNPROPAGATELINKLAYOUT
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Returns true if `castOp` is a layout/shape bridging cast inserted by the
// link-external-functions ABI: both source and result are ranked tensors with
// the same element type, but their types differ in `ttnn_layout` encoding).
static bool isLinkBridgingCast(tensor::CastOp castOp) {
  auto srcType = llvm::dyn_cast<RankedTensorType>(castOp.getSource().getType());
  auto resType = llvm::dyn_cast<RankedTensorType>(castOp.getResult().getType());
  return mlir::tensor::canFoldIntoConsumerOp(castOp) && srcType != resType;
}

class TTNNPropagateLinkLayout
    : public impl::TTNNPropagateLinkLayoutBase<TTNNPropagateLinkLayout> {
public:
  using impl::TTNNPropagateLinkLayoutBase<
      TTNNPropagateLinkLayout>::TTNNPropagateLinkLayoutBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Phase 1: forward graph values into `ttnn.generic` operands.
    //
    // For each `ttnn.generic`, forward the graph value (the source of a
    // link-ABI bridging cast) directly into the operand, replacing the
    // kernel-side type with the graph static shape and `ttnn_layout`.
    //
    // `ttnn.generic` has no results (it is destination-passing: it writes into
    // one of its tensor operands), so only operand types need rewriting. The
    // output-side cast that wraps the destination operand becomes a
    // `graph -> kernel -> graph` chain once the operand is forwarded, which
    // phase 2 collapses back to the graph value.
    moduleOp.walk([&](GenericOp genericOp) {
      for (OpOperand &operand : genericOp->getOpOperands()) {
        auto castOp = operand.get().getDefiningOp<tensor::CastOp>();
        if (!castOp || !isLinkBridgingCast(castOp)) {
          continue;
        }
        operand.set(castOp.getSource());
      }
    });

    // Phase 2: clean up the now-redundant bridging casts. All of the
    // simplifications below are semantics-preserving (identical to what the
    // `tensor.cast` canonicalizer would do), but applying them here keeps the
    // pass self-contained and the resulting IR free of vestigial casts.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<tensor::CastOp> casts;
      moduleOp.walk([&](tensor::CastOp castOp) { casts.push_back(castOp); });

      for (tensor::CastOp castOp : casts) {
        // Dead cast: no remaining users.
        if (castOp.getResult().use_empty()) {
          castOp.erase();
          changed = true;
          continue;
        }
        // Identity cast: source and result types match.
        if (castOp.getSource().getType() == castOp.getResult().getType()) {
          castOp.getResult().replaceAllUsesWith(castOp.getSource());
          castOp.erase();
          changed = true;
          continue;
        }
        // Chained round-trip cast: `x -> y -> x`. Replace this cast's result
        // with the grandparent value if the round-trip returns to its type.
        if (auto parent = castOp.getSource().getDefiningOp<tensor::CastOp>()) {
          Value grandparent = parent.getSource();
          if (grandparent.getType() == castOp.getResult().getType()) {
            castOp.getResult().replaceAllUsesWith(grandparent);
            castOp.erase();
            changed = true;
            continue;
          }
        }
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::ttnn
