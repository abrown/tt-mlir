// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_CBUTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_CBUTILS_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::tt::d2m {

class GenericOp;

Value getOrCreateCB(RewriterBase &rewriter, GenericOp generic, Block *block,
                    unsigned cbOperandIndex);

/// The CB port a scalar L1 access touches (see
/// d2m::utils::isScalarL1AccessType), or nullopt if `op` is not such an access
/// or the buffer it names is not an operand of the enclosing generic -- a
/// region-local scratch allocation, say.
///
/// Scalar accesses name the operand buffer directly rather than going through a
/// `d2m.wait`/`d2m.reserve` on a CB handle, so they cannot be matched against a
/// CB by value; the buffer is traced through any views back to the generic's
/// operand list, whose index is the CB port.
std::optional<unsigned> getScalarL1AccessPort(Operation *op);

/// Trace a value through view-like operations (subview, expand_shape, etc.)
/// and return the defining op if it matches OpT.  Returns null otherwise.
template <typename OpT>
OpT traceToDefiningOp(Value value) {
  while (value) {
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp) {
      return nullptr;
    }
    if (auto op = mlir::dyn_cast<OpT>(definingOp)) {
      return op;
    }
    if (mlir::isa<mlir::ViewLikeOpInterface>(definingOp)) {
      value = definingOp->getOperand(0);
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

} // namespace mlir::tt::d2m

#endif
