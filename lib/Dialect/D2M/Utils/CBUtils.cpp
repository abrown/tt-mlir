// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::tt::d2m {


std::optional<unsigned> getScalarL1AccessPort(Operation *op) {
  Value memref = utils::getScalarL1AccessMemref(op);
  if (!memref) {
    return std::nullopt;
  }
  auto generic = op->getParentOfType<GenericOp>();
  if (!generic) {
    return std::nullopt;
  }
  // Trace back through views onto the operand.
  while (Operation *def = memref.getDefiningOp()) {
    if (!mlir::isa<mlir::ViewLikeOpInterface>(def)) {
      break;
    }
    memref = def->getOperand(0);
  }
  // Once the access has been bracketed by d2m-insert-scalar-access-cb it names
  // the acquired buffer rather than the operand, so resolve the CB handle.
  Value cb;
  if (auto waitOp = memref.getDefiningOp<WaitOp>()) {
    cb = waitOp.getCb();
  } else if (auto reserveOp = memref.getDefiningOp<ReserveOp>()) {
    cb = reserveOp.getCb();
  }
  if (cb) {
    if (auto getCBOp = cb.getDefiningOp<GetCBOp>()) {
      return static_cast<unsigned>(getCBOp.getCbOperandIdx());
    }
    return std::nullopt;
  }
  for (unsigned i = 0; i < generic->getNumOperands(); ++i) {
    if (generic->getOperand(i) == memref) {
      return i;
    }
  }
  return std::nullopt;
}

Value getOrCreateCB(RewriterBase &rewriter, GenericOp generic, Block *block,
                    unsigned cbOperandIndex) {
  // If CB already exists, return it
  Value cb = nullptr;
  block->walk([&](GetCBOp getCBOp) {
    if (getCBOp.getCbOperandIdx() == cbOperandIndex) {
      cb = getCBOp.getResult();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (cb) {
    return cb;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(block);
  Value localBuffer = generic.getOperand(cbOperandIndex);
  cb = rewriter
           .create<GetCBOp>(
               generic.getLoc(),
               CBType::get(rewriter.getContext(),
                           mlir::cast<ShapedType>(localBuffer.getType())),
               cbOperandIndex,
               ResolutionStageAttr::get(rewriter.getContext(),
                                        ResolutionStage::Compile))
           .getResult();
  return cb;
}

} // namespace mlir::tt::d2m
