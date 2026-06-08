// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRLINKEXTERNALFUNCTIONS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// If `path` is relative, prepend the directory of the MLIR file in which
// `moduleOp` was defined.  The source location is obtained from the module's
// `FileLineColLoc`, which is set automatically when `ttmlir-opt` (or any
// `mlir::parseSourceFile` caller) loads a file from disk.
// If the module has no file location, or if `path` is already absolute, the
// original `path` is returned unchanged.
static std::string resolvePath(StringRef path, ModuleOp moduleOp) {
  if (llvm::sys::path::is_absolute(path)) {
    return path.str();
  }

  // Walk the location chain to find a FileLineColLoc.
  StringRef moduleFile;
  moduleOp.getLoc()->walk([&](mlir::Location loc) -> mlir::WalkResult {
    if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
      moduleFile = fileLoc.getFilename().getValue();
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  if (moduleFile.empty()) {
    return path.str();
  }

  llvm::SmallString<256> resolved(llvm::sys::path::parent_path(moduleFile));
  llvm::sys::path::append(resolved, path);
  return resolved.str().str();
}

// Merges every symbol-defining op from `externalModule` into `destModule`.
//
// For each symbol in the external module whose name already exists in the
// destination module, a unique name is generated (by appending "_0", "_1", …).
// All uses of the original name *inside* the external module are updated via
// `SymbolTable::replaceAllSymbolUses` before the op is moved, so that
// cross-symbol references within the incoming module remain consistent.
//
// Returns a map { originalName → finalName } for every symbol that was
// renamed.  Symbols whose name did not change are absent from the map.
// Returns failure if renaming or verification fails.
static FailureOr<llvm::StringMap<std::string>>
mergeExternalModule(ModuleOp destModule,
                    mlir::OwningOpRef<ModuleOp> &externalModule) {
  MLIRContext *ctx = destModule.getContext();

  // Build the set of symbol names already present in the destination module.
  llvm::StringSet<> usedNames;
  for (auto &op : destModule.getBody()->getOperations()) {
    if (auto symAttr =
            op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
      usedNames.insert(symAttr.getValue());
    }
  }

  // Compute renames: walk every symbol in the external module and, if its name
  // collides with `usedNames`, pick a fresh name. `usedNames` is updated as we
  // go so that symbols within the external module also don't collide with each
  // other after renaming.
  llvm::StringMap<std::string> renameMap;
  for (auto &op : externalModule->getBody()->getOperations()) {
    auto symAttr =
        op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    if (!symAttr) {
      continue;
    }
    StringRef origName = symAttr.getValue();
    if (!usedNames.contains(origName)) {
      usedNames.insert(origName);
      continue;
    }
    // Choose a unique name by appending "_N".
    std::string newName;
    for (unsigned i = 0;; ++i) {
      newName = (origName + "_" + Twine(i)).str();
      if (!usedNames.contains(newName)) {
        break;
      }
    }
    renameMap[origName] = newName;
    usedNames.insert(newName);
  }

  // Apply each rename inside the external module:
  //   1. Replace all *uses* of the old name (call sites, symbol references).
  //   2. Rename the defining op itself.
  // Step 1 must precede step 2 because `SymbolTable::replaceAllSymbolUses`
  // searches by name attribute, which would not find the op after renaming.
  for (auto &[origName, newName] : renameMap) {
    StringAttr origAttr = StringAttr::get(ctx, origName);
    StringAttr newAttr = StringAttr::get(ctx, newName);

    if (failed(SymbolTable::replaceAllSymbolUses(origAttr, newAttr,
                                                 externalModule.get()))) {
      externalModule->emitError()
          << "failed to replace uses of symbol '" << origName << "' with '"
          << newName << "' inside external module";
      return failure();
    }

    // Rename the definition: `replaceAllSymbolUses` only touches *uses*, so the
    // `sym_name` attribute on the defining op is still the original name.
    Operation *symOp =
        SymbolTable::lookupSymbolIn(externalModule.get(), origAttr);
    assert(symOp && "defining op must still be findable by original name");
    symOp->setAttr(SymbolTable::getSymbolAttrName(), newAttr);
  }

  // Move external module body to the destination module, marking all incoming
  // symbols as private. Implementation note: `splice` uses
  // `transferNodesFromList` which atomically updates each op's block pointer
  // without a `remove`+`push_back` round-trip (which triggers an "already in an
  // operation block" assertion in `addNodeToList`).
  for (auto &op : externalModule->getBody()->getOperations()) {
    if (op.hasAttr(SymbolTable::getSymbolAttrName())) {
      op.setAttr(SymbolTable::getVisibilityAttrName(),
                 StringAttr::get(ctx, "private"));
    }
  }
  Block *srcBlock = externalModule->getBody();
  Block *dstBlock = destModule.getBody();
  dstBlock->getOperations().splice(dstBlock->end(), srcBlock->getOperations());

  return renameMap;
}

// Extracts a scalar value from a 0-D or 1-D single-element ranked tensor
// using `tensor.extract`. Returns an error if the tensor does not have
// exactly one element.
static llvm::Expected<Value> adaptTensorToScalar(OpBuilder &builder,
                                                 Location loc, Value caller) {
  assert(mlir::isa<RankedTensorType>(caller.getType()) &&
         "expected caller to provide a ranked tensor for scalar argument");
  auto ty = mlir::cast<RankedTensorType>(caller.getType());
  if (ty.getNumElements() != 1) {
    return llvm::createStringError(llvm::formatv(
        "expected 0-D or 1-D ranked tensor for scalar argument: found {0}",
        caller.getType()));
  }
  auto index = ValueRange{};
  // A `tensor<1xT>` requires an index; `tensor<T>` does not.
  if (ty.getRank() == 1) {
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    index = ValueRange{zero};
  }
  return builder.create<tensor::ExtractOp>(loc, caller, index).getResult();
}

// Truncates `caller` to `calleeType` via `arith.trunci` when both are integer
// types and the caller is at least as wide. Returns `caller` unchanged when
// the types already match. Returns an error if the coercion is not valid.
static llvm::Expected<Value> adaptIntegerWidth(OpBuilder &builder, Location loc,
                                               Value caller, Type calleeType) {
  if (caller.getType() == calleeType) {
    return caller;
  }
  auto callerInt = mlir::dyn_cast<IntegerType>(caller.getType());
  auto calleeInt = mlir::dyn_cast<IntegerType>(calleeType);
  if (!callerInt || !calleeInt || callerInt.getWidth() < calleeInt.getWidth()) {
    return llvm::createStringError(llvm::formatv(
        "cannot coerce scalar type {0} to {1}", caller.getType(), calleeType));
  }
  return builder.create<arith::TruncIOp>(loc, calleeType, caller).getResult();
}

// Collapses an initial empty dimension: `tensor<1x...>` → `tensor<...>` via
// `tensor.collapse_shape`. Returns `caller` unchanged if the caller type does
// not have an initial empty dimension.
static Value pruneEmptyDimension(OpBuilder &builder, Location loc, Value caller,
                                 Type calleeType) {
  auto callerType = mlir::cast<RankedTensorType>(caller.getType());
  if (callerType.getRank() > 0 && callerType.getShape()[0] == 1) {
    assert(mlir::cast<RankedTensorType>(calleeType).getRank() ==
               callerType.getRank() - 1 &&
           "callee must have exactly one less dimension than caller");
    auto collapsedShape = llvm::ArrayRef<int64_t>(
        callerType.getShape().begin() + 1, callerType.getShape().end());
    auto collapsedTy =
        RankedTensorType::get(collapsedShape, callerType.getElementType());
    auto reassoc = SmallVector<ReassociationIndices>{{0, 1}};
    for (int64_t i = 2; i <= collapsedTy.getRank(); ++i) {
      reassoc.push_back({i});
    }
    return builder.create<tensor::CollapseShapeOp>(loc, collapsedTy, caller,
                                                   reassoc);
  } else {
    return caller;
  }
}

// Adapts the arguments of a `ttir.invoke_external` op to match the callee
// function parameter types:
//
//   - Tensors with concrete static shapes where the callee expects dynamic
//     shapes (or a different encoding). These are bridged with `tensor.cast`.
//   - Tensors with an initial empty dimension where the callee expects no such
//     dimension. The empty dimension is pruned with `tensor.collapse_shape`.
//   - 0-D or 1-D ranked tensors where the callee expects a bare scalar type.
//     These are unwrapped with `tensor.extract`.
//   - Wider integer types where the callee expects narrower integer types.
//     These are coerced with `arith.trunci`.
//
// Returns the adapted argument values, or failure if arity does not match.
static FailureOr<SmallVector<Value>>
adaptInputsToLinkAbi(OpBuilder &builder, ttir::InvokeExternalOp invokeOp,
                     func::FuncOp calleeFunc) {
  if (invokeOp.getArguments().size() != calleeFunc.getNumArguments()) {
    return invokeOp.emitOpError()
           << "argument count mismatch: caller provides "
           << invokeOp.getArguments().size() << " argument(s) but callee '"
           << calleeFunc.getSymName() << "' expects "
           << calleeFunc.getNumArguments();
  }

  Location loc = invokeOp.getLoc();
  SmallVector<Value> adaptedArgs;
  adaptedArgs.reserve(calleeFunc.getNumArguments());

  for (auto [callerArg, calleeParamType] :
       llvm::zip(invokeOp.getArguments(), calleeFunc.getArgumentTypes())) {
    Type callerArgType = callerArg.getType();
    if (!isa<RankedTensorType>(calleeParamType) &&
        isa<RankedTensorType>(callerArgType)) {
      // Case: tensor → scalar.
      auto scalar = adaptTensorToScalar(builder, loc, callerArg);
      if (!scalar) {
        return invokeOp.emitOpError() << llvm::toString(scalar.takeError());
      }
      // Case: wider int → narrower int.
      auto coerced = adaptIntegerWidth(builder, loc, *scalar, calleeParamType);
      if (!coerced) {
        return invokeOp.emitOpError() << llvm::toString(coerced.takeError());
      }
      adaptedArgs.push_back(*coerced);
    } else if (isa<RankedTensorType>(calleeParamType) &&
               isa<RankedTensorType>(callerArgType) &&
               callerArgType != calleeParamType) {
      // Case: tensor<1xT> → tensor<?>.
      callerArg = pruneEmptyDimension(builder, loc, callerArg, calleeParamType);
      adaptedArgs.push_back(
          builder.create<tensor::CastOp>(loc, calleeParamType, callerArg));
    } else if (callerArgType != calleeParamType) {
      // Case: unimplemented.
      return invokeOp.emitOpError()
             << "unimplement ABI detail: cannot yet adapt type "
             << callerArgType << " to type " << calleeParamType;
    } else {
      // Case: types match; pass through unchanged.
      adaptedArgs.push_back(callerArg);
    }
  }

  return adaptedArgs;
}

// Expands the first dimension to an initial empty dimension: `tensor<...>` →
// `tensor<1x...>` via `tensor.expand_shape`. This is the reverse of
// `collapseEmptyDimension`. Returns `callee` unchanged if the caller type does
// not require an initial empty dimension.
static Value expandEmptyDimension(OpBuilder &builder, Location loc, Value callee,
                               RankedTensorType callerType) {
  if (callerType.getRank() > 0 && callerType.getShape()[0] == 1) {
    auto calleeType = cast<RankedTensorType>(callee.getType());
    assert(callerType.getRank() == calleeType.getRank() + 1 &&
           "callee must have exactly one less dimension than caller");
    SmallVector<int64_t> expandedShape = {1};
    expandedShape.append(calleeType.getShape().begin(),
                         calleeType.getShape().end());
    auto expandedTy =
        RankedTensorType::get(expandedShape, callerType.getElementType());
    auto reassoc = SmallVector<ReassociationIndices>{{0, 1}};
    for (int64_t i = 2; i <= calleeType.getRank(); ++i) {
      reassoc.push_back({i});
    }
    return builder.create<tensor::ExpandShapeOp>(loc, expandedTy, callee,
                                                 reassoc);
  } else {
    return callee;
  }
}

// Adapts the results of a `func.call` back to the types declared on the
// originating `ttir.invoke_external` op. This is the reverse of
// `adaptInputsToLinkAbi` for return values:
//
//   - Tensors with dynamic static shapes where the caller expects static
//     shapes. These are bridged with `tensor.cast`.
//   - Tensors where the caller expects an initial empty dimension. The empty
//     dimension is added with `tensor.expand_shape`.
//
// The callee may return dynamic-shaped tensors while the surrounding IR
// expects the concrete shapes declared on the invoke op. `tensor.cast` is
// inserted where the types differ.
//
// Returns the adapted result values.
static SmallVector<Value> adaptOutputsToLinkAbi(OpBuilder &builder,
                                                ttir::InvokeExternalOp invokeOp,
                                                func::CallOp callOp) {
  Location loc = invokeOp.getLoc();
  SmallVector<Value> adaptedResults;
  adaptedResults.reserve(callOp.getNumResults());

  for (auto [callResult, invokeResultType] :
       llvm::zip(callOp.getResults(), invokeOp.getResultTypes())) {
    if (callResult.getType() != invokeResultType) {
      Value result = callResult;
      if (isa<RankedTensorType>(invokeResultType)) {
        result =
            expandEmptyDimension(builder, loc, result,
                              mlir::cast<RankedTensorType>(invokeResultType));
      }
      adaptedResults.push_back(
          builder.create<tensor::CastOp>(loc, invokeResultType, result));
    } else {
      adaptedResults.push_back(callResult);
    }
  }

  return adaptedResults;
}

struct TTIRLinkExternalFunctionsPass
    : public impl::TTIRLinkExternalFunctionsBase<
          TTIRLinkExternalFunctionsPass> {

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Cache: (target ModuleOp) → (resolvedPath → renameMap).
    // Each external module is parsed and merged at most once per target
    // module regardless of how many `ttir.invoke_external` ops reference it.
    llvm::DenseMap<ModuleOp, llvm::StringMap<llvm::StringMap<std::string>>>
        mergedByModule;

    // Collect ops before walking to avoid mutation-during-walk issues.
    SmallVector<ttir::InvokeExternalOp> invokeOps;
    moduleOp.walk([&](ttir::InvokeExternalOp op) { invokeOps.push_back(op); });

    for (auto invokeOp : invokeOps) {
      StringRef path = invokeOp.getPath();
      StringRef entry = invokeOp.getEntry();

      // Merge into the nearest ModuleOp parent of the invoke op so that the
      // resulting func.call can see the callee (MLIR symbol lookup does not
      // cross module boundaries).
      ModuleOp targetModule = invokeOp->getParentOfType<ModuleOp>();
      std::string resolvedPath = resolvePath(path, targetModule);

      auto &mergedPaths = mergedByModule[targetModule];
      if (!mergedPaths.contains(resolvedPath)) {
        mlir::ParserConfig config(targetModule.getContext());
        mlir::OwningOpRef<mlir::ModuleOp> externalModule =
            mlir::parseSourceFile<mlir::ModuleOp>(resolvedPath, config);
        if (!externalModule) {
          invokeOp.emitOpError()
              << "failed to parse external MLIR file '" << resolvedPath << "'";
          return signalPassFailure();
        }

        if (failed(mlir::verify(*externalModule))) {
          invokeOp.emitOpError() << "external MLIR file '" << resolvedPath
                                 << "' failed verification";
          return signalPassFailure();
        }

        auto renameMapOrErr = mergeExternalModule(targetModule, externalModule);
        if (failed(renameMapOrErr)) {
          invokeOp.emitOpError() << "failed to merge external module from '"
                                 << resolvedPath << "'";
          return signalPassFailure();
        }

        mergedPaths.try_emplace(resolvedPath, std::move(*renameMapOrErr));
      }

      // Resolve the final name of the entry symbol, accounting for any
      // rename.
      const auto &renameMap = mergedPaths[resolvedPath];
      std::string finalEntry = entry.str();
      if (auto it = renameMap.find(entry); it != renameMap.end()) {
        finalEntry = it->second;
      }

      // Look up the callee function now that it has been merged into the
      // target module (its name may have been rewritten).
      auto calleeFunc = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupSymbolIn(targetModule, finalEntry));
      if (!calleeFunc) {
        invokeOp.emitOpError()
            << "entry symbol '" << finalEntry << "' not found in module";
        return signalPassFailure();
      }

      // Replace the `ttir.invoke_external` op with a `func.call`. We adapt
      // the values to a "link ABI":
      // - we expect input and output tensors of the callee to have a dynamic
      //   shape (`tensor<?x?xf32>`); prior to the `func.call`, we
      //   `tensor.cast` from/to the concrete static shapes given to
      //   `ttir.invoke_external`.
      // - we expect scalars in the callee to be wrapped by 0D tensors
      //   (`tensor<f32>`); prior to the `func.call`, we `tensor.extract`
      //   them.
      OpBuilder builder(invokeOp);
      auto adaptedArgsOrErr =
          adaptInputsToLinkAbi(builder, invokeOp, calleeFunc);
      if (failed(adaptedArgsOrErr)) {
        return signalPassFailure();
      }
      auto callOp = builder.create<func::CallOp>(
          invokeOp.getLoc(), calleeFunc.getSymName(),
          calleeFunc.getResultTypes(), *adaptedArgsOrErr);
      auto bridgedResults = adaptOutputsToLinkAbi(builder, invokeOp, callOp);
      invokeOp.replaceAllUsesWith(bridgedResults);
      invokeOp.erase();
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::ttcore::TTCoreDialect>();
    registry.insert<mlir::tt::ttnn::TTNNDialect>();
    registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
    registry.insert<mlir::emitc::EmitCDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
  }
};

} // namespace

} // namespace mlir::tt::ttir
