//===- FIRRTLDialect.cpp - Implement the FIRRTL dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/FieldRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void FIRRTLDialect::initialize() {
  // Register types and attributes.
  registerTypes();
  registerAttributes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/FIRRTL/FIRRTL.cpp.inc"
      >();
}

/// Registered hook to materialize a single constant operation from a given
/// attribute value with the desired resultant type. This method should use
/// the provided builder to create the operation without changing the
/// insertion position. The generated operation is expected to be constant
/// like, i.e. single result, zero operands, non side-effecting, etc. On
/// success, this hook should return the value generated to represent the
/// constant value. Otherwise, it should return null on failure.
Operation *FIRRTLDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {

  // Boolean constants. Boolean attributes are always a special constant type
  // like ClockType and ResetType.  Since BoolAttrs are also IntegerAttrs, its
  // important that this goes first.
  if (auto attrValue = value.dyn_cast<BoolAttr>()) {
    assert((type.isa<ClockType>() || type.isa<AsyncResetType>() ||
            type.isa<ResetType>()) &&
           "BoolAttrs can only be materialized for special constant types.");
    return builder.create<SpecialConstantOp>(loc, type, attrValue);
  }

  // Integer constants.
  if (auto attrValue = value.dyn_cast<IntegerAttr>()) {
    // Integer attributes (ui1) might still be special constant types.
    if (attrValue.getValue().getBitWidth() == 1 &&
        (type.isa<ClockType>() || type.isa<AsyncResetType>() ||
         type.isa<ResetType>()))
      return builder.create<SpecialConstantOp>(
          loc, type, builder.getBoolAttr(attrValue.getValue().isAllOnes()));

    assert((!type.cast<IntType>().hasWidth() ||
            (unsigned)type.cast<IntType>().getWidthOrSentinel() ==
                attrValue.getValue().getBitWidth()) &&
           "type/value width mismatch materializing constant");
    return builder.create<ConstantOp>(loc, type, attrValue);
  }

  // Aggregate constants.
  if (auto arrayAttr = value.dyn_cast<ArrayAttr>()) {
    if (type.isa<BundleType, FVectorType>())
      return builder.create<AggregateConstantOp>(loc, type, arrayAttr);
  }

  return nullptr;
}

namespace {
// During canonicalization, constant folding can result in operand types
// changing from non-'const' to 'const'. This can cause result types to likewise
// change their inference from non-'const' to 'const'. This pattern handles
// those result type changes.
struct ReinferResultTypes
    : public mlir::OpInterfaceRewritePattern<mlir::InferTypeOpInterface> {
  ReinferResultTypes(MLIRContext *context)
      : OpInterfaceRewritePattern<mlir::InferTypeOpInterface>(context) {}

  LogicalResult matchAndRewrite(mlir::InferTypeOpInterface op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Type> inferredResultTypes;
    if (failed(op.inferReturnTypes(op->getContext(), op->getLoc(),
                                   op->getOperands(), op->getAttrDictionary(),
                                   op->getRegions(), inferredResultTypes))) {
      return failure();
    }

    bool anyChanged = false;
    for (size_t i = 0, e = inferredResultTypes.size(); i != e; ++i) {
      auto result = op->getResult(i);
      auto inferredType = inferredResultTypes[i];
      if (result.getType() != inferredType) {
        anyChanged = true;
        break;
      }
    }

    if (!anyChanged)
      return failure();

    rewriter.setInsertionPointAfter(op);

    auto *updatedOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                        op->getOperands(), inferredResultTypes, op->getAttrs());

    rewriter.replaceOp(op, updatedOp->getResults());
    return success();
  }
};

// This pattern is a workaround for the fact that a 'const' type cannot be
// folded to replace a non-'const' result type
struct FoldConst
    : public mlir::OpInterfaceRewritePattern<mlir::InferTypeOpInterface> {
  FoldConst(MLIRContext *context)
      : OpInterfaceRewritePattern<mlir::InferTypeOpInterface>(context) {}

  LogicalResult matchAndRewrite(mlir::InferTypeOpInterface op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Attribute> operandConstants;

    // Check to see if any operands to the operation is constant and whether
    // the operation knows how to constant fold itself
    operandConstants.assign(op->getNumOperands(), Attribute());
    for (size_t i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&operandConstants[i]));

    // Attempt to constant fold the operation
    SmallVector<OpFoldResult> foldResults;
    if (failed(op->fold(operandConstants, foldResults)))
      return failure();

    for (size_t i = 0, e = foldResults.size(); i != e; ++i) {
      auto originalResult = op->getResults()[i];
      Value replacementValue;
      if (auto value = foldResults[i].dyn_cast<Value>()) {
        replacementValue = value;
      } else if (auto constant = foldResults[i].dyn_cast<Attribute>()) {
        auto type = op->getResultTypes()[i];
        if (auto baseType = type.dyn_cast<FIRRTLBaseType>())
          type = baseType.getConstType(true);
        auto *constantOp = op->getDialect()->materializeConstant(
            rewriter, constant, type, op.getLoc());
        replacementValue = constantOp->getResults()[0];
      } else {
        return failure();
      }

      originalResult.replaceAllUsesWith(replacementValue);
      rewriter.eraseOp(op);
    }

    return success();
  }
};
} // namespace

void FIRRTLDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.add<ReinferResultTypes, FoldConst>(getContext());
}

// Provide implementations for the enums we use.
#include "circt/Dialect/FIRRTL/FIRRTLEnums.cpp.inc"

#include "circt/Dialect/FIRRTL/FIRRTLDialect.cpp.inc"
