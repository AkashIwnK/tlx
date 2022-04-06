//===- TensorUtils.h - Tensor Utilities -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions for tensor intrinsics lowering passes.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_TLX_TENSOR_UTILS_H
#define LLVM_TLX_TENSOR_UTILS_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"


namespace llvm {

Constant *GetConstantValue(LLVMContext &Ctx, Type *Ty, int64_t Val) {
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID:
    return ConstantInt::get(Type::getInt32Ty(Ctx), (int)Val);
  case Type::FloatTyID:
    return ConstantFP::get(Type::getFloatTy(Ctx), (float)Val);
  case Type::DoubleTyID:
    return ConstantFP::get(Type::getDoubleTy(Ctx), (double)Val);
  case Type::HalfTyID:
  case Type::BFloatTyID:
  default:
    assert(false && "Invalid element type.");
  }
  return nullptr;
}

Value *ConvertToFloat(Value *V, Instruction *InsertBefore) {
  switch (V->getType()->getTypeID()) {
  case Type::IntegerTyID:
    return new SIToFPInst(
        V, Type::getFloatTy(InsertBefore->getParent()->getContext()), "",
        InsertBefore);
  case Type::FloatTyID:
  case Type::DoubleTyID:
    return V;
  case Type::HalfTyID:
  case Type::BFloatTyID:
  default:
    assert(false && "Invalid element type.");
  }
  return nullptr;
}

int64_t GetMaxFor(Type *Ty) {
  switch(Ty->getTypeID()) {
    case Type::IntegerTyID: 
      switch(Ty->getIntegerBitWidth()) {
        case 1:
          return 1;
        case 8:
          return (int64_t)std::numeric_limits<int8_t>::max();
        case 16:
          return (int64_t)std::numeric_limits<int16_t>::max();
        case 32:
          return (int64_t)std::numeric_limits<int32_t>::max();
        case 64:
          return (int64_t)std::numeric_limits<int64_t>::max();
        default:
          assert(false && "Get max for valid integer type.");
      }
    case Type::FloatTyID:
      return (int64_t)std::numeric_limits<float>::max();
    case Type::DoubleTyID:
      return (int64_t)std::numeric_limits<double>::max();
    default:
      assert(false && "Get max for valid type.");
  }
}

int64_t GetMinFor(Type *Ty) {
  switch(Ty->getTypeID()) {
    case Type::IntegerTyID: 
      switch(Ty->getIntegerBitWidth()) {
        case 1:
          return 0;
        case 8:
          return (int64_t)std::numeric_limits<int8_t>::min();
        case 16:
          return (int64_t)std::numeric_limits<int16_t>::min();
        case 32:
          return (int64_t)std::numeric_limits<int32_t>::min();
        case 64:
          return (int64_t)std::numeric_limits<int64_t>::min();
        default:
          assert(false && "Get min for valid integer type.");
      }
    case Type::FloatTyID:
      return (int64_t)std::numeric_limits<float>::min();
    case Type::DoubleTyID:
      return (int64_t)std::numeric_limits<double>::min();
    default:
      assert(false && "Get min for valid type.");
  }
}

} // namespace llvm

#endif // LLVM_TLX_TENSOR_UTILS_H


