//===-------- x86Legalizer.cpp -  Legalize tensor intrinsics -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Legalizer for x86 instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/IntrinsicsX86.h"

#include "llvm/Analysis/TensorProperties.h"
#include "llvm/IR/TensorType.h"


namespace llvm {

class X86LegalizationPass : public FunctionPass {
public:
    static char ID;

    X86LegalizationPass() : FunctionPass(ID) {
        initializeX86LegalizationPassPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F);

    void getAnalysisUsage(AnalysisUsage &AU) const {
        AU.addRequired<TensorInfoWrapperPass>();
    }
};

}


using namespace llvm;
using namespace PatternMatch;


class X86Legalizer : public Legalizer {
public:
  X86Legalizer(TensorInfo &TI) : Legalizer(TI) {}
  
  virtual bool legalize(Instruction *I) {
    IRBuilder<> Builder(I);
    auto &Ctx = I->getParent()->getContext();
    Value *Ptr, *Shape, *Layout, *Padding, *Strides, *Vector, *Token, *RTensor, *LTensor, *Val;

    auto GetStride = [&](Value *V) {
      assert(dyn_cast<ConstantDataVector>(V));
      auto *CV = dyn_cast<ConstantDataVector>(V);
      auto *C = dyn_cast<ConstantInt>(CV->getAggregateElement(1));
      return Builder.CreateZExt(C, Type::getInt64Ty(Ctx));
    };

    if(match(I, m_Intrinsic<Intrinsic::tensor_load>(
            m_Value(Ptr), m_Value(Shape), m_Value(Layout), m_Value(Padding), m_Value(Strides)))) {
      // Insert an AMX load
      auto &ShapeVect = TI.getShapeVectorFor(I);
      SmallVector<Value *, 2> ShapeValVect = {ConstantInt::get(Type::getInt16Ty(Ctx), ShapeVect[0]), 
                                              ConstantInt::get(Type::getInt16Ty(Ctx), ShapeVect[1])};
      Value *Stride = GetStride(Strides);
      std::vector<Value *> Args;
      Args.insert(Args.end(), ShapeValVect.begin(), ShapeValVect.end());
      Args.push_back(Ptr);
      Args.push_back(Stride);
      auto *LI = Builder.CreateIntrinsic(Intrinsic::x86_tileloadd64_internal, 
                                              None, ArrayRef<Value *>(Args), nullptr, "amx.load");
      ToBeRemoved.insert(I);
      InstToInstMap[I] = LI;
      return true;
    }

    if(match(I, m_Intrinsic<Intrinsic::tensor_store>(
            m_Value(Ptr), m_Value(Strides), m_Value(Token)))) {
      // Insert an AMX store
      auto &ShapeVect = TI.getShapeVectorFor(Token);
      SmallVector<Value *, 2> ShapeValVect = {ConstantInt::get(Type::getInt16Ty(Ctx), ShapeVect[0]), 
                                              ConstantInt::get(Type::getInt16Ty(Ctx), ShapeVect[1])};
      Value *Stride = GetStride(Strides);
      auto *AMXTypeVal = InstToInstMap[Token];
      if (!AMXTypeVal) {
        ToBeLegalized.push_back(I);
        return false;
      }
      std::vector<Value *> Args;
      Args.insert(Args.end(), ShapeValVect.begin(), ShapeValVect.end());
      Args.push_back(Ptr);
      Args.push_back(Stride);
      Args.push_back(AMXTypeVal);
      auto *SI = Builder.CreateIntrinsic(Intrinsic::x86_tilestored64_internal, 
                                                None, ArrayRef<Value *>(Args));
      ToBeRemoved.insert(I);
      InstToInstMap[I] = SI;
      return true;
    }

    if (match(I, m_Intrinsic<Intrinsic::tensor_smma>(
                  m_Value(Token), m_Value(RTensor), m_Value(LTensor)))) {
      // Insert an AMX mma
      auto RTileShapeVect = TI.getShapeVectorFor(RTensor);
      auto TokenShapeVect = TI.getShapeVectorFor(Token);
      auto *Acc = InstToInstMap[Token];
      auto *RTile = InstToInstMap[RTensor];
      auto *LTile = InstToInstMap[LTensor];
      if (!RTile || !LTile) {
        ToBeLegalized.push_back(I);
        return false;
      }
      std::vector<Value *> Args = {ConstantInt::get(Type::getInt16Ty(Ctx), TokenShapeVect[0]),
                                  ConstantInt::get(Type::getInt16Ty(Ctx), TokenShapeVect[1]),
                                  ConstantInt::get(Type::getInt16Ty(Ctx), RTileShapeVect[1])};
      Args.push_back(Acc);
      Args.push_back(RTile);
      Args.push_back(LTile);
      auto *MMA = Builder.CreateIntrinsic(Intrinsic::x86_tdpbssd_internal, 
                                            None, ArrayRef<Value *>(Args), nullptr, "amx.mma");
      ToBeRemoved.insert(I);
      InstToInstMap[I] = MMA;
      return true;
    }

    if (match(I, m_Intrinsic<Intrinsic::tensor_umma>(
                  m_Value(Token), m_Value(RTensor), m_Value(LTensor)))) {
      // Insert an AMX mma
      auto RTileShapeVect = TI.getShapeVectorFor(RTensor);
      auto TokenShapeVect = TI.getShapeVectorFor(Token);
      auto *Acc = InstToInstMap[Token];
      auto *RTile = InstToInstMap[RTensor];
      auto *LTile = InstToInstMap[LTensor];
      if (!RTile || !LTile) {
        ToBeLegalized.push_back(I);
        return false;
      }
      std::vector<Value *> Args = {ConstantInt::get(Type::getInt16Ty(Ctx), TokenShapeVect[0]),
                                  ConstantInt::get(Type::getInt16Ty(Ctx), TokenShapeVect[1]),
                                  ConstantInt::get(Type::getInt16Ty(Ctx), RTileShapeVect[1])};
      Args.push_back(Acc);
      Args.push_back(RTile);
      Args.push_back(LTile);
      auto *MMA = Builder.CreateIntrinsic(Intrinsic::x86_tdpbuud_internal, 
                                            None, ArrayRef<Value *>(Args), nullptr, "amx.mma");
      ToBeRemoved.insert(I);
      InstToInstMap[I] = MMA;
      return true;
    }

    if (match(I, m_Intrinsic<Intrinsic::tensor_summa>(
                  m_Value(Token), m_Value(RTensor), m_Value(LTensor)))) {
      // Insert an AMX mma
      auto RTileShapeVect = TI.getShapeVectorFor(RTensor);
      auto TokenShapeVect = TI.getShapeVectorFor(Token);
      auto *Acc = InstToInstMap[Token];
      auto *RTile = InstToInstMap[RTensor];
      auto *LTile = InstToInstMap[LTensor];
      if (!RTile || !LTile) {
        ToBeLegalized.push_back(I);
        return false;
      }
      std::vector<Value *> Args = {ConstantInt::get(Type::getInt16Ty(Ctx), TokenShapeVect[0]),
                                  ConstantInt::get(Type::getInt16Ty(Ctx), TokenShapeVect[1]),
                                  ConstantInt::get(Type::getInt16Ty(Ctx), RTileShapeVect[1])};
      Args.push_back(Acc);
      Args.push_back(RTile);
      Args.push_back(LTile);
      auto *MMA = Builder.CreateIntrinsic(Intrinsic::x86_tdpbsud_internal, 
                                            None, ArrayRef<Value *>(Args), nullptr, "amx.mma");
      ToBeRemoved.insert(I);
      InstToInstMap[I] = MMA;
      return true;
    }
  
    if (match(I, m_Intrinsic<Intrinsic::tensor_usmma>(
                  m_Value(Token), m_Value(RTensor), m_Value(LTensor)))) {
      // Insert an AMX mma
      auto RTileShapeVect = TI.getShapeVectorFor(RTensor);
      auto TokenShapeVect = TI.getShapeVectorFor(Token);
      auto *Acc = InstToInstMap[Token];
      auto *RTile = InstToInstMap[RTensor];
      auto *LTile = InstToInstMap[LTensor];
      if (!RTile || !LTile) {
        ToBeLegalized.push_back(I);
        return false;
      }
      std::vector<Value *> Args = {ConstantInt::get(Type::getInt16Ty(Ctx), TokenShapeVect[0]),
                                  ConstantInt::get(Type::getInt16Ty(Ctx), TokenShapeVect[1]),
                                  ConstantInt::get(Type::getInt16Ty(Ctx), RTileShapeVect[1])};
      Args.push_back(Acc);
      Args.push_back(RTile);
      Args.push_back(LTile);
      auto *MMA = Builder.CreateIntrinsic(Intrinsic::x86_tdpbusd_internal, 
                                            None, ArrayRef<Value *>(Args), nullptr, "amx.mma");
      ToBeRemoved.insert(I);
      InstToInstMap[I] = MMA;
      return true;
    }

    if (match(I, m_Intrinsic<Intrinsic::tensor_broadcast>(
                  m_Value(Token), m_Value(Val)))) {
      if (auto *C = dyn_cast<ConstantInt>(Val)) {
        if (C->isZero()) {
          // Insert an AMX zero tile intrinsic
          auto ShapeVect = TI.getShapeVectorFor(Token);
          std::vector<Value *> Args = {ConstantInt::get(Type::getInt16Ty(Ctx), ShapeVect[0]), 
                                      ConstantInt::get(Type::getInt16Ty(Ctx), ShapeVect[1])};
          auto *ZeroTile = Builder.CreateIntrinsic(Intrinsic::x86_tilezero_internal, 
                                                None, ArrayRef<Value *>(Args));
          ToBeRemoved.insert(I);
          InstToInstMap[I] = ZeroTile;
          return true;
        }
      }
      return false;
    }

    if (match(I, m_Intrinsic<Intrinsic::tensor_typeinfo>(
                  m_Value(Vector), m_Value(Shape), m_Value(Layout), m_Value(Padding)))) {
      auto *Inst = InstToInstMap[Vector];
      if (!Inst) {
        ToBeLegalized.push_back(I);
        return false;
      }
      ToBeRemoved.insert(I);
      InstToInstMap[I] = Inst;
      return false;
    }

    if (auto *PHI = dyn_cast<PHINode>(I)) {
      if (PHI->getType()->isVectorTy()) {
        if (!InstToInstMap[I]) {
          auto *NewPHI = Builder.CreatePHI(Type::getX86_AMXTy(Ctx), 
                        PHI->getNumIncomingValues(), "amx.phi");
          ToBeRemoved.insert(I);
          InstToInstMap[I] = NewPHI;
        }

        SmallVector<Value *, 2> PHIOperands;
        for (unsigned i = 0; i < PHI->getNumIncomingValues(); i++) {
          Instruction *Op = dyn_cast<Instruction>(PHI->getIncomingValue(i));
          if (!Op) {
            PHIOperands.push_back(Op);
            continue;
          }
          if (!InstToInstMap[Op]) {
            ToBeLegalized.push_back(I);
            return false;
          }
          PHIOperands.push_back(Op);
        }

        auto *NewPHI = dyn_cast<PHINode>(InstToInstMap[I]);
        for (unsigned i = 0; i < PHI->getNumIncomingValues(); i++) {
            auto *Op = PHIOperands[i];
            NewPHI->addIncoming(InstToInstMap[Op], PHI->getIncomingBlock(i));
        }

        return true;
      }
      return false;
    }

    return false;
  }

  bool isTensorInst(Instruction *I) const {
    if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      switch(II->getIntrinsicID()) {
        case Intrinsic::tensor_mma:
        case Intrinsic::tensor_load:
        case Intrinsic::tensor_store:
        case Intrinsic::tensor_typeinfo:
        case Intrinsic::tensor_broadcast:
          return true;
        default:
          return false;
      }
    }
    if (auto *PHI = dyn_cast<PHINode>(I)) {
      if (PHI->getType()->isVectorTy())
        return true;
    }
    return false;
  }
};


bool X86LegalizationPass::runOnFunction(Function &F) {
  // Get the tensor values to tensor properties mappings
  auto &TI = getAnalysis<TensorInfoWrapperPass>().getTensorInfo(&F);

  // Initialize the legalizer
  Legalizer *L = new X86Legalizer(TI);
  bool Changed = L->legalize(F);

  return Changed;
}


char X86LegalizationPass::ID = 0;

INITIALIZE_PASS_BEGIN(
    X86LegalizationPass, "legalize-x86",
    "Pass to legalize tensor intrinsics", false, false)
INITIALIZE_PASS_DEPENDENCY(TensorInfoWrapperPass)
INITIALIZE_PASS_END(
    X86LegalizationPass, "legalize-x86",
    "Pass to legalize tensor intrinsics", false, false)

FunctionPass *llvm::createX86LegalizationPass() {
  return new X86LegalizationPass();
}

