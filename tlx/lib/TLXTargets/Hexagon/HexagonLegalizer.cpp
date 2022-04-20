//===-------- HexagonLegalizer.cpp -  Legalize tensor intrinsics -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Legalizer for Hexagon instructions.
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
#include "llvm/IR/IntrinsicsHexagon.h"

#include "llvm/Analysis/TensorProperties.h"
#include "llvm/IR/TensorType.h"
#include "Legalizer.h"


namespace llvm {

class HexagonLegalizationPass : public FunctionPass {
public:
    static char ID;

    HexagonLegalizationPass() : FunctionPass(ID) {
        initializeHexagonLegalizationPassPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F);

    void getAnalysisUsage(AnalysisUsage &AU) const {
        AU.addRequired<TensorInfoWrapperPass>();
    }
};

}


using namespace llvm;
using namespace PatternMatch;


class HexagonLegalizer : public Legalizer {
public:
  HexagonLegalizer(TensorInfo &TI) : Legalizer(TI) {}
  
  virtual bool legalize(Instruction *I) {
    IRBuilder<> Builder(I);
    auto &Ctx = I->getParent()->getContext();
    Value *Acc, *Vector1, *Vector2, *Vector, *LaneOffset, *LaneSize, *Val;

    if (match(I, m_Intrinsic<Intrinsic::vector_reduce_mac_acc>(
                  m_Value(Acc), m_Value(Vector1), m_Value(Vector2), m_Value(LaneSize)))) {
      auto *Acc = InstToInstMap[Acc];
      auto *Vect1 = InstToInstMap[Vector1];
      auto *Vect2 = InstToInstMap[Vector2];
      std::vector<Value *> Args;
      Args.push_back(Acc);
      Args.push_back(Vector1);
      Args.push_back(Vector2);
      Args.push_back(LaneSize);
      IntrinsicInst *Mac = nullptr;
      if (dyn_cast<FixedVectorType)(I->getType())->getNumElements() == 8) {
        Mac = Builder.CreateIntrinsic(Intrinsic::hexagon_V6_vrmpybus_acc_128B, 
                                            None, ArrayRef<Value *>(Args), nullptr, "hvx.reduce.mac");
      } else {
        assert(false && "Invalid element type");
      }
      ToBeRemoved.insert(I);
      InstToInstMap[I] = Mac;
      return true;
    }

    if (match(I, m_Intrinsic<Intrinsic::vector_interleave>(
                 m_Value(Vector1), m_Value(Vector2), m_Value(LaneSize), m_Value(LaneOffset)))) {
      auto *Acc = InstToInstMap[Vector1];
      auto *RTile = InstToInstMap[RTensor];
      auto *LTile = InstToInstMap[Vector2];
      std::vector<Value *> Args;
      Args.push_back(Vector1);
      Args.push_back(Vector2);
      IntrinsicInst *Shuffle = nullptr;
      auto *C = dyn_cast<ConstantInt>(LaneOffset);
      assert(C != nullptr);
      if (C->isZero()) {
        if (dyn_cast<FixedVectorType)(I->getType())->getNumElements() == 8) {
          Shuffle = Builder.CreateIntrinsic(Intrinsic::hexagon_V6_vshuffb_128B, 
                                                None, ArrayRef<Value *>(Args), nullptr, "hvx.low.shuffle");
        } else if (dyn_cast<FixedVectorType)(I->getType())->getNumElements() == 16) {
          Shuffle = Builder.CreateIntrinsic(Intrinsic::hexagon_V6_vshuffh_128B, 
                                                None, ArrayRef<Value *>(Args), nullptr, "hvx.low.shuffle");
        } else (dyn_cast<FixedVectorType)(I->getType())->getNumElements() == 32) {
          Shuffle = Builder.CreateIntrinsic(Intrinsic::hexagon_V6_shuffeqw_128B, 
                                                None, ArrayRef<Value *>(Args), nullptr, "hvx.low.shuffle");
        } else {
          assert(false && "Invalid element type");
        }
      } else {
        if (dyn_cast<FixedVectorType)(I->getType())->getNumElements() == 8) {
          Shuffle = Builder.CreateIntrinsic(Intrinsic::hexagon_V6_vshuffeb_128B, 
                                                None, ArrayRef<Value *>(Args), nullptr, "hvx.high.shuffle");
        } else if (dyn_cast<FixedVectorType)(I->getType())->getNumElements() == 16) {
          Shuffle = Builder.CreateIntrinsic(Intrinsic::hexagon_V6_shuffeqh_128B, 
                                                None, ArrayRef<Value *>(Args), nullptr, "hvx.high.shuffle");
        } else (dyn_cast<FixedVectorType)(I->getType())->getNumElements() == 32) {
          Shuffle = Builder.CreateIntrinsic(Intrinsic::hexagon_V6_shuffeqw_128B, 
                                                None, ArrayRef<Value *>(Args), nullptr, "hvx.high.shuffle");
        } else {
          assert(false && "Invalid element type");
        }
      }
      ToBeRemoved.insert(I);
      InstToInstMap[I] = Shuffle;
      return true;
    }

    if (match(I, m_Intrinsic<Intrinsic::vector_splat>(
                  m_Value(Vector), m_Value(Val)))) {
      if (auto *C = dyn_cast<ConstantInt>(Val)) {
        std::vector<Value *> Args = {Vector, Val};
        IntrinsicInst *VectorSplat = nullptr;
        if (dyn_cast<FixedVectorType)(I->getType())->getNumElements() == 8) {
          VectorSplat = Builder.CreateIntrinsic(Intrinsic::hexagon_V6_lvsplatb_128B, 
                                                    None, ArrayRef<Value *>(Args));
        } else if (dyn_cast<FixedVectorType)(I->getType())->getNumElements() == 16) {
          VectorSplat = Builder.CreateIntrinsic(Intrinsic::hexagon_V6_lvsplath_128B, 
                                              None, ArrayRef<Value *>(Args));
        } else (dyn_cast<FixedVectorType)(I->getType())->getNumElements() == 32) {
          VectorSplat = Builder.CreateIntrinsic(Intrinsic::hexagon_V6_lvsplatw_128B, 
                                      None, ArrayRef<Value *>(Args));
        } else {
          assert(false && "Invalid element type");
        }
        ToBeRemoved.insert(I);
        InstToInstMap[I] = VectorSplat;
        return true;
      }
      return false;
    }

    return false;
  }

  bool isVectorInst(Instruction *I) const {
    if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      switch(II->getIntrinsicID()) {
        case Intrinsic::vector_interleave:
        case Intrinsic::vector_reduce_mac_acc:
        case Intrinsic::vector_splat:
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


bool HexagonLegalizationPass::runOnFunction(Function &F) {
  // Get the tensor values to tensor properties mappings
  auto &TI = getAnalysis<TensorInfoWrapperPass>().getTensorInfo(&F);

  // Initialize the legalizer
  Legalizer *L = new HexagonLegalizer(TI);
  bool Changed = L->legalize(F);

  return Changed;
}


char HexagonLegalizationPass::ID = 0;

INITIALIZE_PASS_BEGIN(
    HexagonLegalizationPass, "legalize-hexagon",
    "Pass to legalize vector intrinsics to target Hexagon", false, false)
INITIALIZE_PASS_DEPENDENCY(TensorInfoWrapperPass)
INITIALIZE_PASS_END(HexagonLegalizationPass, "legalize-Hexagon",
    "Pass to legalize vector intrinsics to target Hexagon", false, false)

FunctionPass *llvm::createHexagonLegalizationPass() {
  return new HexagonLegalizationPass();
}


