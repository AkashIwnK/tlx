//===- TilingSupport.h - Tiling Support -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interfaces to support tiling of loops when lowering tensor operations.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_TILING_SUPPORT_H
#define LLVM_TILING_SUPPORT_H

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

struct TiledLoopNestInfo {
  /// Loop Bounds from outermost loop to innermost loop
  SmallVector<unsigned, 4> LoopBounds;

  /// Loop steps from outermost loop to innermost loop
  SmallVector<unsigned, 4> LoopSteps;

  /// Loop start indices
  SmallVector<unsigned, 4> LoopStartIndices;

  /// Loop latches from outermost loop to innermost loop
  SmallVector<BasicBlock *, 4> LoopLatches;

  /// Loop headers from outermost loop to innermost loop
  SmallVector<BasicBlock *, 4> LoopHeaders;

  /// Preheaders for loops in loop nest
  SmallVector<BasicBlock *, 4> LoopPreheaders;

  /// The loop nest indices vector
  SmallVector<Value *, 4> LoopIndices;

  /// The innermost block of the loop nest
  BasicBlock *InnerLoopBody = nullptr;

  TiledLoopNestInfo(
      SmallVector<unsigned, 4> LoopBounds, SmallVector<unsigned, 4> LoopSteps,
      SmallVector<unsigned, 4> LoopStartIndices)
      : LoopBounds(LoopBounds), LoopSteps(LoopSteps),
        LoopStartIndices(LoopStartIndices) {}

  TiledLoopNestInfo() = default;
};

BasicBlock *CreateLoop(
    BasicBlock *Preheader, BasicBlock *Exit, Value *Bound, Value *Step,
    Value *StartIndex, bool MustHaveBody, StringRef Name, DomTreeUpdater &DTU,
    Loop *L, LoopInfo &LI) {
  LLVMContext &Ctx = Preheader->getContext();
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *Header = BasicBlock::Create(
      Preheader->getContext(), Name + ".header", Preheader->getParent(), Exit);
  BasicBlock *Body = nullptr;
  if (MustHaveBody) {
    Body = BasicBlock::Create(
        Header->getContext(), Name + ".body", Header->getParent(), Exit);
  }
  auto *Latch = BasicBlock::Create(
      Header->getContext(), Name + ".latch", Header->getParent(), Exit);
  if (MustHaveBody) {
    BranchInst::Create(Body, Header);
    BranchInst::Create(Latch, Body);
  } else {
    BranchInst::Create(Latch, Header);
  }

  auto *IV = PHINode::Create(I32Ty, 2, Name + ".iv", Header->getTerminator());
  IV->addIncoming(StartIndex, Preheader);

  Value *Inc =
      BinaryOperator::Create(Instruction::Add, IV, Step, Name + ".step", Latch);
  Value *Cond = CmpInst::Create(
      Instruction::ICmp, ICmpInst::ICMP_NE, Inc, Bound, Name + ".step", Latch);
  BranchInst::Create(Header, Exit, Cond, Latch);
  IV->addIncoming(Inc, Latch);

  auto *PreheaderBr = cast<BranchInst>(Preheader->getTerminator());
  BasicBlock *Tmp = PreheaderBr->getSuccessor(0);
  PreheaderBr->setSuccessor(0, Header);

  if(MustHaveBody) {
    DTU.applyUpdatesPermissive({
        {DominatorTree::Delete, Preheader, Tmp},
        {DominatorTree::Insert, Header, Body},
        {DominatorTree::Insert, Body, Latch},
        {DominatorTree::Insert, Latch, Header},
        {DominatorTree::Insert, Latch, Exit},
        {DominatorTree::Insert, Preheader, Header},
    });
    L->addBasicBlockToLoop(Header, LI);
    L->addBasicBlockToLoop(Body, LI);
    L->addBasicBlockToLoop(Latch, LI);

    return Body;
  } else {
    DTU.applyUpdatesPermissive({
        {DominatorTree::Delete, Preheader, Tmp},
        {DominatorTree::Insert, Header, Latch},
        {DominatorTree::Insert, Latch, Header},
        {DominatorTree::Insert, Latch, Exit},
        {DominatorTree::Insert, Preheader, Header},
    });
    L->addBasicBlockToLoop(Header, LI);
    L->addBasicBlockToLoop(Latch, LI);

    return Header;
  }

  return nullptr;
}

/// Creates the following loop nest skeleton:
///  for m = 0; m < M; m += TileSize_M
///    for n = 0; n < N; n += TileSize_N
///      for k = 0; k < K ; k += TileSize_K
///         ...
void CreateTiledLoops(
    BasicBlock *Start, BasicBlock *End, DomTreeUpdater &DTU, LoopInfo &LI,
    TiledLoopNestInfo &TI, bool MustHaveBody = false) {
  SmallVector<Loop *, 4> Loops;
  for (unsigned I = 0; I < TI.LoopBounds.size(); I++) {
    Loops.push_back(LI.AllocateLoop());
  }
  for (unsigned I = 0; I < Loops.size() - 1; I++) {
    Loops[I]->addChildLoop(Loops[I + 1]);
  }
  if (Loop *ParentL = LI.getLoopFor(Start)) {
    ParentL->addChildLoop(Loops[0]);
  } else {
    LI.addTopLevelLoop(Loops[0]);
  }

  auto &Ctx = Start->getContext();
  auto *Int32Ty = Type::getInt32Ty(Ctx);

  unsigned NumLoops = Loops.size();
  BasicBlock *Body = Start;
  BasicBlock *Latch = End;
  for (unsigned I = 0; I < NumLoops; I++) {
    TI.LoopPreheaders.push_back(Body);
    MustHaveBody = (I == NumLoops - 1) ? true : MustHaveBody;
    Body = CreateLoop(
        Body, Latch, ConstantInt::get(Int32Ty, TI.LoopBounds[I]),
        ConstantInt::get(Int32Ty, TI.LoopSteps[I]),
        ConstantInt::get(Int32Ty, TI.LoopStartIndices[I]), MustHaveBody, "loop",
        DTU, Loops[I], LI);
    Latch = Body->getSingleSuccessor();
    BasicBlock *Header;
    if (MustHaveBody) {
      Header = Body->getSinglePredecessor();
    } else {
      Header = Body;
    }
    TI.LoopLatches.push_back(Latch);
    TI.LoopHeaders.push_back(Header);
    TI.LoopIndices.push_back(&*(Header)->begin());
  }
  TI.InnerLoopBody = Body;
}

}  // namespace llvm


#endif // LLVM_TILING_SUPPORT_H


