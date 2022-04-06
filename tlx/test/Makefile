LLVM_ROOT=../..
BIN=$(LLVM_ROOT)/build/bin
BUILD=$(LLVM_ROOT)/build


TESTS=$(SOURCES:%.c)

all: licm dse cse dce adce

licm:
	$(BIN)/clang -S -emit-llvm licm_test.c -O0 -Xclang -disable-O0-optnone
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg  licm_test.ll -S -o licm.mem2reg.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg -tensor -licm  licm_test.ll -S -o licm_test.licm.ll
	$(BIN)/clang -S -emit-llvm licm_test2.c -O0 -Xclang -disable-O0-optnone
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg -tensor -licm  licm_test2.ll -S -o licm_test2.licm.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -load $(BUILD)/lib/LLVMTensorProperties.so -mem2reg -tensor -tensor-analysis licm_test2.ll -S -o licm_test2.tensor.ll

dse:
	$(BIN)/clang -S -emit-llvm dse_test.c -O0 -Xclang -disable-O0-optnone
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg -tensor  -dse  dse_test.ll -S -o dse_test.dse.ll
	$(BIN)/clang -S -emit-llvm dse_test2.c -O0 -Xclang -disable-O0-optnone
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg -tensor  -dse  dse_test2.ll -S -o dse_test2.dse.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -load $(BUILD)/lib/LLVMTensorProperties.so -mem2reg -tensor -tensor-analysis dse_test2.ll -S -o dse_test2.tensor.ll

cse:
	$(BIN)/clang -S -emit-llvm cse_test.c -O0 -Xclang -disable-O0-optnone
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg -tensor -early-cse  cse_test.ll -S -o cse_test.cse.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -load $(BUILD)/lib/LLVMTensorProperties.so -mem2reg -tensor -tensor-analysis cse_test.ll -S -o cse_test.tensor.ll
	$(BIN)/clang -S -emit-llvm cse_test2.c -O0 -Xclang -disable-O0-optnone
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg -tensor -early-cse  cse_test2.ll -S -o cse_test2.cse.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -load $(BUILD)/lib/LLVMTensorProperties.so -mem2reg -tensor -tensor-analysis cse_test2.ll -S -o cse_test2.tensor.ll

dce:
	$(BIN)/clang -S -emit-llvm dce_test.c -O0 -Xclang -disable-O0-optnone
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg  dce_test.ll -S -o dce.mem2reg.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg -tensor -dce  dce_test.ll -S -o dce_test.dce.ll

adce:
	$(BIN)/clang -S -emit-llvm adce_test.c -O0 -Xclang -disable-O0-optnone
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg -tensor -adce  adce_test.ll -S -o adce_test.adce.ll


lower_matmul1:
	$(BIN)/clang -S -emit-llvm matmul1.c -O0 -Xclang -disable-O0-optnone
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -enable-new-pm=0 -mem2reg -tensor -tensor-analysis -lower-tensor matmul1.ll -S -o matmul1.lowered.ll

lower_matmul2:
	$(BIN)/clang -S -emit-llvm matmul2.c -O0 -Xclang -disable-O0-optnone
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -enable-new-pm=0 -mem2reg -tensor -tensor-analysis -lower-tensor matmul2.ll -S -o matmul2.lowered.ll

lower_matmul3:
	$(BIN)/clang -S -emit-llvm matmul3.c -O0 -Xclang -disable-O0-optnone
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -enable-new-pm=0 -mem2reg -tensor -tensor-analysis -lower-tensor matmul3.ll -S -o matmul3.lowered.ll


lower_reduction1:
	$(BIN)/clang -S -emit-llvm reduction1.c -O0 -Xclang -disable-O0-optnone
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -enable-new-pm=0 -mem2reg -tensor -tensor-analysis -lower-tensor reduction1.ll -S -o reduction1.lowered.ll

lower_reduction2:
	$(BIN)/clang -S -emit-llvm reduction2.c -O0 -Xclang -disable-O0-optnone
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -enable-new-pm=0 -mem2reg -tensor -tensor-analysis -lower-tensor reduction2.ll -S -o reduction2.lowered.ll

lower_reduction2:
	$(BIN)/clang -S -emit-llvm reduction3.c -O0 -Xclang -disable-O0-optnone
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -enable-new-pm=0 -mem2reg -tensor -tensor-analysis -lower-tensor reduction3.ll -S -o reduction3.lowered.ll


clean:
	rm -rf *.ll
