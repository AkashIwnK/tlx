LLVM_ROOT=../..
BIN=$(LLVM_ROOT)/build/bin
BUILD=$(LLVM_ROOT)/build


TESTS=$(SOURCES:%.c)

all: licm dse cse dce adce

licm:
	$(BIN)/clang -S -emit-llvm typeinfo/licm_test.c -O0 -Xclang -disable-O0-optnone -o typeinfo/licm_test.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg  typeinfo/licm_test.ll -S -o typeinfo/licm.mem2reg.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg -tensor -licm  typeinfo/licm_test.ll -S -o typeinfo/licm_test.licm.ll
	$(BIN)/clang -S -emit-llvm typeinfo/licm_test2.c -O0 -Xclang -disable-O0-optnone -o typeinfo/licm_test2.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg -tensor -licm  typeinfo/licm_test2.ll -S -o typeinfo/licm_test2.licm.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -load $(BUILD)/lib/LLVMTensorProperties.so -mem2reg -tensor -tensor-analysis typeinfo/licm_test2.ll -S -o typeinfo/licm_test2.tensor.ll

dse:
	$(BIN)/clang -S -emit-llvm typeinfo/dse_test.c -O0 -Xclang -disable-O0-optnone -o typeinfo/dse_test.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg -tensor  -dse  dse_test.ll -S -o typeinfo/dse_test.dse.ll
	$(BIN)/clang -S -emit-llvm typeinfo/dse_test2.c -O0 -Xclang -disable-O0-optnone -o typeinfo/dse_test2.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg -tensor  -dse  typeinfo/dse_test2.ll -S -o typeinfo/dse_test2.dse.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -load $(BUILD)/lib/LLVMTensorProperties.so -mem2reg -tensor -tensor-analysis typeinfo/dse_test2.ll -S -o typeinfo/dse_test2.tensor.ll

cse:
	$(BIN)/clang -S -emit-llvm cse_test.c -O0 -Xclang -disable-O0-optnone -o typeinfo/cse_test.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg -tensor -early-cse  typeinfo/cse_test.ll -S -o typeinfo/cse_test.cse.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -load $(BUILD)/lib/LLVMTensorProperties.so -mem2reg -tensor -tensor-analysis cse_test.ll -S -o cse_test.tensor.ll
	$(BIN)/clang -S -emit-llvm typeinfo/cse_test2.c -O0 -Xclang -disable-O0-optnone -o typeinfo/cse_test2.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg -tensor -early-cse  typeinfo/cse_test2.ll -S -o typeinfo/cse_test2.cse.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -load $(BUILD)/lib/LLVMTensorProperties.so -mem2reg -tensor -tensor-analysis typeinfo/cse_test2.ll -S -o typeinfo/cse_test2.tensor.ll

dce:
	$(BIN)/clang -S -emit-llvm dce_test.c -O0 -Xclang -disable-O0-optnone -o typeinfo/dce_test.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg  typeinfo/dce_test.ll -S -o typeinfo/dce.mem2reg.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg -tensor -dce  typeinfo/dce_test.ll -S -o typeinfo/dce_test.dce.ll

adce:
	$(BIN)/clang -S -emit-llvm adce_test.c -O0 -Xclang -disable-O0-optnone -o typeinfo/adce_test.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -mem2reg -tensor -adce  typeinfo/adce_test.ll -S -o typeinfo/adce_test.adce.ll


lower_matmul1:
	$(BIN)/clang -S -emit-llvm matmul/matmul1.c -O0 -Xclang -disable-O0-optnone -o matmul/matmul1.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -enable-new-pm=0 -mem2reg -tensor -tensor-analysis -lower-tensor matmul/matmul1.ll -S -o matmul/matmul1.lowered.ll

lower_matmul2:
	$(BIN)/clang -S -emit-llvm matmul/matmul2.c -O0 -Xclang -disable-O0-optnone -o matmul/matmul2.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -enable-new-pm=0 -mem2reg -tensor -tensor-analysis -lower-tensor matmul/matmul2.ll -S -o matmul/matmul2.lowered.ll

lower_matmul3:
	$(BIN)/clang -S -emit-llvm matmul/matmul3.c -O0 -Xclang -disable-O0-optnone -o matmul/matmul3.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -enable-new-pm=0 -mem2reg -tensor -tensor-analysis -lower-tensor matmul/matmul3.ll -S -o matmul/matmul3.lowered.ll


lower_reduction1:
	$(BIN)/clang -S -emit-llvm reduction/reduction1.c -O0 -Xclang -disable-O0-optnone -o reduction/reduction1.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -enable-new-pm=0 -mem2reg -tensor -tensor-analysis -lower-tensor reduction/reduction1.ll -S -o reduction/reduction1.lowered.ll

lower_reduction2:
	$(BIN)/clang -S -emit-llvm reduction/reduction2.c -O0 -Xclang -disable-O0-optnone  -o reduction/reduction2.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -enable-new-pm=0 -mem2reg -tensor -tensor-analysis -lower-tensor reduction/reduction2.ll -S -o reduction/reduction2.lowered.ll

lower_reduction2:
	$(BIN)/clang -S -emit-llvm reduction/reduction3.c -O0 -Xclang -disable-O0-optnone -o  -o reduction/reduction3.ll
	$(BIN)/opt -load $(BUILD)/lib/LLVMTensor.so -enable-new-pm=0 -mem2reg -tensor -tensor-analysis -lower-tensor reduction/reduction3.ll -S -o reduction/reduction3.lowered.ll


clean:
	rm -rf *.ll

