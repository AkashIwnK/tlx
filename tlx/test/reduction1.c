#include "tensor.h"
#include <stdio.h>

void print(_tensor_t tensor) {
  printf("start\n");
   for(unsigned i = 0; i < 2; i++) {
    printf("printing tensor: %d\n", tensor[i]);
  }
  printf("end\n");
}

_tensor_t  foo(_tensor_t tensor) {
  _shape_t shape = {1, 1, 4, 4};
  _layout_t layout =  {0, 1, 2, 3};
  _padding_t padding =  {0, 0, 0, 0};
  _stride_t stride = {1, 1, 2, 2};
  _window_t window = {1, 1, 2, 2};
  _shape_t shape1 = {1, 1, 2, 2};

  _token_t tensor_token =  tensor_typeinfo(tensor, shape, layout, padding);

  _tensor_t tensor1 = tensor_reduce_add(tensor_token, window, stride);
  _token_t tensor1_token = tensor_typeinfo(tensor1, shape1, layout, padding);

  return tensor1;
}


int main(void) {

  _tensor_t tensor = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  _tensor_t tensor3 = foo(tensor);

  unsigned output_size = 4;
  for(unsigned i = 0; i < output_size; i++) {
    printf("output: %d\n", tensor3[i]);
  }

  return 0;
}

