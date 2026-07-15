// RUN: ttmlir-opt --ttnn-inline-linked-kernels -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// `ttnn-inline-linked-kernels` must only inline injected kernels (callees whose
// body contains a `ttnn.generic`). Any other `func.call` must be left intact.

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  // The call to @not_a_kernel must NOT be inlined: the callee has no
  // ttnn.generic, so it is not recognized as a linked kernel.
  // CHECK-LABEL: func.func @caller
  // CHECK: call @not_a_kernel
  func.func @caller(%arg0: tensor<64x32xf32, #layout>) -> tensor<64x32xf32, #layout> {
    %0 = call @not_a_kernel(%arg0) : (tensor<64x32xf32, #layout>) -> tensor<64x32xf32, #layout>
    return %0 : tensor<64x32xf32, #layout>
  }

  // CHECK: func.func private @not_a_kernel
  func.func private @not_a_kernel(%arg0: tensor<64x32xf32, #layout>) -> tensor<64x32xf32, #layout> {
    return %arg0 : tensor<64x32xf32, #layout>
  }
}
