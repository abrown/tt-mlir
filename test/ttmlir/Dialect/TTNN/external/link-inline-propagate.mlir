// RUN: ttmlir-opt --ttir-link-external-functions --ttnn-inline-linked-kernels --ttnn-propagate-link-layout --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// End-to-end check of the link -> inline -> propagate flow for an external
// kernel injected via `ttir.invoke_external`:
//   - The kernel is compiled with dynamic shapes and a placeholder `ttnn_layout`
//     (1x1 grid). The caller (graph) has concrete static shapes and the real
//     `ttnn_layout` (2x1 grid).
//   - `ttir-link-external-functions` bridges the difference with `tensor.cast`.
//   - `ttnn-inline-linked-kernels` inlines the kernel into the caller.
//   - `ttnn-propagate-link-layout` forwards the graph shape + `ttnn_layout` into
//     the `ttnn.generic` operands and removes the casts.

#dram = #ttnn.buffer_type<dram>
#graph_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  // CHECK-LABEL: func.func @test_link_inline_propagate
  // After propagation there should be no bridging casts left, the kernel must
  // be inlined (no func.call), and the `ttnn.generic` operands must carry the
  // graph layout (static 64x32 shape, graph_layout encoding).
  // CHECK-NOT: tensor.cast
  // CHECK-NOT: call @kernel
  // CHECK: "ttnn.generic"(%arg0, %arg1)
  // CHECK-SAME: (tensor<64x32xf32, #ttnn_layout>, tensor<64x32xf32, #ttnn_layout>) -> ()
  func.func @test_link_inline_propagate(
      %in: tensor<64x32xf32, #graph_layout>,
      %out: tensor<64x32xf32, #graph_layout>) -> tensor<64x32xf32, #graph_layout> {
    %0 = "ttir.invoke_external"(%in, %out)
         {path = "link-inline-propagate.mlir.ext", entry = "kernel"}
         : (tensor<64x32xf32, #graph_layout>, tensor<64x32xf32, #graph_layout>)
         -> tensor<64x32xf32, #graph_layout>
    return %0 : tensor<64x32xf32, #graph_layout>
  }

  // The kernel entry function must be removed after inlining, but the helper
  // kernel function referenced by the program attribute must be retained.
  // CHECK-NOT: func.func {{.*}}@kernel(
  // CHECK: func.func private @datamovement_kernel0
}
