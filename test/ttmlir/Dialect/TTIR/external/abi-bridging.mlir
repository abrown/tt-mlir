// RUN: ttmlir-opt --ttir-link-external-functions -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Check the ABI bridging inserted between the XLA caller's types and the
// linked-in kernel's parameter types:
//
//   Case 1 (tensor.cast, inputs):  static-shaped tensor  → dynamic tensor
//   Case 3 (tensor.cast, results): dynamic result tensor → concrete tensor
//   Case 4 (pass-through):         exact type match      → no bridging ops

module {
  // Cases 1 and 3 are exercised together.
  //
  // The caller provides:
  //   %mat  : tensor<32x64xf32>  – static shape, callee expects tensor<?x?xf32>
  //
  // The invoke op declares tensor<32x64xf32> as its result type while the
  // callee returns tensor<?x?xf32>, so the result also needs a cast.
  //
  // CHECK-LABEL: func.func @test_abi_bridging
  // CHECK:         tensor.cast {{.*}} : tensor<32x64xf32> to tensor<?x?xf32>
  // CHECK:         call @dynamic_kernel
  // CHECK:         tensor.cast {{.*}} : tensor<?x?xf32> to tensor<32x64xf32>
  func.func @test_abi_bridging(%mat: tensor<32x64xf32>) -> tensor<32x64xf32> {
    %0 = "ttir.invoke_external"(%mat)
         {path = "abi-bridging.mlir.ext", entry = "dynamic_kernel"}
         : (tensor<32x64xf32>) -> tensor<32x64xf32>
    return %0 : tensor<32x64xf32>
  }

  // Case 4: the caller's argument and result types exactly match the callee's
  // parameter and result types.  No tensor.cast or tensor.extract should be
  // emitted anywhere between the function entry and the call.
  //
  // CHECK-LABEL: func.func @test_passthrough
  // CHECK-NOT:   tensor.cast
  // CHECK-NOT:   tensor.extract
  // CHECK:       call @passthrough(%arg0)
  func.func @test_passthrough(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = "ttir.invoke_external"(%arg0)
         {path = "abi-bridging.mlir.ext", entry = "passthrough"}
         : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
