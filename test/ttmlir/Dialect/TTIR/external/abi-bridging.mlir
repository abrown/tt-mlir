// RUN: ttmlir-opt --ttir-link-external-functions -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Check the ABI bridging inserted between the XLA caller's types and the
// linked-in kernel's parameter types:
//
//   Case 1 (tensor.cast, inputs):  static-shaped tensor  → dynamic tensor
//   Case 2 (tensor.extract):       0-D scalar tensor     → bare scalar
//   Case 3 (tensor.cast, results): dynamic result tensor → concrete tensor
//   Case 4 (pass-through):         exact type match      → no bridging ops
//   Case 5 (extract + trunci):     0-D wide-int tensor   → narrow scalar (i1)
//   Case 6 (1D extract):           tensor<1xT>           → bare scalar T
//   Case 8 (extract + bitcast + extsi):  tensor<si32> → i64 (signed ext)
//   Case 9 (extract + bitcast + extui):  tensor<ui32> → i64 (unsigned ext)
//   Case 10 (extract + bitcast + trunci): tensor<si64> → i32 (truncation)

module {
  // Cases 1, 2, and 3 are exercised together.
  //
  // The caller provides:
  //   %mat  : tensor<32x64xf32>  – static shape, callee expects tensor<?x?xf32>
  //   %n    : tensor<i32>        – 0-D scalar tensor, callee expects i32
  //
  // The invoke op declares tensor<32x64xf32> as its result type while the
  // callee returns tensor<?x?xf32>, so the result also needs a cast.
  //
  // CHECK-LABEL: func.func @test_abi_bridging
  // CHECK:         tensor.cast {{.*}} : tensor<32x64xf32> to tensor<?x?xf32>
  // CHECK:         tensor.extract {{.*}}[] : tensor<i32>
  // CHECK:         call @dynamic_kernel
  // CHECK:         tensor.cast {{.*}} : tensor<?x?xf32> to tensor<32x64xf32>
  func.func @test_abi_bridging(%mat: tensor<32x64xf32>,
                                %n: tensor<i32>) -> tensor<32x64xf32> {
    %0 = "ttir.invoke_external"(%mat, %n)
         {path = "abi-bridging.mlir.ext", entry = "dynamic_kernel"}
         : (tensor<32x64xf32>, tensor<i32>) -> tensor<32x64xf32>
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

  // Case 5: 0-D tensor<i64> → i1 (tensor.extract followed by arith.trunci).
  //
  // The caller wraps a boolean flag in a 0-D i64 tensor (common from XLA).
  // The callee expects a bare i1, so the bridging must:
  //   1. Extract the i64 value from the 0-D tensor with tensor.extract.
  //   2. Truncate it from i64 to i1 with arith.trunci.
  //
  // CHECK-LABEL: func.func @test_i64_to_i1
  // CHECK:         tensor.cast {{.*}} : tensor<4x4xf32> to tensor<?x?xf32>
  // CHECK:         tensor.extract {{.*}}[] : tensor<i64>
  // CHECK:         arith.trunci {{.*}} : i64 to i1
  // CHECK:         call @flagged_kernel
  // CHECK:         tensor.cast {{.*}} : tensor<?x?xf32> to tensor<4x4xf32>
  func.func @test_i64_to_i1(%mat: tensor<4x4xf32>,
                              %flag: tensor<i64>) -> tensor<4x4xf32> {
    %0 = "ttir.invoke_external"(%mat, %flag)
         {path = "abi-bridging.mlir.ext", entry = "flagged_kernel"}
         : (tensor<4x4xf32>, tensor<i64>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // Case 6: tensor<1xf32> → f32 (1D tensor.extract with an explicit index 0).
  //
  // The caller wraps a scalar in a rank-1 size-1 tensor.  The callee expects a
  // bare f32, so the bridging emits arith.constant 0 : index and
  // tensor.extract %arg[%c0].
  //
  // CHECK-LABEL: func.func @test_1d_scalar
  // CHECK:         arith.constant 0 : index
  // CHECK:         tensor.extract {{.*}}[{{.*}}] : tensor<1xf32>
  // CHECK:         call @scale_kernel
  // CHECK:         tensor.cast {{.*}} : tensor<?x?xf32> to tensor<4x4xf32>
  func.func @test_1d_scalar(%scale: tensor<1xf32>,
                              %mat: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = "ttir.invoke_external"(%scale, %mat)
         {path = "abi-bridging.mlir.ext", entry = "scale_kernel"}
         : (tensor<1xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // Case 8: tensor<si32> → i64 (sign-extend across a signedness boundary).
  //
  // The caller wraps a signed 32-bit integer in a 0-D tensor.  The callee
  // expects a bare signless i64.  Bridging must:
  //   1. Extract the si32 value from the 0-D tensor with tensor.extract.
  //   2. Bitcast si32 → i32 (arith ops require signless operands).
  //   3. Sign-extend i32 → i64 with arith.extsi (preserves the signed value).
  // A warning is emitted about the signedness conversion.
  //
  // CHECK-LABEL: func.func @test_si32_to_i64
  // CHECK:         tensor.extract {{.*}}[] : tensor<si32>
  // CHECK:         builtin.unrealized_conversion_cast {{.*}} : si32 to i32
  // CHECK:         arith.extsi {{.*}} : i32 to i64
  // CHECK:         call @signed_ext_kernel
  // CHECK:         tensor.cast {{.*}} : tensor<?x?xf32> to tensor<4x4xf32>
  func.func @test_si32_to_i64(%flag: tensor<si32>,
                               %mat: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = "ttir.invoke_external"(%flag, %mat)
         {path = "abi-bridging.mlir.ext", entry = "signed_ext_kernel"}
         : (tensor<si32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // Case 9: tensor<ui32> → i64 (zero-extend across a signedness boundary).
  //
  // The caller wraps an unsigned 32-bit integer in a 0-D tensor.  The callee
  // expects a bare signless i64.  Bridging must:
  //   1. Extract the ui32 value from the 0-D tensor with tensor.extract.
  //   2. Bitcast ui32 → i32 (arith ops require signless operands).
  //   3. Zero-extend i32 → i64 with arith.extui (preserves the unsigned value).
  // A warning is emitted about the signedness conversion.
  //
  // CHECK-LABEL: func.func @test_ui32_to_i64
  // CHECK:         tensor.extract {{.*}}[] : tensor<ui32>
  // CHECK:         builtin.unrealized_conversion_cast {{.*}} : ui32 to i32
  // CHECK:         arith.extui {{.*}} : i32 to i64
  // CHECK:         call @unsigned_ext_kernel
  // CHECK:         tensor.cast {{.*}} : tensor<?x?xf32> to tensor<4x4xf32>
  func.func @test_ui32_to_i64(%flag: tensor<ui32>,
                               %mat: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = "ttir.invoke_external"(%flag, %mat)
         {path = "abi-bridging.mlir.ext", entry = "unsigned_ext_kernel"}
         : (tensor<ui32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // Case 10: tensor<si64> → i32 (truncation from a signed type).
  //
  // The caller wraps a signed 64-bit integer in a 0-D tensor.  The callee
  // expects a bare signless i32.  Bridging must:
  //   1. Extract the si64 value from the 0-D tensor with tensor.extract.
  //   2. Bitcast si64 → i64 (arith ops require signless operands).
  //   3. Truncate i64 → i32 with arith.trunci (high bits are discarded).
  // Warnings are emitted for both the truncation and the signedness conversion.
  //
  // CHECK-LABEL: func.func @test_si64_to_i32
  // CHECK:         tensor.extract {{.*}}[] : tensor<si64>
  // CHECK:         builtin.unrealized_conversion_cast {{.*}} : si64 to i64
  // CHECK:         arith.trunci {{.*}} : i64 to i32
  // CHECK:         call @truncated_kernel
  // CHECK:         tensor.cast {{.*}} : tensor<?x?xf32> to tensor<4x4xf32>
  func.func @test_si64_to_i32(%flag: tensor<si64>,
                               %mat: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = "ttir.invoke_external"(%flag, %mat)
         {path = "abi-bridging.mlir.ext", entry = "truncated_kernel"}
         : (tensor<si64>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
