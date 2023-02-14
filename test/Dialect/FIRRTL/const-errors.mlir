// RUN: circt-opt %s -split-input-file -verify-diagnostics

firrtl.circuit "test" {
// expected-error @+1 {{'const' can only be specified once the outermost 'const' type}}
firrtl.module @test(in %a: !firrtl.const.const.uint<1>) {}
}

// -----

firrtl.circuit "test" {
// expected-error @+1 {{'const' can only be specified once the outermost 'const' type}}
firrtl.module @test(in %a: !firrtl.const.bundle<a: const.uint<1>>) {}
}

// -----

firrtl.circuit "test" {
// expected-error @+1 {{'const' can only be specified once the outermost 'const' type}}
firrtl.module @test(in %a: !firrtl.const.vector<const.uint<1>, 2>) {}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a: !firrtl.const.uint<1>, in %b: !firrtl.uint<1>) {
  // expected-error @+1 {{operand constness must match}}
  %0 = firrtl.add %a, %b : (!firrtl.const.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
}
}
