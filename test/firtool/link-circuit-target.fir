; Test linking circuits

; First build the source circuit
; RUN: firtool %S/link-circuit-source.fir -split-verilog -o %t

; Then build this circuit, importing the exports from the source circuit
; RUN: firtool %s -link-circuit-exports %t/exports.mlir | FileCheck %s

FIRRTL version 3.0.0
circuit Target:
  extmodule Source:
    output direct_probe: Probe<UInt<1>>
    output inner_probe: Probe<UInt<2>>
    output keyword_probe: Probe<UInt<3>>

  module Target:
    output a: UInt<1>
    output b: UInt<2>
    output c: UInt<3>
    inst source of Source
    a <= read(source.direct_probe)
    b <= read(source.inner_probe)
    c <= read(source.keyword_probe)

; CHECK-LABEL: module Target(
; CHECK-NEXT:     output a,
; CHECK-NEXT:     b,
; CHECK-NEXT:     c
; CHECK-NEXT:  );
; CHECK-EMPTY:
; CHECK-NEXT:     Source source ();
; CHECK-NEXT:     assign a = Target.source.direct;
; CHECK-NEXT:     assign b = Target.source.inner.x;
; CHECK-NEXT:     assign c = Target.source.always_0;
; CHECK-NEXT:  endmodule
