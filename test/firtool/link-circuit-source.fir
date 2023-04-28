; This is the source circuit that is linked to `link-circuit-target.fir`
; RUN: true

FIRRTL version 3.0.0
circuit Source:
  module Source:
    output direct_probe: Probe<UInt<1>>
    output inner_probe: Probe<UInt<2>>
    output keyword_probe: Probe<UInt<3>>
    
    wire direct: UInt<1>
    invalidate direct
    define direct_probe = probe(direct)

    inst inner of InnerSource
    define inner_probe = inner.x_probe

    wire always: UInt<3>
    invalidate always
    define keyword_probe = probe(always)
  
  module InnerSource:
    output x_probe: Probe<UInt<2>>
    wire x: UInt<2>
    invalidate x
    define x_probe = probe(x)
