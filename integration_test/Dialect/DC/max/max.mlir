// REQUIRES: iverilog,cocotb

// RUN: circt-opt %s \
// RUN:    --lower-dc-to-hw \
// RUN:    --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw \
// RUN:    --export-verilog -o %t_low.mlir > %t.sv

// RUN: %PYTHON% %S/../../../cocotb_driver.py \
// RUN:   --objdir=%T                         \
// RUN:   --topLevel=top                      \
// RUN:   --pythonModule=max                  \
// RUN:   --pythonFolder=%S %t.sv 2>&1 | FileCheck %s


hw.module @top(%in0: !dc.value<i64>, %in1: !dc.value<i64>, %in2: !dc.value<i64>, %in3: !dc.value<i64>, %in4: !dc.value<i64>, %in5: !dc.value<i64>, %in6: !dc.value<i64>, %in7: !dc.value<i64>, %in8: !dc.token) -> (out0: !dc.value<i64>, out1: !dc.token) {
  %token, %outputs = dc.unpack %in0 : !dc.value<i64>
  %token_0, %outputs_1 = dc.unpack %in1 : !dc.value<i64>
  %token_2, %outputs_3 = dc.unpack %in2 : !dc.value<i64>
  %token_4, %outputs_5 = dc.unpack %in3 : !dc.value<i64>
  %token_6, %outputs_7 = dc.unpack %in4 : !dc.value<i64>
  %token_8, %outputs_9 = dc.unpack %in5 : !dc.value<i64>
  %token_10, %outputs_11 = dc.unpack %in6 : !dc.value<i64>
  %token_12, %outputs_13 = dc.unpack %in7 : !dc.value<i64>
  %0 = arith.cmpi sge, %outputs, %outputs_1 : i64
  %1 = arith.select %0, %outputs, %outputs_1 : i64
  %2 = arith.cmpi sge, %outputs_3, %outputs_5 : i64
  %3 = arith.select %2, %outputs_3, %outputs_5 : i64
  %4 = arith.cmpi sge, %outputs_7, %outputs_9 : i64
  %5 = arith.select %4, %outputs_7, %outputs_9 : i64
  %6 = arith.cmpi sge, %outputs_11, %outputs_13 : i64
  %7 = arith.select %6, %outputs_11, %outputs_13 : i64
  %8 = arith.cmpi sge, %1, %3 : i64
  %9 = arith.select %8, %1, %3 : i64
  %10 = arith.cmpi sge, %5, %7 : i64
  %11 = arith.select %10, %5, %7 : i64
  %12 = arith.cmpi sge, %9, %11 : i64
  %14 = arith.select %12, %9, %11 : i64
  %13 = dc.join %token, %token_0, %token_2, %token_4, %token_6, %token_8, %token_10, %token_12
  %15 = dc.pack %13[%14] : i64
  hw.output %15, %in8 : !dc.value<i64>, !dc.token
}
