// RUN: circt-opt --lower-dc-to-hw %s | FileCheck %s

dc.func @max(%arg0: !dc.value<i64>, %arg1: !dc.value<i64>, %arg2: !dc.value<i64>, %arg3: !dc.value<i64>, %arg4: !dc.value<i64>, %arg5: !dc.value<i64>, %arg6: !dc.value<i64>, %arg7: !dc.value<i64>) -> !dc.value<i64> {
  %token, %outputs = dc.unpack %arg0 : !dc.value<i64>
  %token_0, %outputs_1 = dc.unpack %arg1 : !dc.value<i64>
  %0 = comb.icmp slt, %outputs, %outputs_1 : i64
  %1 = comb.mux %0, %outputs, %outputs_1 : i64
  %token_2, %outputs_3 = dc.unpack %arg2 : !dc.value<i64>
  %token_4, %outputs_5 = dc.unpack %arg3 : !dc.value<i64>
  %2 = comb.icmp slt, %outputs_3, %outputs_5 : i64
  %3 = comb.mux %2, %outputs_3, %outputs_5 : i64
  %token_6, %outputs_7 = dc.unpack %arg4 : !dc.value<i64>
  %token_8, %outputs_9 = dc.unpack %arg5 : !dc.value<i64>
  %4 = comb.icmp slt, %outputs_7, %outputs_9 : i64
  %5 = comb.mux %4, %outputs_7, %outputs_9 : i64
  %token_10, %outputs_11 = dc.unpack %arg6 : !dc.value<i64>
  %token_12, %outputs_13 = dc.unpack %arg7 : !dc.value<i64>
  %6 = comb.icmp slt, %outputs_11, %outputs_13 : i64
  %7 = comb.mux %6, %outputs_11, %outputs_13 : i64
  %8 = comb.icmp slt, %1, %3 : i64
  %9 = comb.mux %8, %1, %3 : i64
  %10 = comb.icmp slt, %5, %7 : i64
  %11 = comb.mux %10, %5, %7 : i64
  %12 = comb.icmp slt, %9, %11 : i64
  %13 = dc.join %token, %token_0, %token_2, %token_4, %token_6, %token_8, %token_10, %token_12
  %14 = comb.mux %12, %9, %11 : i64
  %15 = dc.pack %13[%14] : (i64) -> !dc.value<i64>
  dc.return %15 : !dc.value<i64>
}
