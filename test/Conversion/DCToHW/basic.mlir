// RUN: circt-opt --lower-dc-to-hw %s | FileCheck %s

hw.module @simple(%0 : !dc.token, %1 : !dc.value<i64>, %2 : i1, %3 : !dc.value<i1, i2>)
        -> (out0: !dc.token, out1: !dc.value<i64>, out2: i1, out3: !dc.value<i1, i2>) {
    hw.output %0, %1, %2, %3 : !dc.token, !dc.value<i64>, i1, !dc.value<i1, i2>
}

hw.module @pack(%token : !dc.token, %v1 : i64, %v2 : i1) -> (out0: !dc.value<i64, i1>) {
    %out = dc.pack %token [%v1, %v2] : i64, i1
    hw.output %out : !dc.value<i64, i1>
}

hw.module @unpack(%v : !dc.value<i64, i1>) -> (out0: !dc.token, out1: i64, out2: i1) {
    %out:3 = dc.unpack %v : !dc.value<i64, i1>
    hw.output %out#0, %out#1, %out#2 : !dc.token, i64, i1
}

hw.module @join(%t1 : !dc.token, %t2 : !dc.token) -> (out0: !dc.token) {
    %out = dc.join %t1, %t2
    hw.output %out : !dc.token
}

hw.module @fork(%t : !dc.token, %clk : i1 {"dc.clock"}, %rst : i1 {"dc.reset"}) -> (out0: !dc.token, out1: !dc.token) {
    %out:2 = dc.fork [2] %t
    hw.output %out#0, %out#1 : !dc.token, !dc.token
}

hw.module @bufferToken(%t1 : !dc.token, %clk : i1 {"dc.clock"}, %rst : i1 {"dc.reset"}) -> (out0: !dc.token) {
    %out = dc.buffer [2] %t1 : !dc.token
    hw.output %out : !dc.token
}

hw.module @bufferValue(%v1 : !dc.value<i64>, %clk : i1 {"dc.clock"}, %rst : i1 {"dc.reset"}) -> (out0: !dc.value<i64>) {
    %out = dc.buffer [2] %v1 : !dc.value<i64>
    hw.output %out : !dc.value<i64>
}

hw.module @branch(%sel : !dc.value<i1>) -> (out0: !dc.token, out1: !dc.token) {
    %true, %false = dc.branch %sel
    hw.output %true, %false : !dc.token, !dc.token
}

hw.module @select(%sel : !dc.value<i1>, %true : !dc.token, %false : !dc.token) -> (out0: !dc.token) {
    %0 = dc.select %sel, %true, %false
    hw.output %0 : !dc.token
}
