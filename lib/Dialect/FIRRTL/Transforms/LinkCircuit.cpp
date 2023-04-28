//===- LowerXMR.cpp - FIRRTL link circuit -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements FIRRTL circuit linking.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-link-circuit"

using namespace circt;
using namespace firrtl;

namespace {
class LinkCircuitPass : public LinkCircuitBase<LinkCircuitPass> {
public:
  void runOnOperation() override {
    auto circuits = getOperation().getOps<CircuitOp>();
    if (circuits.empty()) {
      getOperation()->emitError("expected circuit");
      signalPassFailure();
      return;
    }

    auto circuit = *circuits.begin();

    llvm::StringMap<llvm::StringMap<StringAttr>> linkedModuleRefPortPaths;

    for (auto exportRef :
         llvm::make_early_inc_range(getOperation().getOps<hw::ExportRefOp>())) {
      linkedModuleRefPortPaths[exportRef.getModuleName()]
                              [exportRef.getRefName()] =
                                  exportRef.getInternalPathAttr();
      exportRef->erase();
    }

    for (auto extModule : circuit.getOps<FExtModuleOp>()) {
      StringRef exportName;
      if (auto defname = extModule.getDefname())
        exportName = *defname;
      else
        exportName = extModule.getName();

      auto refPortPaths = linkedModuleRefPortPaths.find(exportName);
      if (refPortPaths == linkedModuleRefPortPaths.end())
        continue;

      SmallVector<Attribute> internalPaths;
      for (size_t portIndex = 0, numPorts = extModule.getNumPorts();
           portIndex != numPorts; ++portIndex) {
        if (!extModule.getPortType(portIndex).isa<RefType>())
          continue;
        auto refName = extModule.getPortName(portIndex);
        auto internalPath = refPortPaths->second.find(refName);
        if (internalPath == refPortPaths->second.end()) {
          extModule->emitError("failed to link ref port " + refName);
          signalPassFailure();
          return;
        }
        internalPaths.push_back(internalPath->second);
      }
      extModule.setInternalPathsAttr(
          ArrayAttr::get(&getContext(), internalPaths));
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createLinkCircuitPass() {
  return std::make_unique<LinkCircuitPass>();
}
