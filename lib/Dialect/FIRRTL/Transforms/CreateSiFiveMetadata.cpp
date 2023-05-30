//===- CreateSiFiveMetadata.cpp - Create various metadata -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CreateSiFiveMetadata pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/OM/OMAttributes.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMOps.h"
#include "circt/Dialect/OM/OMTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/JSON.h"

using namespace circt;
using namespace firrtl;
using circt::hw::InstancePath;

namespace {
class CreateSiFiveMetadataPass
    : public CreateSiFiveMetadataBase<CreateSiFiveMetadataPass> {
  LogicalResult emitRetimeModulesMetadata();
  LogicalResult emitSitestBlackboxMetadata();
  LogicalResult emitMemoryMetadata();
  void getDependentDialects(mlir::DialectRegistry &registry) const override;
  void runOnOperation() override;

  // The set of all modules underneath the design under test module.
  DenseSet<Operation *> dutModuleSet;
  // The design under test module.
  FModuleOp dutMod;
  CircuitOp circuitOp;

public:
  CreateSiFiveMetadataPass(bool _replSeqMem, StringRef _replSeqMemCircuit,
                           StringRef _replSeqMemFile) {
    replSeqMem = _replSeqMem;
    replSeqMemCircuit = _replSeqMemCircuit.str();
    replSeqMemFile = _replSeqMemFile.str();
  }
};
} // end anonymous namespace

/// This function collects all the firrtl.mem ops and creates a verbatim op with
/// the relevant memory attributes.
LogicalResult CreateSiFiveMetadataPass::emitMemoryMetadata() {
  if (!replSeqMem)
    return success();

  // The instance graph analysis will be required to print the hierarchy names
  // of the memory.
  auto instancePathCache = InstancePathCache(getAnalysis<InstanceGraph>());

  // Everything goes in the DUT if (1) there is no DUT specified or (2) if the
  // DUT is the top module.
  bool everythingInDUT =
      !dutMod ||
      instancePathCache.instanceGraph.getTopLevelNode()->getModule() == dutMod;

  auto *context = circuitOp.getContext();
  auto builderOM = OpBuilder::atBlockEnd(circuitOp.getBodyBlock());

  using NameTypePair = std::pair<StringAttr, mlir::Type>;
  SmallVector<NameTypePair> classFields;
  classFields.push_back({StringAttr::get(context, "name"),
                         om::StringOMType::get(builderOM.getContext())});
  classFields.push_back(
      {StringAttr::get(context, "depth"),
       mlir::IntegerType::get(context, 64, IntegerType::Unsigned)});
  classFields.push_back(
      {StringAttr::get(context, "width"),
       mlir::IntegerType::get(context, 32, IntegerType::Unsigned)});
  classFields.push_back(
      {StringAttr::get(context, "maskBits"),
       mlir::IntegerType::get(context, 32, IntegerType::Unsigned)});
  classFields.push_back(
      {StringAttr::get(context, "readPorts"),
       mlir::IntegerType::get(context, 32, IntegerType::Unsigned)});
  classFields.push_back(
      {StringAttr::get(context, "writePorts"),
       mlir::IntegerType::get(context, 32, IntegerType::Unsigned)});
  classFields.push_back(
      {StringAttr::get(context, "readwritePorts"),
       mlir::IntegerType::get(context, 32, IntegerType::Unsigned)});
  classFields.push_back(
      {StringAttr::get(context, "writeLatency"),
       mlir::IntegerType::get(context, 32, IntegerType::Unsigned)});
  classFields.push_back(
      {StringAttr::get(context, "readLatency"),
       mlir::IntegerType::get(context, 32, IntegerType::Unsigned)});

  auto unknownLoc = builderOM.getUnknownLoc();
  auto paramNames = builderOM.getArrayAttr(llvm::to_vector(llvm::map_range(
      classFields, [&](NameTypePair e) -> Attribute { return e.first; })));

  auto memMetadataClass = builderOM.create<circt::om::ClassOp>(
      builderOM.getUnknownLoc(),
      StringAttr::get(builderOM.getContext(), "MemoryConfModules"), paramNames);
  Block *body = new Block();
  memMetadataClass.getRegion().push_back(body);
  builderOM.setInsertionPointToEnd(body);
  for (auto field : classFields)
    builderOM.create<om::ClassFieldOp>(
        unknownLoc, field.first, body->addArgument(field.second, unknownLoc));

  builderOM.setInsertionPointToEnd(circuitOp.getBodyBlock());
  auto metadataClass = builderOM.create<circt::om::ClassOp>(
      builderOM.getUnknownLoc(),
      StringAttr::get(builderOM.getContext(), "MemoryMetadata"),
      builderOM.getArrayAttr({}));
  auto *memMetadataBlock = new Block();
  metadataClass.getRegion().push_back(memMetadataBlock);
  builderOM.setInsertionPointToEnd(memMetadataBlock);

  unsigned index = 0;
  // This lambda, writes to the given Json stream all the relevant memory
  // attributes. Also adds the memory attrbutes to the string for creating the
  // memmory conf file.
  auto createMemMetadata = [&](FMemModuleOp mem,
                               llvm::json::OStream &jsonStream,
                               std::string &seqMemConfStr) {
    // Get the memory data width.
    auto width = mem.getDataWidth();
    // Metadata needs to be printed for memories which are candidates for
    // macro replacement. The requirements for macro replacement::
    // 1. read latency and write latency of one.
    // 2. undefined read-under-write behavior.
    if (!((mem.getReadLatency() == 1 && mem.getWriteLatency() == 1) &&
          width > 0))
      return;
    auto createConstField = [&](Type type, Attribute constVal) {
      return builderOM.create<om::ConstantOp>(unknownLoc, type,
                                              constVal.cast<mlir::TypedAttr>());
    };

    SmallVector<Value> memFields;
    for (auto field : classFields)
      memFields.push_back(createConstField(
          field.second,
          llvm::StringSwitch<TypedAttr>(field.first.getValue())
              .Case("name", om::StringOMAttr::get(context, mem.getNameAttr()))
              .Case("depth", mem.getDepthAttr())
              .Case("width", mem.getDataWidthAttr())
              .Case("maskBits", mem.getMaskBitsAttr())
              .Case("readPorts", mem.getNumReadPortsAttr())
              .Case("writePorts", mem.getNumWritePortsAttr())
              .Case("readwritePorts", mem.getNumReadWritePortsAttr())
              .Case("readLatency", mem.getReadLatencyAttr())
              .Case("writeLatency", mem.getWriteLatencyAttr())));

    auto object = builderOM.create<om::ObjectOp>(
        unknownLoc,
        om::ClassType::get(context,
                           mlir::FlatSymbolRefAttr::get(memMetadataClass)),
        memMetadataClass.getNameAttr(), memFields);
    builderOM.create<om::ClassFieldOp>(
        unknownLoc, StringAttr::get(context, "m" + Twine(index++)), object);

    // Compute the mask granularity.
    auto isMasked = mem.isMasked();
    auto maskGran = width / mem.getMaskBits();
    // Now create the config string for the memory.
    std::string portStr;
    for (uint32_t i = 0; i < mem.getNumWritePorts(); ++i) {
      if (!portStr.empty())
        portStr += ",";
      portStr += isMasked ? "mwrite" : "write";
    }
    for (uint32_t i = 0; i < mem.getNumReadPorts(); ++i) {
      if (!portStr.empty())
        portStr += ",";
      portStr += "read";
    }
    for (uint32_t i = 0; i < mem.getNumReadWritePorts(); ++i) {
      if (!portStr.empty())
        portStr += ",";
      portStr += isMasked ? "mrw" : "rw";
    }

    auto memExtName = mem.getName();
    auto maskGranStr =
        !isMasked ? "" : " mask_gran " + std::to_string(maskGran);
    seqMemConfStr = (StringRef(seqMemConfStr) + "name " + memExtName +
                     " depth " + Twine(mem.getDepth()) + " width " +
                     Twine(width) + " ports " + portStr + maskGranStr + "\n")
                        .str();

    // Do not emit any JSON for memories which are not in the DUT.
    if (!everythingInDUT && !dutModuleSet.contains(mem))
      return;
    // This adds a Json array element entry corresponding to this memory.
    jsonStream.object([&] {
      jsonStream.attribute("module_name", memExtName);
      jsonStream.attribute("depth", (int64_t)mem.getDepth());
      jsonStream.attribute("width", (int64_t)width);
      jsonStream.attribute("masked", isMasked);
      jsonStream.attribute("read", mem.getNumReadPorts());
      jsonStream.attribute("write", mem.getNumWritePorts());
      jsonStream.attribute("readwrite", mem.getNumReadWritePorts());
      if (isMasked)
        jsonStream.attribute("mask_granularity", (int64_t)maskGran);
      jsonStream.attributeArray("extra_ports", [&] {
        for (auto attr : mem.getExtraPorts()) {
          jsonStream.object([&] {
            auto port = attr.cast<DictionaryAttr>();
            auto name = port.getAs<StringAttr>("name").getValue();
            jsonStream.attribute("name", name);
            auto direction = port.getAs<StringAttr>("direction").getValue();
            jsonStream.attribute("direction", direction);
            auto width = port.getAs<IntegerAttr>("width").getUInt();
            jsonStream.attribute("width", width);
          });
        }
      });
      // Record all the hierarchy names.
      SmallVector<std::string> hierNames;
      jsonStream.attributeArray("hierarchy", [&] {
        // Get the absolute path for the parent memory, to create the
        // hierarchy names.
        auto paths = instancePathCache.getAbsolutePaths(mem);
        for (auto p : paths) {
          if (p.empty())
            continue;
          auto top = p.front();
          std::string hierName =
              top->getParentOfType<FModuleOp>().getName().str();
          for (auto inst : p) {
            auto parentModule = inst->getParentOfType<FModuleOp>();
            if (dutMod == parentModule)
              hierName = parentModule.getName().str();
            hierName = (Twine(hierName) + "." + inst.getInstanceName()).str();
          }
          hierNames.push_back(hierName);
          // Only include the memory path if it is under the DUT or we are in
          // a situation where everything is deemed to be "in the DUT", i.e.,
          // when the DUT is the top module or when no DUT is specified.
          if (everythingInDUT ||
              llvm::any_of(p, [&](circt::hw::HWInstanceLike inst) {
                return inst.getReferencedModule() == dutMod;
              }))
            jsonStream.value(hierName);
        }
      });
    });
  };

  std::string dutJsonBuffer;
  llvm::raw_string_ostream dutOs(dutJsonBuffer);
  llvm::json::OStream dutJson(dutOs, 2);

  std::string seqMemConfStr;
  dutJson.array([&] {
    for (auto mem : circuitOp.getOps<FMemModuleOp>())
      createMemMetadata(mem, dutJson, seqMemConfStr);
  });

  auto builder = OpBuilder::atBlockEnd(circuitOp.getBodyBlock());
  AnnotationSet annos(circuitOp);
  auto dirAnno = annos.getAnnotation(metadataDirectoryAttrName);
  StringRef metadataDir = "metadata";
  if (dirAnno)
    if (auto dir = dirAnno.getMember<StringAttr>("dirname"))
      metadataDir = dir.getValue();

  // Use unknown loc to avoid printing the location in the metadata files.
  auto dutVerbatimOp =
      builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), dutJsonBuffer);
  auto fileAttr = hw::OutputFileAttr::getFromDirectoryAndFilename(
      context, metadataDir, "seq_mems.json", /*excludeFromFilelist=*/true);
  dutVerbatimOp->setAttr("output_file", fileAttr);

  auto confVerbatimOp =
      builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), seqMemConfStr);
  if (replSeqMemFile.empty()) {
    emitError(circuitOp->getLoc())
        << "metadata emission failed, the option "
           "`-repl-seq-mem-file=<filename>` is mandatory for specifying a "
           "valid seq mem metadata file";
    return failure();
  }

  fileAttr = hw::OutputFileAttr::getFromFilename(context, replSeqMemFile,
                                                 /*excludeFromFilelist=*/true);
  confVerbatimOp->setAttr("output_file", fileAttr);

  return success();
}

/// This will search for a target annotation and remove it from the operation.
/// If the annotation has a filename, it will be returned in the output
/// argument.  If the annotation is missing the filename member, or if more than
/// one matching annotation is attached, it will print an error and return
/// failure.
static LogicalResult removeAnnotationWithFilename(Operation *op,
                                                  StringRef annoClass,
                                                  StringRef &filename) {
  filename = "";
  bool error = false;
  AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
    // If there was a previous error or its not a match, continue.
    if (error || !anno.isClass(annoClass))
      return false;

    // If we have already found a matching annotation, error.
    if (!filename.empty()) {
      op->emitError("more than one ") << annoClass << " annotation attached";
      error = true;
      return false;
    }

    // Get the filename from the annotation.
    auto filenameAttr = anno.getMember<StringAttr>("filename");
    if (!filenameAttr) {
      op->emitError(annoClass) << " requires a filename";
      error = true;
      return false;
    }

    // Require a non-empty filename.
    filename = filenameAttr.getValue();
    if (filename.empty()) {
      op->emitError(annoClass) << " requires a non-empty filename";
      error = true;
      return false;
    }

    return true;
  });

  // If there was a problem above, return failure.
  return failure(error);
}

/// This function collects the name of each module annotated and prints them
/// all as a JSON array.
LogicalResult CreateSiFiveMetadataPass::emitRetimeModulesMetadata() {

  auto *context = &getContext();

  // Get the filename, removing the annotation from the circuit.
  StringRef filename;
  if (failed(removeAnnotationWithFilename(circuitOp, retimeModulesFileAnnoClass,
                                          filename)))
    return failure();

  if (filename.empty())
    return success();

  auto builder = OpBuilder::atBlockEnd(circuitOp.getBodyBlock());

  auto retimeModuleOMClass = builder.create<circt::om::ClassOp>(
      builder.getUnknownLoc(),
      StringAttr::get(builder.getContext(), "RetimeModules"),
      builder.getArrayAttr({StringAttr::get(context, "moduleName")}));
  Block *body = new Block();
  retimeModuleOMClass.getRegion().push_back(body);
  auto arg = body->addArgument(om::ReferenceType::get(builder.getContext()),
                               builder.getUnknownLoc());
  builder.setInsertionPointToEnd(body);
  builder.create<om::ClassFieldOp>(
      builder.getUnknownLoc(),
      StringAttr::get(builder.getContext(), "moduleName"), arg);

  builder.setInsertionPointToEnd(circuitOp.getBodyBlock());
  auto metadataClass = builder.create<circt::om::ClassOp>(
      builder.getUnknownLoc(),
      StringAttr::get(builder.getContext(),
                      "RetimeModulesMetadata_" + filename),
      builder.getArrayAttr({}));
  auto *mBody = new Block();
  metadataClass.getRegion().push_back(mBody);
  builder.setInsertionPointToEnd(mBody);
  // Create a string buffer for the json data.
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  llvm::json::OStream j(os, 2);

  // The output is a json array with each element a module name.
  unsigned index = 0;
  SmallVector<Attribute> symbols;
  SmallString<3> placeholder;
  j.array([&] {
    for (auto module : circuitOp.getBodyBlock()->getOps<FModuleLike>()) {
      // The annotation has no supplemental information, just remove it.
      if (!AnnotationSet::removeAnnotations(module, retimeModuleAnnoClass))
        continue;

      // We use symbol substitution to make sure we output the correct thing
      // when the module goes through renaming.
      j.value(("{{" + Twine(index++) + "}}").str());
      symbols.push_back(SymbolRefAttr::get(module.getModuleNameAttr()));
      auto modEntry = builder.create<om::ConstantOp>(
          builder.getUnknownLoc(), om::ReferenceType::get(context),
          om::ReferenceAttr::get(
              context, hw::InnerRefAttr::get(module.getModuleNameAttr(),
                                             module.getModuleNameAttr())));
      auto object = builder.create<om::ObjectOp>(
          builder.getUnknownLoc(),
          om::ClassType::get(context,
                             FlatSymbolRefAttr::get(retimeModuleOMClass)),
          retimeModuleOMClass.getNameAttr(), ValueRange({modEntry}));
      builder.create<om::ClassFieldOp>(
          builder.getUnknownLoc(),
          StringAttr::get(builder.getContext(), "m" + Twine(index)), object);
    }
  });

  // Put the retime information in a verbatim operation.
  builder.setInsertionPointToEnd(circuitOp.getBodyBlock());
  auto verbatimOp = builder.create<sv::VerbatimOp>(
      builder.getUnknownLoc(), buffer, ValueRange(),
      builder.getArrayAttr(symbols));
  auto fileAttr = hw::OutputFileAttr::getFromFilename(
      context, filename, /*excludeFromFilelist=*/true);
  verbatimOp->setAttr("output_file", fileAttr);
  return success();
}

/// This function finds all external modules which will need to be generated for
/// the test harness to run.
LogicalResult CreateSiFiveMetadataPass::emitSitestBlackboxMetadata() {

  // Any extmodule with these annotations or one of these ScalaClass classes
  // should be excluded from the blackbox list.
  std::array<StringRef, 3> classBlackList = {
      "freechips.rocketchip.util.BlackBoxedROM",
      "sifive.enterprise.grandcentral.MemTap"};
  std::array<StringRef, 6> blackListedAnnos = {
      blackBoxAnnoClass, blackBoxInlineAnnoClass, blackBoxPathAnnoClass,
      dataTapsBlackboxClass, memTapBlackboxClass};

  auto *context = &getContext();

  // Get the filenames from the annotations.
  StringRef dutFilename, testFilename;
  if (failed(removeAnnotationWithFilename(circuitOp, sitestBlackBoxAnnoClass,
                                          dutFilename)) ||
      failed(removeAnnotationWithFilename(
          circuitOp, sitestTestHarnessBlackBoxAnnoClass, testFilename)))
    return failure();

  // If we don't have either annotation, no need to run this pass.
  if (dutFilename.empty() && testFilename.empty())
    return success();

  // Find all extmodules in the circuit. Check if they are black-listed from
  // being included in the list. If they are not, separate them into two
  // groups depending on if theyre in the DUT or the test harness.
  SmallVector<StringRef> dutModules;
  SmallVector<StringRef> testModules;
  for (auto extModule : circuitOp.getBodyBlock()->getOps<FExtModuleOp>()) {
    // If the module doesn't have a defname, then we can't record it properly.
    // Just skip it.
    if (!extModule.getDefname())
      continue;

    // If its a generated blackbox, skip it.
    AnnotationSet annos(extModule);
    if (llvm::any_of(blackListedAnnos, [&](auto blackListedAnno) {
          return annos.hasAnnotation(blackListedAnno);
        }))
      continue;

    // If its a blacklisted scala class, skip it.
    if (auto scalaAnno = annos.getAnnotation(scalaClassAnnoClass)) {
      auto scalaClass = scalaAnno.getMember<StringAttr>("className");
      if (scalaClass &&
          llvm::is_contained(classBlackList, scalaClass.getValue()))
        continue;
    }

    // Record the defname of the module.
    if (!dutMod || dutModuleSet.contains(extModule)) {
      dutModules.push_back(*extModule.getDefname());
    } else {
      testModules.push_back(*extModule.getDefname());
    }
  }

  auto builderOM = OpBuilder::atBlockEnd(circuitOp.getBodyBlock());

  auto sitestBBClass = builderOM.create<circt::om::ClassOp>(
      builderOM.getUnknownLoc(),
      StringAttr::get(builderOM.getContext(), "SitestBlackBoxModules"),
      builderOM.getArrayAttr({StringAttr::get(context, "moduleName")}));
  Block *body = new Block();
  sitestBBClass.getRegion().push_back(body);
  auto arg = body->addArgument(om::StringOMType::get(builderOM.getContext()),
                               builderOM.getUnknownLoc());
  builderOM.setInsertionPointToEnd(body);
  builderOM.create<om::ClassFieldOp>(
      builderOM.getUnknownLoc(),
      StringAttr::get(builderOM.getContext(), "moduleName"), arg);
  auto buildOMClass = [&](StringRef className) {
    builderOM.setInsertionPointToEnd(circuitOp.getBodyBlock());
    auto metadataClass = builderOM.create<circt::om::ClassOp>(
        builderOM.getUnknownLoc(),
        StringAttr::get(builderOM.getContext(),
                        "SitestBlackBoxMetadata_" + className),
        builderOM.getArrayAttr({}));
    metadataClass.getRegion().push_back(new Block());
    return metadataClass;
  };
  auto dutMetadataClass = buildOMClass(dutFilename);
  auto testMetadataClass = buildOMClass(testFilename);

  auto unknownLoc = builderOM.getUnknownLoc();
  unsigned index = 0;
  // This is a helper to create the verbatim output operation.
  auto createOutput = [&](SmallVectorImpl<StringRef> &names,
                          StringRef filename) {
    if (filename.empty())
      return;

    // Sort and remove duplicates.
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());

    // The output is a json array with each element a module name. The
    // defname of a module can't change so we can output them verbatim.
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    llvm::json::OStream j(os, 2);
    j.array([&] {
      for (auto &name : names) {
        j.value(name);
        auto modEntry = builderOM.create<om::ConstantOp>(
            unknownLoc, om::StringOMType::get(context),
            om::StringOMAttr::get(context, StringAttr::get(context, name)));
        auto object = builderOM.create<om::ObjectOp>(
            unknownLoc,
            om::ClassType::get(context, FlatSymbolRefAttr::get(sitestBBClass)),
            sitestBBClass.getNameAttr(), ValueRange({modEntry}));
        builderOM.create<om::ClassFieldOp>(
            unknownLoc, StringAttr::get(context, "m" + Twine(index++)), object);
      }
    });

    auto *body = circuitOp.getBodyBlock();
    // Put the information in a verbatim operation.
    auto builder = OpBuilder::atBlockEnd(body);
    auto verbatimOp =
        builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), buffer);
    auto fileAttr = hw::OutputFileAttr::getFromFilename(
        context, filename, /*excludeFromFilelist=*/true);
    verbatimOp->setAttr("output_file", fileAttr);
  };

  builderOM.setInsertionPointToEnd(testMetadataClass.getBodyBlock());
  createOutput(testModules, testFilename);
  builderOM.setInsertionPointToEnd(dutMetadataClass.getBodyBlock());
  createOutput(dutModules, dutFilename);

  // Clean up all ScalaClassAnnotations, which are no longer needed.
  for (auto op : circuitOp.getOps<FModuleLike>())
    AnnotationSet::removeAnnotations(op, scalaClassAnnoClass);

  return success();
}

void CreateSiFiveMetadataPass::getDependentDialects(
    mlir::DialectRegistry &registry) const {
  // We need this for SV verbatim and HW attributes.
  registry.insert<hw::HWDialect, sv::SVDialect, om::OMDialect>();
}

void CreateSiFiveMetadataPass::runOnOperation() {

  auto moduleOp = getOperation();
  auto circuits = moduleOp.getOps<CircuitOp>();
  if (circuits.empty())
    return;
  auto cIter = circuits.begin();
  circuitOp = *cIter++;

  assert(cIter == circuits.end() &&
         "cannot handle more than one CircuitOp in a mlir::ModuleOp");

  auto *body = circuitOp.getBodyBlock();

  // Find the device under test and create a set of all modules underneath it.
  auto it = llvm::find_if(*body, [&](Operation &op) -> bool {
    return AnnotationSet(&op).hasAnnotation(dutAnnoClass);
  });
  if (it != body->end()) {
    dutMod = dyn_cast<FModuleOp>(*it);
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    auto *node = instanceGraph.lookup(cast<hw::HWModuleLike>(*it));
    llvm::for_each(llvm::depth_first(node), [&](hw::InstanceGraphNode *node) {
      dutModuleSet.insert(node->getModule());
    });
  }

  if (failed(emitRetimeModulesMetadata()) ||
      failed(emitSitestBlackboxMetadata()) || failed(emitMemoryMetadata()))
    return signalPassFailure();

  // This pass does not modify the hierarchy.
  markAnalysesPreserved<InstanceGraph>();

  // Clear pass-global state as required by MLIR pass infrastructure.
  dutMod = {};
  circuitOp = {};
  dutModuleSet.empty();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createCreateSiFiveMetadataPass(
    bool replSeqMem, StringRef replSeqMemCircuit, StringRef replSeqMemFile) {
  return std::make_unique<CreateSiFiveMetadataPass>(
      replSeqMem, replSeqMemCircuit, replSeqMemFile);
}
