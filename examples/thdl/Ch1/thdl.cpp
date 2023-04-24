//===- thdl.cpp - The THDL Compiler  --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the THDL compiler.
//
//===----------------------------------------------------------------------===//

#include "thdl/Parser.h"

#include "circt/Support/Version.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/Timing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace thdl;
using namespace llvm;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input thdl file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum class InputType { THDL, MLIR };
} // namespace

static cl::opt<InputType>
    inputType("x", cl::init(InputType::THDL),
              cl::desc("Decided the kind of output desired"),
              cl::values(clEnumValN(InputType::THDL, "toy",
                                    "load the input file as a Toy source.")),
              cl::values(clEnumValN(InputType::MLIR, "mlir",
                                    "load the input file as an MLIR file")));

namespace {
enum Action { None, DumpAST };
} // namespace

static cl::opt<enum Action>
    emitAction("emit", cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")));

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
// std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
//   llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
//       llvm::MemoryBuffer::getFileOrSTDIN(filename);
//   if (std::error_code ec = fileOrErr.getError()) {
//     llvm::errs() << "Could not open input file: " << ec.message() << "\n";
//     return nullptr;
//   }
//   auto buffer = fileOrErr.get()->getBuffer();
//   LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
//   Parser parser(lexer);
//   return parser.parseModule();
// }
//
int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a Toy AST when the input is MLIR\n";
    return 5;
  }

  // auto moduleAST = parseInputFile(inputFilename);
  // if (!moduleAST)
  // return 1;

  // dump(*moduleAST);
  return 0;
}

// Create the timing manager we use to sample execution times.
// DefaultTimingManager tm;
// applyDefaultTimingManagerCLOptions(tm);
// auto ts = tm.getRootScope();

// Set up the input file.
// std::string errorMessage;
// auto input = openInputFile(inputFilename, &errorMessage);
// if (!input) {
//   llvm::errs() << errorMessage << "\n";
//   return failure();

static void registerPasses() {}

static LogicalResult execute(MLIRContext *context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  switch (action) {
  case Action::DumpAST:
    return dumpAST;
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    return failure();
  }
  if (emitAction == Action::DumpAST)
    return dumpAST();
}

/// Main driver for the thdl executable.  This sets up LLVM and MLIR, and parses
/// command line options before passing off to 'execute'.  This is set up
/// so we can `exit(0)` at the end of the program to avoid teardown of the
/// MLIRContext and modules inside of it (reducing compile time).
int main(int argc, char **argv) {
  InitLLVM init(argc, argv);

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  setBugReportMsg(circtBugReportMsg);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

  // Register passes before parsing command-line options, so that they are
  // available for use with options like `--mlir-print-ir-before`.
  registerPasses();

  MLIRContext context;
  exit(failed(execute(context));

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "thdl compiler\n");
}
