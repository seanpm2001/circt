#include "thdl/AST.h"

#include <llvm/Support/raw_ostream.h>

using namespace thdl;

namespace {

/// State of the AST dumper.
struct State {
  /// Indentation depth.
  unsigned depth = 0;
};

/// RAII
struct Indent {
  Indent(State &state) : state(state) { ++state.depth; }
  ~Indent() { --state.depth; }
  State &state;
};
} // anonymous namespace

void indent(State &state) {
  for (unsigned i = 0; i < state.depth; ++i)
    llvm::outs() << "  ";
}

void dump(State &state, const ast::Module *module) {
  llvm::outs() << "module" << "\n";
}

void dump(State &state, const ast::Root *root) {
  auto indent = Indent(state);
  llvm::outs() << "root:"
             << "\n";
  for (auto *module : root->modules)
    dump(state, module);
}

void dump(const ast::Root *root) {
  State state;
  dump(state, root);
}