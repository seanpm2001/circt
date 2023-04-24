//===- AST.h - Node definition for the Toy AST ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an abstract syntax tree (AST) for .thdl files.
//
//===----------------------------------------------------------------------===//

#ifndef THDL_AST_H
#define THDL_AST_H

#include "thdl/Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <optional>
#include <utility>
#include <vector>

namespace thdl::ast {

enum class ExprKind {
  Apply,
  UInt,
  Num,
  Type,
  VarDecl,
  Return,
  Literal,
  Var,
  BinOp,
  Print,
  Reg,
  Ann,
};

class Expr {
public:
  // Expr(ExprKind kind) : kind(kind), location(location) {}

  ExprKind getKind() const { return kind; }

  // Location getLoc() const { return location; }

private:
  ExprKind kind;
};

enum class StmtKind {
  let,
  ret,
};

/// Statement base class and common functionality.
struct Stmt {
  Stmt(StmtKind kind) : kind(kind) {}

  StmtKind kind;
};

/// A binding statement with the form:
///   let symbol = value
struct LetStmt : Stmt {
  LetStmt(llvm::StringRef symbol, Expr *value)
      : Stmt(StmtKind::let), symbol(symbol), value(value) {}

  llvm::StringRef symbol;
  Expr *value;
};

/// A binding to the result of a module with the form:
///   ret value
struct RetStmt : Stmt {
  RetStmt(Expr *value) : Stmt(StmtKind::ret), value(value) {}
  Expr *value;
};

/// A module parameter.
struct Param {
  llvm::StringRef name;
  Expr *type;
};

/// A module definition.
///   mod foo(param1: type1, param2: type2): return_type {
///     let symbol = value
///     ret symbol
//    }
struct Module {
  llvm::ArrayRef<Param> params;
  Expr *type;
  llvm::ArrayRef<Stmt> body;
};

struct Root {
  llvm::ArrayRef<Module *> modules;
};

void dump(const Root &);

} // namespace thdl::ast

#endif // THDL_AST_H
