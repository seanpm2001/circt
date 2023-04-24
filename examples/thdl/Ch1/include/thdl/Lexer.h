//===- Lexer.h - Lexer for the THDL language ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a lexer for .thdl files.
//
//===----------------------------------------------------------------------===//

#ifndef THDL_LEXER_H
#define THDL_LEXER_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SourceMgr.h"

#include <memory>
#include <string>
#include <string_view>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace thdl {

//===----------------------------------------------------------------------===//
// Tokens
//===----------------------------------------------------------------------===//

enum class TokenKind {
  semicolon, // ;
  colon,     // :
  lparen,    // (
  rparen,    // )
  lbrace,    // {
  rbrace,    // }
  lbrack,    // [
  rbrack,    // ]
  equal,     // =
  period,    // .
  mod,       // mod
  id,        // foo
  num,       // 123
  eof,       // end of file
  error,
};

struct Token {
  llvm::SMLoc getLoc() const;
  llvm::SMLoc getEndLoc() const;
  llvm::SMRange getLocRange() const;

  constexpr Token(TokenKind kind, llvm::StringRef spelling)
      : kind(kind), spelling(spelling) {}

  constexpr Token() = default;

  TokenKind kind = TokenKind::error;
  llvm::StringRef spelling;
};

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

/// Sentinel type representing the "end of token stream".
/// `lexer == LexerEnd()` will return true if the lexer is at the end of the
/// file.
struct LexerEnd {};

class Lexer {
public:
  Lexer(const llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context);

  /// Look at the current token in the stream.
  const Token &operator*() const { return token; }

  /// Move to the next token in the stream.
  Lexer &operator++() {
    token = lex();
    return *this;
  }

  /// Test if the lexer is at the end of it's input, or in an error state.
  bool operator==(LexerEnd) const {
    return token.kind == TokenKind::eof || token.kind == TokenKind::error;
  }

  /// Get the beginning of the token stream, which is the lexer itself.
  Lexer &begin() { return *this; }

  /// Get a sentinel object that can be used to test if the lexer is at the end
  /// of input.
  constexpr LexerEnd end() { return {}; }

  const llvm::SourceMgr &getSourceMgr() const { return sourceMgr; }

  /// Convert a source-manager location to an MLIR location.
  mlir::Location translateLocation(llvm::SMLoc loc);

private:
  Token lex();
  Token lexID(const char *start);
  Token emitError(const char *loc, const llvm::Twine &message);

  /// Helper to form a token that spans from the start address, to the current
  /// cursor location.
  Token formToken(TokenKind kind, const char *start) const;

  /// The last token read from the input.
  Token token;

  /// The source manager, which tells us about the files loaded into the
  /// compiler.
  const llvm::SourceMgr &sourceMgr;
  mlir::MLIRContext *context;

  /// Pointer to the current character being lexed.
  const char *cursor = nullptr;
  /// Pointer to the end of the source buffer;
  const char *eof = nullptr;

  mlir::StringAttr mainFilename;
};

} // namespace thdl

#endif // THDL_LEXER_H
