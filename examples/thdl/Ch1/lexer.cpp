//===- lexer.cpp - .thdl file lexer implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the lexer for .thdl file parsing.
//
//===----------------------------------------------------------------------===//

#include "thdl/Lexer.h"

#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringExtras.h"

using namespace thdl;
using namespace llvm;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Token Locations
//===----------------------------------------------------------------------===//

SMLoc Token::getLoc() const { return SMLoc::getFromPointer(spelling.data()); }

SMLoc Token::getEndLoc() const {
  return SMLoc::getFromPointer(spelling.data() + spelling.size());
}

SMRange Token::getLocRange() const { return SMRange(getLoc(), getEndLoc()); }

//===----------------------------------------------------------------------===//
// Lexer Initialization
//===----------------------------------------------------------------------===//

/// Get the identifier of the main file from the source manager, and convert
/// it to an MLIR StringAttr, for use as MLIR locations.
static StringAttr getMainFilename(const llvm::SourceMgr &sourceMgr,
                                  MLIRContext *context) {
  const auto *buffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef name = buffer->getBufferIdentifier();
  if (name.empty())
    name = "<unknown>";
  return StringAttr::get(context, name);
}

Lexer::Lexer(const SourceMgr &sourceMgr, MLIRContext *context)
    : sourceMgr(sourceMgr), context(context),
      mainFilename(getMainFilename(sourceMgr, context)) {
  auto buffer =
      sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer();
  cursor = buffer.begin();
  eof = buffer.end();
  token = lex();
}

Token Lexer::formToken(TokenKind kind, const char *start) const {
  return Token{kind, StringRef(start, cursor - start)};
}

//===----------------------------------------------------------------------===//
// Locations and Error Reporting
//===----------------------------------------------------------------------===//

/// Convert a SourceMgr location to an MLIR location.
Location Lexer::translateLocation(SMLoc loc) {
  assert(loc.isValid());
  unsigned mainFileID = sourceMgr.getMainFileID();
  auto [line, column] = sourceMgr.getLineAndColumn(loc, mainFileID);
  return FileLineColLoc::get(mainFilename, line, column);
}

Token Lexer::emitError(const char *loc, const Twine &message) {
  mlir::emitError(translateLocation(SMLoc::getFromPointer(loc)), message);
  return formToken(TokenKind::error, loc);
}

//===----------------------------------------------------------------------===//
// Lexing Implementation
//===----------------------------------------------------------------------===//

Token Lexer::lexID(const char *start) {
  while (llvm::isAlpha(*cursor) || llvm::isDigit(*cursor) || *cursor == '_' ||
         *cursor == '$' || *cursor == '-')
    ++cursor;

  return formToken(TokenKind::id, start);
}

Token Lexer::lex() {
  while (true) {
    const char *start = cursor;
    switch (*cursor++) {
    // nul character
    case '\0':
      // This may either be a nul character in the source file or may be the
      // EOF marker that llvm::MemoryBuffer guarantees will be there.
      if (cursor - 1 == eof)
        return formToken(TokenKind::eof, start);
      continue;

    // whitespace
    case ' ':
    case '\t':
    case '\n':
    case '\r':
    case ',':
      continue;

    // punctuation
    case '(':
      return formToken(TokenKind::lparen, start);
    case ')':
      return formToken(TokenKind::rparen, start);
    case '{':
      return formToken(TokenKind::lbrace, start);
    case '}':
      return formToken(TokenKind::rbrace, start);
    case '[':
      return formToken(TokenKind::lbrack, start);
    case ']':
      return formToken(TokenKind::rbrack, start);
    case '=':
      return formToken(TokenKind::equal, start);
    case '.':
      return formToken(TokenKind::period, start);

    // identifiers
    default:
      if (isalpha(cursor[-1]))
        return lexID(start);
      return emitError(start, "unexpected character");
    }
  }
}
