#ifndef THDL_PARSER_H
#define THDL_PARSER_H

#include "thdl/AST.h"
#include "thdl/Lexer.h"

namespace thdl {

class Parser {
public:
  Parser();
};

ast::Root parse(const llvm::SourceMgr &sourceMgr);

} // namespace thdl

#endif // THDL_PARSER_H
