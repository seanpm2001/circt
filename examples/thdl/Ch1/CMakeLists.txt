set(LLVM_LINK_COMPONENTS
  Support
)

add_thdl_chapter(thdl-ch1
  thdl.cpp
  AST.cpp
  lexer.cpp
  parser.cpp
)

target_include_directories(thdl-ch1
  PRIVATE
    include/
    ${CMAKE_CURRENT_BINARY_DIR}/include/
)

target_link_libraries(thdl-ch1
  PRIVATE
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTransforms
)
