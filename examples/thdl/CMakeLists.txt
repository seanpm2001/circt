add_custom_target(thdl)

set_target_properties(thdl
  PROPERTIES
    FOLDER Examples
)

macro(add_thdl_chapter name)
  add_dependencies(thdl ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_thdl_chapter name)

add_subdirectory(Ch1)
