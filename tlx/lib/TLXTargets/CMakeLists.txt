if( NOT LLVM_REQUIRES_RTTI )
  if( NOT LLVM_REQUIRES_EH )
    set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/LegalizerX86.exports)
  endif()
endif()

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS Core Support)
endif()

add_llvm_library( LLVMTensor MODULE BUILDTREE_ONLY
  LegalizerX86.cpp

  DEPENDS
  intrinsics_gen
  PLUGIN_TOOL
  opt
  )
