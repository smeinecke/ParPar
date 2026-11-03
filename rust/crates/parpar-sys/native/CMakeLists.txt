cmake_minimum_required(VERSION 3.16)
project(parpar_native C CXX)

# Native ParPar compute backend for the Rust FFI shim.
# Sources live at the repository root (shared with the reference Node addon);
# this project only supplies build rules + the shim translation unit.

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(GF16_DIR "${REPO_ROOT}/gf16")
set(HASHER_DIR "${REPO_ROOT}/hasher")
set(SRC_DIR "${REPO_ROOT}/src")
set(SHIM_DIR "${CMAKE_CURRENT_LIST_DIR}/../shim")

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

option(PARPAR_NATIVE_OPENCL "Build with OpenCL backend support" ON)
option(PARPAR_NATIVE_SANITIZE "Enable ASan+UBSan" OFF)
option(PARPAR_NATIVE_TUNING "Enable -march=native style tuning" OFF)

# binding.gyp target_defaults applies these defines to every target, not just
# the addon: kernel sources emit OpenCL checksum helpers (gf16_cksum_copy_*,
# gf16_grp2_finish_*) only under PARPAR_OPENCL_SUPPORT.
add_compile_definitions(PARPAR_ENABLE_HASHER_MULTIMD5)
if(PARPAR_NATIVE_OPENCL)
	add_compile_definitions(PARPAR_OPENCL_SUPPORT)
endif()

# --- architecture detection (mirrors test/common.cmake) ---
set(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
if(${TARGET_ARCH} MATCHES "i386|i686|x86|x86_64|x64|amd64|AMD64")
	set(IS_X86 TRUE)
	if(${TARGET_ARCH} MATCHES "x86_64|x64|amd64|AMD64")
		set(IS_X64 TRUE)
	endif()
endif()
if(${TARGET_ARCH} MATCHES "arm|ARM|aarch64|arm64|ARM64|armeb|aarch64be|aarch64_be")
	set(IS_ARM TRUE)
	if(${TARGET_ARCH} MATCHES "arm64|ARM64|aarch64|armeb|aarch64be|aarch64_be")
		set(IS_ARM64 TRUE)
	endif()
endif()
if(${TARGET_ARCH} MATCHES "riscv64|rv64")
	set(IS_RISCV64 TRUE)
endif()
if(${TARGET_ARCH} MATCHES "riscv32|rv32")
	set(IS_RISCV32 TRUE)
endif()
message(STATUS "parpar_native: building for ${TARGET_ARCH}")

# --- libuv (required: controllers are built in USE_LIBUV mode like the addon) ---
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
	pkg_check_modules(UV QUIET libuv)
endif()
if(NOT UV_FOUND)
	find_path(UV_INCLUDE_DIR uv.h)
	find_library(UV_LIBRARY NAMES uv libuv)
	if(NOT UV_INCLUDE_DIR OR NOT UV_LIBRARY)
		message(FATAL_ERROR "libuv not found: install libuv-dev/libuv-devel or set UV_INCLUDE_DIR/UV_LIBRARY")
	endif()
endif()
if(UV_FOUND)
	set(UV_INCLUDES ${UV_INCLUDE_DIRS})
	set(UV_LIBS ${UV_LINK_LIBRARIES})
	if(NOT UV_LIBS)
		set(UV_LIBS ${UV_LIBRARIES})
	endif()
else()
	set(UV_INCLUDES ${UV_INCLUDE_DIR})
	set(UV_LIBS ${UV_LIBRARY})
endif()
message(STATUS "parpar_native: libuv include='${UV_INCLUDES}' lib='${UV_LIBS}'")

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

# Probe a flag set once per configure, mirroring the supports_* checks in
# binding.gyp: if the compiler rejects the flags, the sources compile
# without them and emit available_* = 0 stubs instead.
function(probe_flags out_var)
	string(MD5 flaghash "${ARGN}")
	set(check_var "PARPAR_FLAG_OK_${flaghash}")
	string(REPLACE ";" " " flagstr "${ARGN}")
	check_c_compiler_flag("${flagstr}" ${check_var})
	if(${check_var})
		set(${out_var} "${ARGN}" PARENT_SCOPE)
	else()
		set(${out_var} "" PARENT_SCOPE)
	endif()
endfunction()

# Compile one OBJECT library from a list of repo-relative sources with an
# optional flag list, mirroring the per-ISA static targets in binding.gyp.
# Usage: add_isa_lib(<name> "<flag;list>" <src...>)
function(add_isa_lib name flags)
	set(srcs "")
	foreach(f ${ARGN})
		list(APPEND srcs ${REPO_ROOT}/${f})
	endforeach()
	add_library(${name} OBJECT ${srcs})
	target_include_directories(${name} PRIVATE ${REPO_ROOT} ${GF16_DIR} ${GF16_DIR}/opencl-include)
	target_compile_definitions(${name} PRIVATE NDEBUG PARPAR_ENABLE_HASHER_MULTIMD5)
	# drop GCC-style flags under MSVC (callers pass both /arch and -m variants)
	if(MSVC AND flags MATCHES "^-")
		set(flags "")
	endif()
	if(flags)
		set_source_files_properties(${srcs} PROPERTIES COMPILE_OPTIONS "${flags}")
	endif()
	set_property(GLOBAL APPEND PROPERTY PARPAR_ISA_OBJECTS $<TARGET_OBJECTS:${name}>)
endfunction()

# --- controller/proc objects (USE_LIBUV selects the production callback path) ---
add_library(parpar_proc OBJECT
	${GF16_DIR}/controller.cpp
	${GF16_DIR}/controller_cpu.cpp
	${GF16_DIR}/gf16mul.cpp
	${GF16_DIR}/gfmat_coeff.c
	${SHIM_DIR}/parpar_shim.cc
)
target_include_directories(parpar_proc PRIVATE ${REPO_ROOT} ${GF16_DIR} ${GF16_DIR}/opencl-include ${HASHER_DIR} ${UV_INCLUDES} ${SHIM_DIR})
target_compile_definitions(parpar_proc PRIVATE USE_LIBUV PARPAR_ENABLE_HASHER_MULTIMD5 NDEBUG)
if(NOT MSVC)
	target_compile_options(parpar_proc PRIVATE -fexceptions)
	set_source_files_properties(${GF16_DIR}/gfmat_coeff.c PROPERTIES COMPILE_OPTIONS -Wno-unused-function)
endif()
set_property(GLOBAL APPEND PROPERTY PARPAR_ISA_OBJECTS $<TARGET_OBJECTS:parpar_proc>)

if(PARPAR_NATIVE_OPENCL)
	target_sources(parpar_proc PRIVATE ${GF16_DIR}/controller_ocl.cpp ${GF16_DIR}/controller_ocl_init.cpp ${GF16_DIR}/opencl-include/cl.c)
	target_compile_definitions(parpar_proc PRIVATE PARPAR_OPENCL_SUPPORT PARPAR_LIBDL_SUPPORT)
endif()

# --- generic C kernels + hasher C helpers ---
add_isa_lib(parpar_c ""
	gf16/gf16_lookup.c
	gf16/gf_add_generic.c
	gf16/gf16_cksum_generic.c
	hasher/crc_zeropad.c
	hasher/md5-final.c
)

# --- hasher core (C++) ---
if(MSVC)
	set(HASHER_EXC_FLAGS "")
else()
	set(HASHER_EXC_FLAGS "-fexceptions")
endif()
add_isa_lib(parpar_hasher "${HASHER_EXC_FLAGS}"
	hasher/hasher.cpp
	hasher/hasher_input.cpp
	hasher/hasher_md5mb.cpp
	hasher/hasher_scalar.cpp
	hasher/tables.cpp
)

# --- per-ISA sources (flags mirror binding.gyp target settings) ---
if(MSVC)
	if(IS_X86)
		add_isa_lib(isa_sse2 "" gf16/gf16_xor_sse2.c gf16/gf16_lookup_sse2.c gf16/gf_add_sse2.c gf16/gf16_cksum_sse2.c hasher/hasher_sse.cpp)
		add_isa_lib(isa_ssse3 "" gf16/gf16_shuffle_ssse3.c)
		add_isa_lib(isa_avx "" gf16/gf16_shuffle_avx.c)
		add_isa_lib(isa_avx2 "/arch:AVX2" gf16/gf16_xor_avx2.c gf16/gf16_shuffle_avx2.c gf16/gf_add_avx2.c gf16/gf16_cksum_avx2.c hasher/hasher_avx2.cpp)
		add_isa_lib(isa_avx512 "/arch:AVX512" gf16/gf16_xor_avx512.c gf16/gf16_shuffle_avx512.c gf16/gf_add_avx512.c gf16/gf16_cksum_avx512.c hasher/hasher_avx512.cpp hasher/hasher_avx512vl.cpp)
		add_isa_lib(isa_vbmi "/arch:AVX512" gf16/gf16_shuffle_vbmi.c)
		add_isa_lib(isa_gfni "" gf16/gf16_affine_gfni.c)
		add_isa_lib(isa_gfni_avx2 "/arch:AVX2" gf16/gf16_affine_avx2.c src/platform_warnings.c)
		add_isa_lib(isa_gfni_avx512 "/arch:AVX512" gf16/gf16_affine_avx512.c)
		add_isa_lib(isa_bmm "/arch:AVX512" gf16/gf16_affine_bmm.c)
		add_isa_lib(isa_clmul "" hasher/hasher_clmul.cpp)
		add_isa_lib(isa_xopbmi "/arch:AVX" hasher/hasher_xop.cpp hasher/hasher_bmi1.cpp)
	endif()
	# binding.gyp compiles every kernel source on every target: non-x86 files
	# self-guard and emit available_* = 0 stubs, so build them here too.
	add_isa_lib(isa_neon "" gf16/gf16_shuffle_neon.c gf16/gf16_clmul_neon.c gf16/gf_add_neon.c gf16/gf16_cksum_neon.c hasher/hasher_neon.cpp)
	add_isa_lib(isa_armcrc "" hasher/hasher_armcrc.cpp hasher/hasher_neoncrc.cpp)
	add_isa_lib(isa_sha3 "" gf16/gf16_clmul_sha3.c)
	add_isa_lib(isa_sve "" gf16/gf16_shuffle128_sve.c gf16/gf_add_sve.c gf16/gf16_cksum_sve.c)
	add_isa_lib(isa_sve2 "" gf16/gf16_shuffle128_sve2.c gf16/gf16_shuffle2x128_sve2.c gf16/gf16_shuffle512_sve2.c gf16/gf16_clmul_sve2.c gf16/gf_add_sve2.c hasher/hasher_sve2.cpp)
	add_isa_lib(isa_rvv "" gf16/gf16_shuffle128_rvv.c gf16/gf_add_rvv.c gf16/gf16_cksum_rvv.c)
	add_isa_lib(isa_rvv_zvbc "" gf16/gf16_clmul_rvv.c)
	add_isa_lib(isa_rvzbc "" hasher/hasher_rvzbc.cpp)
else()
	# binding.gyp compiles every kernel source on every target: files
	# self-guard on arch macros and emit available_* = 0 stubs when the ISA
	# flags are absent. Flags are probed (supports_* equivalents) so unknown
	# flags degrade to stubs rather than breaking the build.
	if(IS_X86)
		probe_flags(F_SSE2 -msse2)
		probe_flags(F_SSSE3 -mssse3)
		probe_flags(F_AVX -mavx)
		probe_flags(F_AVX2 -mavx2)
		probe_flags(F_AVX512 -mavx512vl;-mavx512bw)
		probe_flags(F_VBMI -mavx512vl;-mavx512vbmi)
		probe_flags(F_GFNI -mgfni;-mssse3)
		probe_flags(F_GFNI_AVX2 -mgfni;-mavx2)
		probe_flags(F_GFNI_AVX512 -mgfni;-mavx512vl;-mavx512bw)
		probe_flags(F_BMM -mavx512vl;-mavx512bmm)
		probe_flags(F_CLMUL -mpclmul;-msse4.1)
		probe_flags(F_XOP -mxop;-mavx)
		probe_flags(F_BMI1 -mpclmul;-mavx;-mbmi)
		probe_flags(F_HAVX512 -mavx512f)
		probe_flags(F_HAVX512VL -mavx512vl;-mavx512bw;-mbmi2;-mpclmul)
	endif()
	if(IS_ARM64)
		probe_flags(F_ARMCRC -march=armv8-a+crc)
		probe_flags(F_SHA3 -march=armv8.2-a+sha3)
		probe_flags(F_SVE -march=armv8-a+sve)
		probe_flags(F_SVE2 -march=armv8-a+sve2)
	elseif(IS_ARM)
		probe_flags(F_NEON -mfpu=neon;-march=armv7-a)
		probe_flags(F_ARMCRC -mfpu=neon;-march=armv8-a+crc)
		probe_flags(F_SHA3 -march=armv8.2-a+sha3)
	endif()
	if(IS_RISCV64)
		probe_flags(F_RVV -march=rv64gcv)
		probe_flags(F_RVV_ZVBC -march=rv64gcv_zvbc1)
		probe_flags(F_RVZBC -march=rv64gc_zbkc)
	elseif(IS_RISCV32)
		probe_flags(F_RVV -march=rv32gcv)
		probe_flags(F_RVV_ZVBC -march=rv32gcv_zvbc1)
		probe_flags(F_RVZBC -march=rv32gc_zbkc)
	endif()

	# x86 sources (compiled on every arch; stub out elsewhere)
	add_isa_lib(isa_sse2 "${F_SSE2}" gf16/gf16_xor_sse2.c gf16/gf16_lookup_sse2.c gf16/gf_add_sse2.c gf16/gf16_cksum_sse2.c hasher/hasher_sse.cpp)
	add_isa_lib(isa_ssse3 "${F_SSSE3}" gf16/gf16_shuffle_ssse3.c)
	add_isa_lib(isa_avx "${F_AVX}" gf16/gf16_shuffle_avx.c)
	add_isa_lib(isa_avx2 "${F_AVX2}" gf16/gf16_xor_avx2.c gf16/gf16_shuffle_avx2.c gf16/gf_add_avx2.c gf16/gf16_cksum_avx2.c hasher/hasher_avx2.cpp)
	add_isa_lib(isa_avx512 "${F_AVX512}" gf16/gf16_xor_avx512.c gf16/gf16_shuffle_avx512.c gf16/gf_add_avx512.c gf16/gf16_cksum_avx512.c)
	add_isa_lib(isa_vbmi "${F_VBMI}" gf16/gf16_shuffle_vbmi.c)
	add_isa_lib(isa_gfni "${F_GFNI}" gf16/gf16_affine_gfni.c)
	add_isa_lib(isa_gfni_avx2 "${F_GFNI_AVX2}" gf16/gf16_affine_avx2.c src/platform_warnings.c)
	add_isa_lib(isa_gfni_avx512 "${F_GFNI_AVX512}" gf16/gf16_affine_avx512.c)
	add_isa_lib(isa_bmm "${F_BMM}" gf16/gf16_affine_bmm.c)
	add_isa_lib(isa_clmul "${F_CLMUL}" hasher/hasher_clmul.cpp)
	add_isa_lib(isa_xop "${F_XOP}" hasher/hasher_xop.cpp)
	add_isa_lib(isa_bmi1 "${F_BMI1}" hasher/hasher_bmi1.cpp)
	add_isa_lib(isa_havx512 "${F_HAVX512}" hasher/hasher_avx512.cpp)
	add_isa_lib(isa_havx512vl "${F_HAVX512VL}" hasher/hasher_avx512vl.cpp)

	# ARM sources (stub out on non-ARM)
	add_isa_lib(isa_neon "${F_NEON}" gf16/gf16_shuffle_neon.c gf16/gf16_clmul_neon.c gf16/gf_add_neon.c gf16/gf16_cksum_neon.c hasher/hasher_neon.cpp)
	add_isa_lib(isa_armcrc "${F_ARMCRC}" hasher/hasher_armcrc.cpp hasher/hasher_neoncrc.cpp)
	add_isa_lib(isa_sha3 "${F_SHA3}" gf16/gf16_clmul_sha3.c)
	add_isa_lib(isa_sve "${F_SVE}" gf16/gf16_shuffle128_sve.c gf16/gf_add_sve.c gf16/gf16_cksum_sve.c)
	add_isa_lib(isa_sve2 "${F_SVE2}" gf16/gf16_shuffle128_sve2.c gf16/gf16_shuffle2x128_sve2.c gf16/gf16_shuffle512_sve2.c gf16/gf16_clmul_sve2.c gf16/gf_add_sve2.c hasher/hasher_sve2.cpp)

	# RISC-V sources (stub out elsewhere)
	add_isa_lib(isa_rvv "${F_RVV}" gf16/gf16_shuffle128_rvv.c gf16/gf_add_rvv.c gf16/gf16_cksum_rvv.c)
	add_isa_lib(isa_rvv_zvbc "${F_RVV_ZVBC}" gf16/gf16_clmul_rvv.c)
	add_isa_lib(isa_rvzbc "${F_RVZBC}" hasher/hasher_rvzbc.cpp)
endif()

# --- final static archive: merge every object library ---
get_property(ALL_OBJECTS GLOBAL PROPERTY PARPAR_ISA_OBJECTS)
add_library(parpar_native STATIC ${ALL_OBJECTS})
set_target_properties(parpar_native PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(parpar_native PUBLIC ${SHIM_DIR})
target_link_libraries(parpar_native PUBLIC ${UV_LIBS})
if(NOT WIN32 AND PARPAR_NATIVE_OPENCL)
	target_link_libraries(parpar_native PUBLIC ${CMAKE_DL_LIBS})
endif()
if(NOT WIN32)
	target_link_libraries(parpar_native PUBLIC pthread)
endif()

if(PARPAR_NATIVE_SANITIZE)
	add_compile_options(-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
	add_link_options(-fsanitize=address -fsanitize=undefined)
	target_compile_definitions(parpar_proc PRIVATE HAS_UBSAN)
endif()

if(PARPAR_NATIVE_TUNING AND NOT MSVC AND IS_X86)
	CHECK_C_COMPILER_FLAG("-march=native" SUPPORTS_MARCH_NATIVE)
	if(SUPPORTS_MARCH_NATIVE)
		add_compile_options(-march=native)
	endif()
endif()
