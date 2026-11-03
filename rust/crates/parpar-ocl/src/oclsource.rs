//! OpenCL C kernel sources, extracted verbatim from gf16/controller_ocl_init.cpp.
//! Regenerate by re-extracting the _ocl_* constants if upstream changes them.

pub const _OCL_DEFINES: &str = r#"#define GF16_POLYNOMIAL 0x1100B
#define SHIFT_TOP_BIT(x) ((x) >> (sizeof(x)*8-1))
#if MAX_SLICE_SIZE < 16777216
 #define RBUF_REF(row, stride, col) mad24((uint)(row), (uint)(stride), (uint)(col))
#else
 #define RBUF_REF(row, stride, col) ((row) * (stride) + (col))
#endif
#if MAX_SLICE_SIZE < 16777216 && MAX_SLICE_SIZE*NUM_OUTPUTS <= 0xffffffffUL
 #define WBUF_REF(row, stride, col) mad24((uint)(row), (uint)(stride), (uint)(col))
#else
 #define WBUF_REF(row, stride, col) ((row) * (stride) + (col))
#endif
#define COBUF_REF(row, col) ((row) * SUBMIT_INPUTS + (col))
#if MAX_SLICE_SIZE*SUBMIT_INPUTS > 0xffffffffUL || MAX_SLICE_SIZE*NUM_OUTPUTS > 0xffffffffUL
 typedef size_t memsize_t;
#else
 typedef size_t memsize_t;
#endif
#if VECT_WIDTH == 1
 #define VECT_ONE 0x0001u
 #define SPREAD_POLY(v) (((short)(v) >> 15) & GF16_POLYNOMIAL)
 #define nat_ushort4 ushort4
 #define nat_uint ushort
 #define nat_int short
 #define NAT_BITS 16
#elif VECT_WIDTH == 2
 #define VECT_ONE 0x00010001ul
 #define SPREAD_POLY(v) mul24(((v) >> 15) & VECT_ONE, GF16_POLYNOMIAL & 0xffff)
 #define nat_ushort4 uint2
 #define nat_uint uint
 #define nat_int int
 #define NAT_BITS 32
#elif VECT_WIDTH == 4 || VECT_WIDTH == 8
 #define VECT_ONE (ulong)0x0001000100010001ull
 #define SPREAD_POLY(v) ((((v) >> 15) & VECT_ONE) * (GF16_POLYNOMIAL & 0xffff))
 #define nat_ushort4 ulong
 #define nat_uint ulong
 #define nat_int long
 #define NAT_BITS 64
#else
 #error Unsupported vector width
#endif
#if VECT_WIDTH == 8
 #define val_t ulong2
#else
 #define val_t nat_uint
#endif
#ifndef EX_TABLE_ARGS
 #define EX_TABLE_ARGS
 #define EX_TABLE_ARGS_DECL
#endif
#ifndef EX_TABLE_KARGS_DECL
 #define EX_TABLE_KARGS_DECL EX_TABLE_ARGS_DECL
#endif
#ifdef COPY_COEFF
 #define COEFF_TABLE_TYPE __local
#else
 #define COEFF_TABLE_TYPE __global
#endif
#define READ_SRC(src, idx) src[idx]
"#;

pub const _OCL_METHOD_BY2: &str = r#"#define GF_MULTIPLY gf16_multiply_by2
val_t gf16_multiply_by2(val_t a, nat_uint b) { nat_int btmp = b << (NAT_BITS - 16); val_t res = SHIFT_TOP_BIT(btmp) & a;
#pragma unroll
for(int i=0; i<15; i++) { val_t poly = SPREAD_POLY(res); res = ((res + res) & ~VECT_ONE) ^ poly; btmp <<= 1; res ^= SHIFT_TOP_BIT(btmp) & a; } return res; }"#;

pub const _OCL_METHOD_LH: &str = r#"#define LUT_REF(name, idx) ((__local ushort*)(name) + (idx)*512)
#define LUT_REF_X2(name, idx) ((__local uint*)(name) + (idx)*512)
#define LUT_DECLARATION(name, size) __local ushort name[512 * (size)] __attribute__ ((aligned (VECT_WIDTH*2)))
#define LUT_DECLARATION_X2(name, size) __local uint name[512 * (size)] __attribute__ ((aligned (VECT_WIDTH*2)))
#define LUT_DECLARATION_X2_1(name, src) __local ushort* name = (__local ushort*)(src)
#define LUT_GENERATE compute_lhtable
#define LUT_GENERATE_X2 compute_lhtable_x2
#define LUT_GENERATE_HAS_BARRIER
#define LUT_MULTIPLY lhtable_multiply
#define LUT_MULTIPLY_X2 lhtable_multiply_x2
ushort gf16_multiply_256(ushort v) { return (v << 8) ^ mul256poly[v >> 8]; } void gf16_multiply_write_lh(__local nat_uint* table, nat_uint val, nat_uint coeff, nat_ushort4 coeff256) { nat_int b = coeff << (NAT_BITS - 16);
#if VECT_WIDTH >= 2
nat_uint prod = 0;
#pragma unroll
for(uint i=1; i<256; i<<=1) prod ^= mul24((uint)(val & i), (uint)coeff); prod ^= mul256poly[prod >> 16]; prod <<= NAT_BITS - 16;
#else
nat_int a = val << (NAT_BITS - 8); nat_int prod = SHIFT_TOP_BIT(a) & b;
#pragma unroll
for(int i=0; i<7; i++) { nat_uint poly = SHIFT_TOP_BIT(prod) & ((nat_uint)(GF16_POLYNOMIAL & 0xffff) << (NAT_BITS-16)); prod = (prod << 1) ^ poly; a <<= 1; prod ^= SHIFT_TOP_BIT(a) & b; }
#endif
nat_uint b2 = (b << 1) ^ (SHIFT_TOP_BIT(b) & ((nat_uint)(GF16_POLYNOMIAL & 0xffff) << (NAT_BITS-16))); ushort prod256 = gf16_multiply_256((nat_uint)prod >> (NAT_BITS - 16));
#if VECT_WIDTH == 1
table[val/VECT_WIDTH] = prod; table[val/VECT_WIDTH + 1] = prod ^ b; table[val/VECT_WIDTH + 2] = prod ^ b2; table[val/VECT_WIDTH + 3] = prod ^ b2 ^ b; table[val/VECT_WIDTH + 256/VECT_WIDTH] = prod256; table[val/VECT_WIDTH + 256/VECT_WIDTH + 1] = prod256 ^ coeff256.s1; table[val/VECT_WIDTH + 256/VECT_WIDTH + 2] = prod256 ^ coeff256.s2; table[val/VECT_WIDTH + 256/VECT_WIDTH + 3] = prod256 ^ coeff256.s3;
#elif VECT_WIDTH == 2
table[val/VECT_WIDTH] = (prod ^ b) | ((nat_uint)prod >> 16); table[val/VECT_WIDTH + 1] = (prod ^ b ^ b2) | (((nat_uint)prod ^ b2) >> 16); table[val/VECT_WIDTH + 256/VECT_WIDTH] = upsample(prod256, prod256) ^ coeff256.s0; table[val/VECT_WIDTH + 256/VECT_WIDTH + 1] = upsample(prod256, prod256) ^ coeff256.s1;
#elif VECT_WIDTH == 4
nat_uint res = (prod ^ b2) | ((nat_uint)prod >> 32); table[val/VECT_WIDTH] = (res ^ b ^ ((nat_uint)b >> 32)) | (res >> 16); table[val/VECT_WIDTH + 256/VECT_WIDTH] = (prod256 * VECT_ONE) ^ coeff256;
#else
 #error Vector width not implemented
#endif
} nat_ushort4 lh_compute_coeff256(nat_uint coeff) { nat_ushort4 result; ushort coeff256 = gf16_multiply_256(coeff); ushort coeff512 = ((nat_uint)coeff256 << 1) ^ (-((nat_uint)coeff256 >> 15) & GF16_POLYNOMIAL);
#if VECT_WIDTH == 1
result.s1 = coeff256; result.s2 = coeff512; result.s3 = coeff256 ^ coeff512;
#elif VECT_WIDTH == 2
result.s0 = (uint)coeff256 << 16; result.s1 = upsample((ushort)(coeff256 ^ coeff512), coeff512);
#elif VECT_WIDTH == 4
result = upsample(upsample((ushort)(coeff256 ^ coeff512), coeff512), (uint)coeff256 << 16);
#else
 #error Vector width not implemented
#endif
return result; } void compute_lhtable(__local ushort* table, __global const ushort* restrict coeffs, const ushort numCoeff) { barrier(CLK_LOCAL_MEM_FENCE);
#ifdef WORKAROUND_ALIAS_BUG
if(((ulong)table & (VECT_WIDTH*2-1)) != 0) { table[0] = 0; } else
#endif
nat_uint localOffset = get_local_id(0)*4;
#if COL_GROUP_SIZE*4 > 256
for(nat_uint coeffBase=0; coeffBase<numCoeff; coeffBase+=(COL_GROUP_SIZE*4/256)) { const nat_uint coeffI = coeffBase + localOffset/256; if(coeffI < numCoeff) { const nat_uint coeff = coeffs[coeffI]; gf16_multiply_write_lh( (__local nat_uint*)(table + coeffI*512), localOffset & 0xff, coeff, lh_compute_coeff256(coeff) ); } }
#else
for(nat_uint coeffI=0; coeffI<numCoeff; coeffI++) { const nat_uint coeff = coeffs[coeffI]; const nat_ushort4 coeff256 = lh_compute_coeff256(coeff); __local nat_uint* nat_table = (__local nat_uint*)(table + coeffI*512);
#pragma unroll
for(nat_uint valI=0; valI<256; valI+=COL_GROUP_SIZE*4) { gf16_multiply_write_lh(nat_table, valI + localOffset, coeff, coeff256); } }
#endif
barrier(CLK_LOCAL_MEM_FENCE); } inline nat_uint lhtable_multiply(__local const ushort* table, nat_uint val) {
#if VECT_WIDTH==2
nat_uint val_h = (val>>8) | 0x1000100; return upsample( (ushort)(table[(val>>16) & 0xff] ^ table[(val_h>>16) & 0x1ff]), (ushort)(table[val & 0xff] ^ table[val_h & 0x1ff]) );
#else
nat_uint result = table[val & 0xff] ^ table[((val>>8) & 0xff) + 256];
#pragma unroll
for(nat_uint v=1; v<VECT_WIDTH; v++) { val >>= 16; result |= (nat_uint)(table[val & 0xff] ^ table[((val>>8) & 0xff) + 256]) << (v*16); } return result;
#endif
}
#if VECT_WIDTH==2 && defined(WRITE_GRP2)
inline uint lh_mul256_x2(uint val) { uint val256 = upsample( mul256poly[val >> 24], mul256poly[(val >> 8) & 0xff] ); return ((val & 0xff00ff) << 8) ^ val256; } void gf16_multiply_write_lh_x2(__local nat_uint* table, nat_uint val, uint coeffPair, uint coeff256Pair) { nat_int a = val << (NAT_BITS - 8); uint prod = SHIFT_TOP_BIT(a) & coeffPair;
#pragma unroll
for(int i=0; i<7; i++) { uint poly = SPREAD_POLY(prod); prod = ((prod + prod) & ~VECT_ONE) ^ poly; a <<= 1; prod ^= SHIFT_TOP_BIT(a) & coeffPair; } uint coeff2Pair = ((coeffPair + coeffPair) & ~VECT_ONE) ^ SPREAD_POLY(coeffPair); table[val] = prod; table[val+1] = prod ^ coeffPair; table[val+2] = prod ^ coeff2Pair; table[val+3] = prod ^ coeff2Pair ^ coeffPair; uint coeff512Pair = ((coeff256Pair + coeff256Pair) & ~VECT_ONE) ^ SPREAD_POLY(coeff256Pair); uint prod256 = lh_mul256_x2(prod); table[val + 256] = prod256; table[val + 257] = prod256 ^ coeff256Pair; table[val + 258] = prod256 ^ coeff512Pair; table[val + 259] = prod256 ^ coeff512Pair ^ coeff256Pair; } void compute_lhtable_x2(__local nat_uint* table, __global const ushort* restrict coeffs, const ushort numInputs) { barrier(CLK_LOCAL_MEM_FENCE); const nat_uint localOffset = get_local_id(0)*4;
#if COL_GROUP_SIZE*4 > 256
for(nat_uint coeffBase=0; coeffBase<numInputs; coeffBase+=(COL_GROUP_SIZE*4/256)) { const nat_uint coeffI = coeffBase + localOffset/256; if(coeffI < numInputs) { const ushort coeffA = coeffs[COBUF_REF(0, coeffI)]; const ushort coeffB = coeffs[COBUF_REF(1, coeffI)]; uint coeff = upsample(coeffB, coeffA); gf16_multiply_write_lh_x2( table + coeffI*512, localOffset & 0xff, coeff, lh_mul256_x2(coeff) ); } }
#else
for(nat_uint coeffI=0; coeffI<numInputs; coeffI++) { const ushort coeffA = coeffs[COBUF_REF(0, coeffI)]; const ushort coeffB = coeffs[COBUF_REF(1, coeffI)]; const uint coeffPair = upsample(coeffB, coeffA); const uint coeff256Pair = lh_mul256_x2(coeffPair); __local nat_uint* nat_table = table + coeffI*512;
#pragma unroll
for(nat_uint valI=0; valI<256; valI+=COL_GROUP_SIZE*4) { gf16_multiply_write_lh_x2(nat_table, valI + localOffset, coeffPair, coeff256Pair); } }
#endif
barrier(CLK_LOCAL_MEM_FENCE); } inline uint2 lhtable_multiply_x2(__local const nat_uint* table, uint val) { uint val_h = (val>>8) | 0x1000100; return (uint2)( table[val & 0xff] ^ table[val_h & 0x1ff], table[(val>>16) & 0xff] ^ table[val_h>>16] ); }
#endif
"#;

pub const _OCL_METHOD_LL: &str = r#"#define LUT_REF(name, idx) ((__local ushort*)(name) + (idx)*256)
#define LUT_DECLARATION(name, size) __local ushort name[256 * (size)] __attribute__ ((aligned (VECT_WIDTH*2)))
#define LUT_GENERATE compute_lltable
#define LUT_GENERATE_HAS_BARRIER
#define LUT_MULTIPLY lltable_multiply
ushort gf16_multiply_256(ushort v) { return (v << 8) ^ mul256poly[v >> 8]; } void gf16_multiply_write_ll(__local nat_uint* table, nat_uint val, nat_uint coeff) { nat_int b = coeff << (NAT_BITS - 16);
#if VECT_WIDTH >= 2
nat_uint prod = 0;
#pragma unroll
for(uint i=1; i<256; i<<=1) prod ^= mul24((uint)(val & i), (uint)coeff); prod ^= mul256poly[prod >> 16]; prod <<= NAT_BITS - 16;
#else
nat_int a = val << (NAT_BITS - 8); nat_int prod = SHIFT_TOP_BIT(a) & b;
#pragma unroll
for(int i=0; i<7; i++) { nat_uint poly = SHIFT_TOP_BIT(prod) & ((nat_uint)(GF16_POLYNOMIAL & 0xffff) << (NAT_BITS-16)); prod = (prod << 1) ^ poly; a <<= 1; prod ^= SHIFT_TOP_BIT(a) & b; }
#endif
nat_uint b2 = (b << 1) ^ (SHIFT_TOP_BIT(b) & ((nat_uint)(GF16_POLYNOMIAL & 0xffff) << (NAT_BITS-16)));
#if VECT_WIDTH == 1
table[val/VECT_WIDTH] = prod; table[val/VECT_WIDTH + 1] = prod ^ b; table[val/VECT_WIDTH + 2] = prod ^ b2; table[val/VECT_WIDTH + 3] = prod ^ b2 ^ b;
#elif VECT_WIDTH == 2
table[val/VECT_WIDTH] = (prod ^ b) | ((nat_uint)prod >> 16); table[val/VECT_WIDTH + 1] = (prod ^ b ^ b2) | (((nat_uint)prod ^ b2) >> 16);
#elif VECT_WIDTH == 4
nat_uint res = (prod ^ b2) | ((nat_uint)prod >> 32); table[val/VECT_WIDTH] = (res ^ b ^ ((nat_uint)b >> 32)) | (res >> 16);
#else
 #error Vector width not implemented
#endif
} void compute_lltable(__local ushort* table, __global const ushort* restrict coeffs, const ushort numCoeff) { barrier(CLK_LOCAL_MEM_FENCE);
#ifdef WORKAROUND_ALIAS_BUG
if(((ulong)table & (VECT_WIDTH*2-1)) != 0) { table[0] = 0; } else
#endif
nat_uint localOffset = get_local_id(0)*4;
#if COL_GROUP_SIZE*4 > 256
for(nat_uint coeffBase=0; coeffBase<numCoeff; coeffBase+=(COL_GROUP_SIZE*4/256)) { const nat_uint coeffI = coeffBase + localOffset/256; if(coeffI < numCoeff) { const nat_uint coeff = coeffs[coeffI]; gf16_multiply_write_ll( (__local nat_uint*)(table + coeffI*256), localOffset & 0xff, coeff ); } }
#else
for(nat_uint coeffI=0; coeffI<numCoeff; coeffI++) { const nat_uint coeff = coeffs[coeffI]; __local nat_uint* nat_table = (__local nat_uint*)(table + coeffI*256);
#pragma unroll
for(nat_uint valI=0; valI<256; valI+=COL_GROUP_SIZE*4) { gf16_multiply_write_ll(nat_table, valI + localOffset, coeff); } }
#endif
barrier(CLK_LOCAL_MEM_FENCE); } inline nat_uint lltable_multiply(__local const ushort* table, nat_uint val) {
#if VECT_WIDTH==2
return upsample( (ushort)(table[(val>>16) & 0xff] ^ gf16_multiply_256(table[(val>>24) & 0xff])), (ushort)(table[val & 0xff] ^ gf16_multiply_256(table[(val>>8) & 0xff])) );
#else
nat_uint result = table[val & 0xff] ^ gf16_multiply_256(table[(val>>8) & 0xff]);
#pragma unroll
for(nat_uint v=1; v<VECT_WIDTH; v++) { val >>= 16; result |= (nat_uint)(table[val & 0xff] ^ gf16_multiply_256(table[(val>>8) & 0xff])) << (v*16); } return result;
#endif
}"#;

pub const _OCL_METHOD_SHUFFLE: &str = r#"#undef READ_SRC
#define READ_SRC gf16_shuffle_read
#define WRITE_DST_OVERRIDE(dst, idx, val) gf16_shuffle_write(dst, idx, val)
#define WRITE_DST_ADD_OVERRIDE(dst, idx, val) gf16_shuffle_add(dst, idx, val)
#undef val_t
#if VECT_WIDTH==1
 #define val_t uchar4
 #define val2_t ushort2
 #define nat_ucharX uchar2
#elif VECT_WIDTH==2
 #define val_t uchar8
 #define val2_t ushort4
 #define nat_ucharX uchar4
#elif VECT_WIDTH==4
 #define val_t uchar16
 #define val2_t ushort8
 #define nat_ucharX uchar8
#else
 #error Unsupported vector width
#endif
#define LUT_REF(name, idx) ((__private uchar4*)(name) + (idx)*16)
#define LUT_DECLARATION(name, size) __private uchar4 name[16 * (size)]
#define LUT_GENERATE compute_shuffletable
#define LUT_MULTIPLY shuffle_multiply
ushort gf16_mult2(const ushort v) { return (v << 1) ^ (-(v >= 0x8000) & (GF16_POLYNOMIAL & 0xffff)); } void compute_shuffletable(__private uchar4* table, __global const ushort* restrict coeffs, const ushort numCoeff) { for(uint i=0; i<numCoeff; i++) { ushort val = coeffs[i]; ushort val2 = gf16_mult2(val); ushort val3 = val2 ^ val; __private uchar4* tbl = table + i*16; tbl[0] = (uchar4)(0, val&0xff, val2&0xff, val3&0xff); tbl[1] = (uchar4)(0, val>>8, val2>>8, val3>>8); uchar4 ri;
#if GF16_POLYNOMIAL != 0x1100b
#error Unsupported polynomial
#endif
#define MUL4(p, c) ri = tbl[p+1] >> (uchar)6;tbl[c+1] = ((tbl[p+1] << (uchar)2) | (tbl[p] >> (uchar)6)) ^ (ri << (uchar)4);tbl[c] = (tbl[p] << (uchar)2) ^ shuffle((uchar4)((uchar)0, (uchar)0xb, (uchar)(0xb<<1), (uchar)((0xb<<1)^0xb)), ri)
MUL4(0, 2); MUL4(2, 4); MUL4(4, 6); MUL4(6, 8); MUL4(8, 10); MUL4(10, 12); MUL4(12, 14);
#undef MUL4
} } val_t gf16_shuffle_read(__global const val_t* restrict src, const size_t idx) { const val_t data = src[idx]; return (val_t)(data.even, data.odd); } void gf16_shuffle_write(__global val_t* dst, const memsize_t offs, const val_t val) { *(__global val2_t*)(dst + offs) = upsample(val.hi, val.lo); } void gf16_shuffle_add(__global val_t* dst, const memsize_t offs, const val_t val) { *(__global val2_t*)(dst + offs) ^= upsample(val.hi, val.lo); } inline val_t shuffle_multiply(__private const uchar4* table, val_t val) { nat_ucharX resLo = shuffle(table[0], val.lo) ^ shuffle(table[8], val.hi); nat_ucharX resHi = shuffle(table[1], val.lo) ^ shuffle(table[9], val.hi); resLo ^= shuffle(table[2], val.lo >>(uchar)2) ^ shuffle(table[10], val.hi >>(uchar)2); resHi ^= shuffle(table[3], val.lo >>(uchar)2) ^ shuffle(table[11], val.hi >>(uchar)2); resLo ^= shuffle(table[4], val.lo >>(uchar)4) ^ shuffle(table[12], val.hi >>(uchar)4); resHi ^= shuffle(table[5], val.lo >>(uchar)4) ^ shuffle(table[13], val.hi >>(uchar)4); resLo ^= shuffle(table[6], val.lo >>(uchar)6) ^ shuffle(table[14], val.hi >>(uchar)6); resHi ^= shuffle(table[7], val.lo >>(uchar)6) ^ shuffle(table[15], val.hi >>(uchar)6); return (val_t)(resLo, resHi); }"#;

pub const _OCL_METHOD_LOG: &str = r#"#define GF_MULTIPLY(a, b) gf16_multiply_log(a, b  EX_TABLE_ARGS)
#undef READ_SRC
#define READ_SRC(src, idx) gf16_log_src(src[idx]  EX_TABLE_ARGS)
#if defined(OCL_METHOD_LOG_TINY) || defined(OCL_METHOD_LOG_SMALL)
 #define LOG_SRC(n) gf16_log_small(n  EX_TABLE_ARGS)
#else
 #define LOG_SRC(n) ((LOG_MEM_TYPE ushort*)gf16_log)[n]
#endif
ushort gf16_log_small(nat_uint val EX_TABLE_ARGS_DECL) { if(val == 0) return 65535;
#ifdef OCL_METHOD_LOG_SMALL
 #if __OPENCL_C_VERSION__ >= 200
  nat_uint log = ctz(val);
 #else
  nat_uint log = clz((nat_uint)((val-1)^val)) ^ (NAT_BITS-1);
 #endif
val >>= log; log += ((LOG_MEM_TYPE ushort*)gf16_log)[val >> 1];
#else
nat_uint log = 0; uint prep; nat_uint shift; LOG_MEM_TYPE uint* gf16_log_prep = (gf16_log + 16384/2); prep = gf16_log_prep[val & 0x7ff]; shift = prep >> 16; log += shift; val = (val >> shift) ^ (prep & 0xffff); prep = gf16_log_prep[val & 0x7ff]; shift = prep >> 16; log += shift; val = (val >> shift) ^ (prep & 0xffff); ushort log2 = ((LOG_MEM_TYPE ushort*)gf16_log)[val >> 2];
#if VECT_WIDTH == 1
log += add_sat(log, log2) == (ushort)65535; log += log2;
#else
log += log2; if(log >= 65535) log -= 65535;
#endif
#endif
return log; } val_t gf16_log_src(val_t val EX_TABLE_ARGS_DECL) {
#if VECT_WIDTH==2
return upsample(LOG_SRC(val >> 16), LOG_SRC(val & 0xffff));
#else
val_t res = LOG_SRC(val & 0xffff);
#pragma unroll
for(int v=1; v<VECT_WIDTH; v++) { int shift = v*16; res |= (val_t)LOG_SRC((val>>shift) & 0xffff) << shift; } return res;
#endif
} nat_uint gfmat_calc_coeff_log(nat_uint recBlock, ushort inCoeff) { uint result = inCoeff * (ushort)recBlock; result = (result >> 16) + (result & 65535); result += result >> 16; return result & 65535; } val_t gf16_multiply_log(val_t a, nat_uint b EX_TABLE_ARGS_DECL) { val_t result = 0;
#pragma unroll
for(int v=0; v<VECT_WIDTH; v++) { nat_uint va = a & 65535; bool isZero = (va == 65535);
#if VECT_WIDTH == 1
va += add_sat(va, b) == (ushort)65535; va += b;
#else
va += b; va = (va >> 16) + (va & 0xffff);
#endif

#ifdef OCL_METHOD_EXP_SMALL
uint vatmp = gf16_antilog[va >> 3]; vatmp <<= va & 7; vatmp ^= gf16_antilog[8192 + (vatmp >> 16)]; va = vatmp & 0xffff;
#elif defined(OCL_METHOD_EXP_TINY)
uint vatmp = gf16_antilog[va >> 4]; vatmp <<= va & 15; vatmp ^= gf16_antilog[4096 + (vatmp >> 24)] << 8; vatmp &= 0xffffff; vatmp ^= gf16_antilog[4096 + (vatmp >> 16)]; va = vatmp & 0xffff;
#else
va = gf16_antilog[va];
#endif
result |= (isZero ? 0 : va) << (v*16); a >>= 16; } return result; }"#;

pub const _OCL_METHOD_SPLITMUL: &str = r#"#define GF_MULTIPLY gf16_multiply_splitmul
val_t gf16_multiply_splitmul(val_t a, nat_uint b) { val_t result = 0; ushort ba = (b>>12)*16; ushort bb = ((b>>8) & 0xf)*16; ushort bc = ((b>>4) & 0xf)*16; ushort bd = (b & 0xf)*16;
#pragma unroll
for(int v=0; v<VECT_WIDTH; v++) { ushort ax = (a >> 8) & 0xff; ushort ay = a & 0xff; ushort val = mul_split1[bd + ay] ^ gf16_mul16(mul_split1[bc + ay]) ^ mul_split256[bb + ay] ^ mul_split256[bd + ax] ^ gf16_mul16(mul_split256[ba + ay] ^ mul_split256[bc + ax]) ^ mul_split4107[bb + ax] ^ gf16_mul16(mul_split4107[ba + ax]) result |= val << (v*16); a >>= 16; } return result; }"#;

pub const _OCL_KERNEL_NOLUT_FUNCS: &str = r#"#ifdef WRITE_DST_OVERRIDE
 #ifdef MUL_ONLY
  #define WRITE_DST WRITE_DST_OVERRIDE
 #else
  #define WRITE_DST WRITE_DST_ADD_OVERRIDE
 #endif
#elif defined(MUL_ONLY)
 #define WRITE_DST(dst, idx, val) dst[idx] = val
#else
 #define WRITE_DST(dst, idx, val) dst[idx] ^= val
#endif
#ifdef OCL_METHOD_LOG
 #define GET_COEFF(o, i) gfmat_calc_coeff_log(outExp[o], coeff[i])
#else
 #define GET_COEFF(o, i) coeff[COBUF_REF(o, i)]
#endif
#define DO_CACHE 1
inline void KERNFN(mulgroup_cache)(__global val_t* restrict dst, __global const val_t* restrict src, COEFF_TABLE_TYPE const ushort* restrict coeff, const nat_uint outBlk, const nat_uint numOutputs, const ushort _numInputs, __local val_t* cache  EX_TABLE_ARGS_DECL) { 
	const memsize_t len = get_global_size(0)*COL_GROUP_ITERS;                                    
	const memsize_t globalCol = get_global_id(0)*COL_GROUP_ITERS;                                
	const uint col = get_local_id(0)*COL_GROUP_ITERS;                                            
	__global const val_t* srcBase = src + globalCol;                                             
	                                                                                             
	val_t result[OUTPUT_GROUPING][COL_GROUP_ITERS];                                              
	val_t val[COL_GROUP_ITERS];                                                                  
	#pragma unroll                                                                               
	for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++) {                                         
		#ifdef DO_CACHE                                                                          
			val[iter] = READ_SRC(srcBase, iter);                                                 
			if(DO_CACHE == 1) cache[col+iter] = val[iter];                                       
		#else                                                                                    
			val[iter] = cache[col+iter];                                                         
		#endif                                                                                   
	}                                                                                            
	                                                                                             
	nat_uint curCoeff;                                                                           
	#ifdef GFMAT_COEFF_SEQUENTIAL                                                                
		curCoeff = gfmat_calc_coeff_log(outExp+outBlk, coeff[0]);                                
		#pragma unroll                                                                           
		for (nat_uint o = 0; o < numOutputs; o++) {                                              
			#pragma unroll                                                                       
			for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++)                                   
				result[o][iter] = GF_MULTIPLY(val[iter], curCoeff);                              
			curCoeff += coeff[0];                                                                
			curCoeff += curCoeff >> 16;                                                          
			curCoeff &= 0xffff;                                                                  
		}                                                                                        
	#else                                                                                        
		#pragma unroll                                                                           
		for (nat_uint o = 0; o < numOutputs; o++) {                                              
			curCoeff = GET_COEFF(outBlk+o, 0u);                                                  
			#pragma unroll                                                                       
			for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++)                                   
				result[o][iter] = GF_MULTIPLY(val[iter], curCoeff);                              
		}                                                                                        
	#endif                                                                                       
	                                                                                             
	for (nat_uint i = 1; i < _numInputs; i++) {                                                  
		srcBase += len;                                                                          
		#pragma unroll                                                                           
		for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++) {                                     
			#ifdef DO_CACHE                                                                      
				val[iter] = READ_SRC(srcBase, iter);                                             
				if(DO_CACHE == 1) cache[i*COL_GROUP_SIZE*COL_GROUP_ITERS + col+iter] = val[iter];
			#else                                                                                
				val[iter] = cache[i*COL_GROUP_SIZE*COL_GROUP_ITERS + col+iter];                  
			#endif                                                                               
		}                                                                                        
		                                                                                         
		#ifdef GFMAT_COEFF_SEQUENTIAL                                                            
			curCoeff = gfmat_calc_coeff_log(outExp+outBlk, coeff[i]);                            
			#pragma unroll                                                                       
			for (nat_uint o = 0; o < numOutputs; o++) {                                          
				#pragma unroll                                                                   
				for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++)                               
					result[o][iter] ^= GF_MULTIPLY(val[iter], curCoeff);                         
				curCoeff += coeff[i];                                                            
				curCoeff += curCoeff >> 16;                                                      
				curCoeff &= 0xffff;                                                              
			}                                                                                    
		#else                                                                                    
			#pragma unroll                                                                       
			for (nat_uint o = 0; o < numOutputs; o++) {                                          
				curCoeff = GET_COEFF(outBlk+o, i);                                               
				#pragma unroll                                                                   
				for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++)                               
					result[o][iter] ^= GF_MULTIPLY(val[iter], curCoeff);                         
			}                                                                                    
		#endif                                                                                   
	}                                                                                            
	__global val_t* dstBase = dst + WBUF_REF(outBlk, len, globalCol);                            
	#pragma unroll                                                                               
	for (nat_uint o = 0; o < numOutputs; o++) {                                                  
		#pragma unroll                                                                           
		for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++) {                                     
			WRITE_DST(dstBase, iter, result[o][iter]);                                           
		}                                                                                        
		dstBase += len;                                                                          
	}                                                                                            
}                                                                                               

#undef DO_CACHE
#define DO_CACHE 0
inline void KERNFN(mulgroup_nocache)(__global val_t* restrict dst, __global const val_t* restrict src, COEFF_TABLE_TYPE const ushort* restrict coeff, const nat_uint outBlk, const nat_uint numOutputs, const ushort _numInputs, __local val_t* cache  EX_TABLE_ARGS_DECL) { 
	const memsize_t len = get_global_size(0)*COL_GROUP_ITERS;                                    
	const memsize_t globalCol = get_global_id(0)*COL_GROUP_ITERS;                                
	const uint col = get_local_id(0)*COL_GROUP_ITERS;                                            
	__global const val_t* srcBase = src + globalCol;                                             
	                                                                                             
	val_t result[OUTPUT_GROUPING][COL_GROUP_ITERS];                                              
	val_t val[COL_GROUP_ITERS];                                                                  
	#pragma unroll                                                                               
	for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++) {                                         
		#ifdef DO_CACHE                                                                          
			val[iter] = READ_SRC(srcBase, iter);                                                 
			if(DO_CACHE == 1) cache[col+iter] = val[iter];                                       
		#else                                                                                    
			val[iter] = cache[col+iter];                                                         
		#endif                                                                                   
	}                                                                                            
	                                                                                             
	nat_uint curCoeff;                                                                           
	#ifdef GFMAT_COEFF_SEQUENTIAL                                                                
		curCoeff = gfmat_calc_coeff_log(outExp+outBlk, coeff[0]);                                
		#pragma unroll                                                                           
		for (nat_uint o = 0; o < numOutputs; o++) {                                              
			#pragma unroll                                                                       
			for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++)                                   
				result[o][iter] = GF_MULTIPLY(val[iter], curCoeff);                              
			curCoeff += coeff[0];                                                                
			curCoeff += curCoeff >> 16;                                                          
			curCoeff &= 0xffff;                                                                  
		}                                                                                        
	#else                                                                                        
		#pragma unroll                                                                           
		for (nat_uint o = 0; o < numOutputs; o++) {                                              
			curCoeff = GET_COEFF(outBlk+o, 0u);                                                  
			#pragma unroll                                                                       
			for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++)                                   
				result[o][iter] = GF_MULTIPLY(val[iter], curCoeff);                              
		}                                                                                        
	#endif                                                                                       
	                                                                                             
	for (nat_uint i = 1; i < _numInputs; i++) {                                                  
		srcBase += len;                                                                          
		#pragma unroll                                                                           
		for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++) {                                     
			#ifdef DO_CACHE                                                                      
				val[iter] = READ_SRC(srcBase, iter);                                             
				if(DO_CACHE == 1) cache[i*COL_GROUP_SIZE*COL_GROUP_ITERS + col+iter] = val[iter];
			#else                                                                                
				val[iter] = cache[i*COL_GROUP_SIZE*COL_GROUP_ITERS + col+iter];                  
			#endif                                                                               
		}                                                                                        
		                                                                                         
		#ifdef GFMAT_COEFF_SEQUENTIAL                                                            
			curCoeff = gfmat_calc_coeff_log(outExp+outBlk, coeff[i]);                            
			#pragma unroll                                                                       
			for (nat_uint o = 0; o < numOutputs; o++) {                                          
				#pragma unroll                                                                   
				for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++)                               
					result[o][iter] ^= GF_MULTIPLY(val[iter], curCoeff);                         
				curCoeff += coeff[i];                                                            
				curCoeff += curCoeff >> 16;                                                      
				curCoeff &= 0xffff;                                                              
			}                                                                                    
		#else                                                                                    
			#pragma unroll                                                                       
			for (nat_uint o = 0; o < numOutputs; o++) {                                          
				curCoeff = GET_COEFF(outBlk+o, i);                                               
				#pragma unroll                                                                   
				for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++)                               
					result[o][iter] ^= GF_MULTIPLY(val[iter], curCoeff);                         
			}                                                                                    
		#endif                                                                                   
	}                                                                                            
	__global val_t* dstBase = dst + WBUF_REF(outBlk, len, globalCol);                            
	#pragma unroll                                                                               
	for (nat_uint o = 0; o < numOutputs; o++) {                                                  
		#pragma unroll                                                                           
		for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++) {                                     
			WRITE_DST(dstBase, iter, result[o][iter]);                                           
		}                                                                                        
		dstBase += len;                                                                          
	}                                                                                            
}                                                                                               

#undef DO_CACHE
inline void KERNFN(mulgroup_read)(__global val_t* restrict dst, __global const val_t* restrict src, COEFF_TABLE_TYPE const ushort* restrict coeff, const nat_uint outBlk, const nat_uint numOutputs, const ushort _numInputs, __local val_t* cache  EX_TABLE_ARGS_DECL) { 
	const memsize_t len = get_global_size(0)*COL_GROUP_ITERS;                                    
	const memsize_t globalCol = get_global_id(0)*COL_GROUP_ITERS;                                
	const uint col = get_local_id(0)*COL_GROUP_ITERS;                                            
	__global const val_t* srcBase = src + globalCol;                                             
	                                                                                             
	val_t result[OUTPUT_GROUPING][COL_GROUP_ITERS];                                              
	val_t val[COL_GROUP_ITERS];                                                                  
	#pragma unroll                                                                               
	for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++) {                                         
		#ifdef DO_CACHE                                                                          
			val[iter] = READ_SRC(srcBase, iter);                                                 
			if(DO_CACHE == 1) cache[col+iter] = val[iter];                                       
		#else                                                                                    
			val[iter] = cache[col+iter];                                                         
		#endif                                                                                   
	}                                                                                            
	                                                                                             
	nat_uint curCoeff;                                                                           
	#ifdef GFMAT_COEFF_SEQUENTIAL                                                                
		curCoeff = gfmat_calc_coeff_log(outExp+outBlk, coeff[0]);                                
		#pragma unroll                                                                           
		for (nat_uint o = 0; o < numOutputs; o++) {                                              
			#pragma unroll                                                                       
			for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++)                                   
				result[o][iter] = GF_MULTIPLY(val[iter], curCoeff);                              
			curCoeff += coeff[0];                                                                
			curCoeff += curCoeff >> 16;                                                          
			curCoeff &= 0xffff;                                                                  
		}                                                                                        
	#else                                                                                        
		#pragma unroll                                                                           
		for (nat_uint o = 0; o < numOutputs; o++) {                                              
			curCoeff = GET_COEFF(outBlk+o, 0u);                                                  
			#pragma unroll                                                                       
			for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++)                                   
				result[o][iter] = GF_MULTIPLY(val[iter], curCoeff);                              
		}                                                                                        
	#endif                                                                                       
	                                                                                             
	for (nat_uint i = 1; i < _numInputs; i++) {                                                  
		srcBase += len;                                                                          
		#pragma unroll                                                                           
		for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++) {                                     
			#ifdef DO_CACHE                                                                      
				val[iter] = READ_SRC(srcBase, iter);                                             
				if(DO_CACHE == 1) cache[i*COL_GROUP_SIZE*COL_GROUP_ITERS + col+iter] = val[iter];
			#else                                                                                
				val[iter] = cache[i*COL_GROUP_SIZE*COL_GROUP_ITERS + col+iter];                  
			#endif                                                                               
		}                                                                                        
		                                                                                         
		#ifdef GFMAT_COEFF_SEQUENTIAL                                                            
			curCoeff = gfmat_calc_coeff_log(outExp+outBlk, coeff[i]);                            
			#pragma unroll                                                                       
			for (nat_uint o = 0; o < numOutputs; o++) {                                          
				#pragma unroll                                                                   
				for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++)                               
					result[o][iter] ^= GF_MULTIPLY(val[iter], curCoeff);                         
				curCoeff += coeff[i];                                                            
				curCoeff += curCoeff >> 16;                                                      
				curCoeff &= 0xffff;                                                              
			}                                                                                    
		#else                                                                                    
			#pragma unroll                                                                       
			for (nat_uint o = 0; o < numOutputs; o++) {                                          
				curCoeff = GET_COEFF(outBlk+o, i);                                               
				#pragma unroll                                                                   
				for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++)                               
					result[o][iter] ^= GF_MULTIPLY(val[iter], curCoeff);                         
			}                                                                                    
		#endif                                                                                   
	}                                                                                            
	__global val_t* dstBase = dst + WBUF_REF(outBlk, len, globalCol);                            
	#pragma unroll                                                                               
	for (nat_uint o = 0; o < numOutputs; o++) {                                                  
		#pragma unroll                                                                           
		for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++) {                                     
			WRITE_DST(dstBase, iter, result[o][iter]);                                           
		}                                                                                        
		dstBase += len;                                                                          
	}                                                                                            
}                                                                                               

#undef GET_COEFF
"#;

pub const _OCL_KERNEL_NOLUT: &str = r#"{
#ifdef LMEM_CACHE_SIZE
__local ushort gf16_antilog[LMEM_CACHE_SIZE] __attribute__ ((aligned (VECT_WIDTH*2))); __local nat_uint* lmemDst = (__local nat_uint*)gf16_antilog; LMEM_SRC_TYPE nat_uint* lmemSrc = (LMEM_SRC_TYPE nat_uint*)gf16_antilog_src; for(ushort i=get_local_id(0); i<LMEM_CACHE_SIZE/VECT_WIDTH; i+=COL_GROUP_SIZE) { lmemDst[i] = lmemSrc[i]; }
#endif

#ifdef COPY_COEFF

 #ifdef METHOD_LOG
__local ushort lcoeff[SUBMIT_INPUTS] __attribute__ ((aligned (VECT_WIDTH*2))); for(uint i=get_local_id(0); i<SUBMIT_INPUTS; i+=COL_GROUP_SIZE) lcoeff[i] = coeff[i];
 #else
__local ushort lcoeff[SUBMIT_INPUTS*NUM_OUTPUTS] __attribute__ ((aligned (VECT_WIDTH*2))); for(uint i=get_local_id(0); i<SUBMIT_INPUTS*NUM_OUTPUTS; i+=COL_GROUP_SIZE) lcoeff[i] = coeff[i];
 #endif

#else
__global const ushort* lcoeff = coeff;
#endif

#if defined(COPY_COEFF) || defined(LMEM_CACHE_SIZE)
barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if NUM_OUTPUTS <= OUTPUTS_PER_THREAD
nat_uint outBlk = 0; const bool isPartialOutputsThread = false;
#else
nat_uint outBlk = get_global_id(1) * OUTPUTS_PER_THREAD;
#if NUM_OUTPUTS % OUTPUTS_PER_THREAD == 0
const bool isPartialOutputsThread = false;
#else
const bool isPartialOutputsThread = get_global_id(1) == get_global_size(1);
#endif

#endif
__local val_t cache[SUBMIT_INPUTS*COL_GROUP_SIZE*COL_GROUP_ITERS]; if(!isPartialOutputsThread) {
#if OUTPUTS_PER_THREAD > OUTPUT_GROUPING
KERNFN(mulgroup_cache)(dst, src, lcoeff, outBlk, OUTPUT_GROUPING, numInputs, cache EX_TABLE_ARGS); outBlk += OUTPUT_GROUPING; for (; outBlk < OUTPUTS_PER_THREAD - OUTPUT_GROUPING + 1; outBlk += OUTPUT_GROUPING) { KERNFN(mulgroup_read)(dst, src, lcoeff, outBlk, OUTPUT_GROUPING, numInputs, cache EX_TABLE_ARGS); }
#endif
#if OUTPUTS_PER_THREAD == OUTPUT_GROUPING || OUTPUTS_PER_THREAD % OUTPUT_GROUPING > 0

#if OUTPUTS_PER_THREAD > OUTPUT_GROUPING
KERNFN(mulgroup_read)(dst, src, lcoeff, outBlk, (OUTPUTS_PER_THREAD == OUTPUT_GROUPING ? OUTPUTS_PER_THREAD : OUTPUTS_PER_THREAD % OUTPUT_GROUPING), numInputs, cache EX_TABLE_ARGS);
#else
KERNFN(mulgroup_nocache)(dst, src, lcoeff, outBlk, (OUTPUTS_PER_THREAD == OUTPUT_GROUPING ? OUTPUTS_PER_THREAD : OUTPUTS_PER_THREAD % OUTPUT_GROUPING), numInputs, (__local val_t*)0 EX_TABLE_ARGS);
#endif

#endif
} else {
#define OUTPUTS_THIS_THREAD (NUM_OUTPUTS % OUTPUTS_PER_THREAD)

#if OUTPUTS_THIS_THREAD > OUTPUT_GROUPING
KERNFN(mulgroup_cache)(dst, src, lcoeff, outBlk, OUTPUT_GROUPING, numInputs, cache EX_TABLE_ARGS); outBlk += OUTPUT_GROUPING; for (; outBlk < OUTPUTS_THIS_THREAD - OUTPUT_GROUPING + 1; outBlk += OUTPUT_GROUPING) { KERNFN(mulgroup_read)(dst, src, lcoeff, outBlk, OUTPUT_GROUPING, numInputs, cache EX_TABLE_ARGS); }
#endif
#if OUTPUTS_THIS_THREAD == OUTPUT_GROUPING || OUTPUTS_THIS_THREAD % OUTPUT_GROUPING > 0

#if OUTPUTS_THIS_THREAD > OUTPUT_GROUPING
KERNFN(mulgroup_read)(dst, src, lcoeff, outBlk, (OUTPUTS_THIS_THREAD == OUTPUT_GROUPING ? OUTPUTS_THIS_THREAD : OUTPUTS_THIS_THREAD % OUTPUT_GROUPING), numInputs, cache EX_TABLE_ARGS);
#else
KERNFN(mulgroup_nocache)(dst, src, lcoeff, outBlk, (OUTPUTS_THIS_THREAD == OUTPUT_GROUPING ? OUTPUTS_THIS_THREAD : OUTPUTS_THIS_THREAD % OUTPUT_GROUPING), numInputs, (__local val_t*)0 EX_TABLE_ARGS);
#endif

#endif
} }
#undef WRITE_DST
"#;

pub const _OCL_KERNEL_CACHELUT: &str = r#"#ifdef WRITE_DST_OVERRIDE
 #ifdef MUL_ONLY
  #define WRITE_DST WRITE_DST_OVERRIDE
 #else
  #define WRITE_DST WRITE_DST_ADD_OVERRIDE
 #endif
#elif defined(MUL_ONLY)
 #define WRITE_DST(dst, idx, val) dst[idx] = val
#else
 #define WRITE_DST(dst, idx, val) dst[idx] ^= val
#endif
{ const uint colLocal = get_local_id(0); const memsize_t colGlobal = colLocal + get_group_id(0)*COL_GROUP_SIZE*COL_GROUP_ITERS; const memsize_t len = get_global_size(0) * COL_GROUP_ITERS;
#if NUM_OUTPUTS <= OUTPUTS_PER_THREAD
const nat_uint outputsThisThread = NUM_OUTPUTS; const nat_uint outBase = 0; __global val_t* dstBase = dst + colGlobal;
#ifdef WRITE_GRP2
__global val_t* dstBase2 = dst + colGlobal*2;
#endif

#else
const nat_uint outBase = get_global_id(1) * OUTPUTS_PER_THREAD;
#if NUM_OUTPUTS % OUTPUTS_PER_THREAD == 0
const nat_uint outputsThisThread = OUTPUTS_PER_THREAD;
#else
const nat_uint outputsThisThread = (get_global_id(1)+1 == get_global_size(1) ? NUM_OUTPUTS % OUTPUTS_PER_THREAD : OUTPUTS_PER_THREAD);
#endif
__global val_t* dstBase = dst + WBUF_REF(outBase, len, colGlobal);
#ifdef WRITE_GRP2
__global val_t* dstBase2 = dst + WBUF_REF(outBase, len, colGlobal*2);
#endif

#endif
__global const val_t* srcBase = src + colGlobal;
#ifdef WRITE_GRP2
LUT_DECLARATION_X2(lut_table_x2, SUBMIT_INPUTS); LUT_DECLARATION_X2_1(lut_table, lut_table_x2);
#else
LUT_DECLARATION(lut_table, SUBMIT_INPUTS);
#endif
if(outputsThisThread == 1) { LUT_GENERATE(lut_table, coeff + COBUF_REF(outBase, 0u), numInputs); for(uint iter=0; iter<COL_GROUP_ITERS; iter++) { uint iterBase = iter*COL_GROUP_SIZE; val_t val = READ_SRC(srcBase, 0u); val_t result = LUT_MULTIPLY(LUT_REF(lut_table, 0u), val); __global const val_t* srcIterBase = srcBase;
#ifdef numInputs
#pragma unroll
#endif
for (nat_uint input = 1; input < numInputs; input++) { srcIterBase += len; val = READ_SRC(srcIterBase, 0u); result ^= LUT_MULTIPLY(LUT_REF(lut_table, input), val); } WRITE_DST(dstBase, iterBase, result); srcBase += COL_GROUP_SIZE; } return; } __local val_t cache[SUBMIT_INPUTS][COL_GROUP_SIZE*COL_GROUP_ITERS];
#ifdef WRITE_GRP2
LUT_GENERATE_X2(lut_table_x2, coeff + COBUF_REF(outBase, 0u), numInputs);
#else
LUT_GENERATE(lut_table, coeff + COBUF_REF(outBase, 0u), numInputs);
#endif

#ifndef LUT_GENERATE_HAS_BARRIER

#endif
for(uint iter=0; iter<COL_GROUP_ITERS; iter++) { uint iterBase = iter*COL_GROUP_SIZE; val_t val = READ_SRC(srcBase, 0u); cache[0][colLocal+iterBase] = val;
#ifdef WRITE_GRP2
uint2 result = LUT_MULTIPLY_X2(LUT_REF_X2(lut_table_x2, 0u), val);
#else
val_t result = LUT_MULTIPLY(LUT_REF(lut_table, 0u), val);
#endif
__global const val_t* srcIterBase = srcBase;
#ifdef numInputs
#pragma unroll
#endif
for (nat_uint input = 1; input < numInputs; input++) { srcIterBase += len; val = READ_SRC(srcIterBase, 0u); cache[input][colLocal + iterBase] = val;
#ifdef WRITE_GRP2
result ^= LUT_MULTIPLY_X2(LUT_REF_X2(lut_table_x2, input), val);
#else
result ^= LUT_MULTIPLY(LUT_REF(lut_table, input), val);
#endif
}
#ifdef WRITE_GRP2
WRITE_DST(dstBase2, iterBase*2, result[0]); WRITE_DST(dstBase2, iterBase*2+1, result[1]);
#else
WRITE_DST(dstBase, iterBase, result);
#endif
srcBase += COL_GROUP_SIZE; }
#ifdef WRITE_GRP2
nat_uint output=2; for(; output<outputsThisThread-1; output+=2) { dstBase2 += len*2; LUT_GENERATE_X2(lut_table_x2, coeff + COBUF_REF(outBase+output, 0u), numInputs); for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++) { uint iterBase = iter*COL_GROUP_SIZE; uint2 result = LUT_MULTIPLY_X2(LUT_REF_X2(lut_table, 0u), cache[0][colLocal + iterBase]);
#ifdef numInputs
#pragma unroll
#endif
for (ushort input = 1; input < numInputs; input++) { result ^= LUT_MULTIPLY_X2(LUT_REF_X2(lut_table, input), cache[input][colLocal + iterBase]); } WRITE_DST(dstBase2, iterBase*2, result[0]); WRITE_DST(dstBase2, iterBase*2+1, result[1]); } } dstBase += WBUF_REF(outputsThisThread-2, len, 0u); if(outputsThisThread & 1)
#else
for(nat_uint output=1; output<outputsThisThread; output++)
#endif
{ dstBase += len; LUT_GENERATE(lut_table, coeff + COBUF_REF(outBase+output, 0u), numInputs); for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++) { uint iterBase = iter*COL_GROUP_SIZE; val_t result = LUT_MULTIPLY(LUT_REF(lut_table, 0u), cache[0][colLocal + iterBase]);
#ifdef numInputs
#pragma unroll
#endif
for (ushort input = 1; input < numInputs; input++) { result ^= LUT_MULTIPLY(LUT_REF(lut_table, input), cache[input][colLocal + iterBase]); } WRITE_DST(dstBase, iterBase, result); } } }
#undef WRITE_DST
"#;

pub const _OCL_KERNEL_LUT_FUNCS: &str = r#"#ifdef WRITE_DST_OVERRIDE
 #ifdef MUL_ONLY
  #define WRITE_DST WRITE_DST_OVERRIDE
 #else
  #define WRITE_DST WRITE_DST_ADD_OVERRIDE
 #endif
#elif defined(MUL_ONLY)
 #define WRITE_DST(dst, idx, val) dst[idx] = val
#else
 #define WRITE_DST(dst, idx, val) dst[idx] ^= val
#endif
inline void KERNFN(mulgroup)(__global val_t* restrict dst, __global const val_t* restrict src, __global const ushort* restrict coeff, __local void* lut_table, const nat_uint outBlk, const nat_uint numOutputs, const ushort _numInputs) { const memsize_t len = get_global_size(0) * COL_GROUP_ITERS; const memsize_t globalCol = get_local_id(0) + get_group_id(0)*COL_GROUP_SIZE*COL_GROUP_ITERS;
#ifdef WRITE_GRP2

#pragma unroll
for (ushort o = 0; o < numOutputs-1; o+=2) { LUT_GENERATE_X2(LUT_REF_X2(lut_table, o*SUBMIT_INPUTS/2), coeff + COBUF_REF(outBlk+o, 0u), _numInputs); } if(numOutputs & 1) { LUT_GENERATE(LUT_REF(lut_table, (numOutputs-1)*SUBMIT_INPUTS), coeff + COBUF_REF(outBlk+(numOutputs-1), 0u), _numInputs); }
#else

#pragma unroll
for (ushort o = 0; o < numOutputs; o++) { LUT_GENERATE(LUT_REF(lut_table, o*SUBMIT_INPUTS), coeff + COBUF_REF(outBlk+o, 0u), _numInputs); }
#endif

#ifdef WRITE_GRP2
__global val_t* dstBase2 = dst + WBUF_REF(outBlk, len, globalCol*2); __global val_t* dstBase = dst + WBUF_REF(outBlk + numOutputs -1, len, globalCol);
#else
__global val_t* dstBase = dst + WBUF_REF(outBlk, len, globalCol);
#endif
__global const val_t* srcBase = src + globalCol; for(nat_uint iter=0; iter<COL_GROUP_ITERS; iter++) { val_t val = READ_SRC(srcBase, 0u);
#ifdef WRITE_GRP2
uint2 result[OUTPUT_GROUPING/2]; uint resultLast;
#pragma unroll
for (ushort o = 0; o < numOutputs/2; o++) { result[o] = LUT_MULTIPLY_X2(LUT_REF_X2(lut_table, o*SUBMIT_INPUTS), val); } if(numOutputs & 1) resultLast = LUT_MULTIPLY(LUT_REF(lut_table, (numOutputs-1)*SUBMIT_INPUTS), val);
#else
val_t result[OUTPUT_GROUPING];
#pragma unroll
for (ushort o = 0; o < numOutputs; o++) { result[o] = LUT_MULTIPLY(LUT_REF(lut_table, o*SUBMIT_INPUTS), val); }
#endif
__global const val_t* srcIterBase = srcBase; for (ushort i = 1; i < _numInputs; i++) { srcIterBase += len; val = READ_SRC(srcIterBase, 0u);
#ifdef WRITE_GRP2

#pragma unroll
for (ushort o = 0; o < numOutputs/2; o++) { result[o] ^= LUT_MULTIPLY_X2(LUT_REF_X2(lut_table, o*SUBMIT_INPUTS+i), val); } if(numOutputs & 1) resultLast ^= LUT_MULTIPLY(LUT_REF(lut_table, (numOutputs-1)*SUBMIT_INPUTS+i), val);
#else

#pragma unroll
for (ushort o = 0; o < numOutputs; o++) { result[o] ^= LUT_MULTIPLY(LUT_REF(lut_table, o*SUBMIT_INPUTS+i), val); }
#endif
}
#ifdef WRITE_GRP2
__global val_t* dstIterBase2 = dstBase2;
#pragma unroll
for (ushort o = 0; o < numOutputs/2; o++) { WRITE_DST(dstIterBase2, 0u, result[o][0]); WRITE_DST(dstIterBase2, 1u, result[o][1]); dstIterBase2 += len*2; } if(numOutputs & 1) { WRITE_DST(dstBase, 0u, resultLast); } dstBase2 += COL_GROUP_SIZE*2;
#else
__global val_t* dstIterBase = dstBase;
#pragma unroll
for (ushort o = 0; o < numOutputs; o++) { WRITE_DST(dstIterBase, 0u, result[o]); dstIterBase += len; }
#endif
srcBase += COL_GROUP_SIZE; dstBase += COL_GROUP_SIZE; } }"#;

pub const _OCL_KERNEL_LUT: &str = r#"{
#if NUM_OUTPUTS <= OUTPUTS_PER_THREAD
nat_uint outBlk = 0; const bool isPartialOutputsThread = false;
#else
nat_uint outBlk = get_global_id(1) * OUTPUTS_PER_THREAD;
#if NUM_OUTPUTS % OUTPUTS_PER_THREAD == 0
const bool isPartialOutputsThread = false;
#else
const bool isPartialOutputsThread = get_global_id(1)+1 == get_global_size(1);
#endif

#endif
LUT_DECLARATION(lut_table, SUBMIT_INPUTS*OUTPUT_GROUPING); if(!isPartialOutputsThread) {
#if OUTPUTS_PER_THREAD > OUTPUT_GROUPING
for (; outBlk < OUTPUTS_PER_THREAD - OUTPUT_GROUPING + 1; outBlk += OUTPUT_GROUPING) { KERNFN(mulgroup)(dst, src, coeff, lut_table, outBlk, OUTPUT_GROUPING, numInputs); }
#endif
#if OUTPUTS_PER_THREAD == OUTPUT_GROUPING || OUTPUTS_PER_THREAD % OUTPUT_GROUPING > 0
{
#define NUM_OUTPUTS_LEFT (OUTPUTS_PER_THREAD == OUTPUT_GROUPING ? OUTPUTS_PER_THREAD : OUTPUTS_PER_THREAD % OUTPUT_GROUPING)
KERNFN(mulgroup)(dst, src, coeff, lut_table, outBlk, NUM_OUTPUTS_LEFT, numInputs);
#undef NUM_OUTPUTS_LEFT
}
#endif
} else {
#define OUTPUTS_THIS_THREAD (NUM_OUTPUTS % OUTPUTS_PER_THREAD)

#if OUTPUTS_THIS_THREAD > OUTPUT_GROUPING
for (; outBlk < OUTPUTS_THIS_THREAD - OUTPUT_GROUPING + 1; outBlk += OUTPUT_GROUPING) { KERNFN(mulgroup)(dst, src, coeff, lut_table, outBlk, OUTPUT_GROUPING, numInputs); }
#endif
#if OUTPUTS_THIS_THREAD == OUTPUT_GROUPING || OUTPUTS_THIS_THREAD % OUTPUT_GROUPING > 0
{
#define NUM_OUTPUTS_LEFT (OUTPUTS_THIS_THREAD == OUTPUT_GROUPING ? OUTPUTS_THIS_THREAD : OUTPUTS_THIS_THREAD % OUTPUT_GROUPING)
KERNFN(mulgroup)(dst, src, coeff, lut_table, outBlk, NUM_OUTPUTS_LEFT, numInputs);
#undef NUM_OUTPUTS_LEFT
}
#endif
} }
#undef WRITE_DST
"#;
