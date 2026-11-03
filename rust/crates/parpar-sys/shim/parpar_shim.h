/* parpar_shim.h - C ABI over the ParPar C++ compute backend.
 *
 * This mirrors the interface the Node addon (src/gf.cc) exposes to JS,
 * with JS callbacks replaced by a polled event queue. All controller calls
 * run on a shim-owned libuv thread, exactly as they ran on Node's main loop.
 *
 * Threading model:
 *   - pps_ctx_new() spawns one libuv loop thread.
 *   - All pps_gfp_* calls marshal to that thread and return the immediate
 *     result; completions arrive later via pps_wait_event().
 *   - Input/output buffer pointers are borrowed: the caller must keep the
 *     memory alive and unmodified until the matching completion event.
 *   - the pps_hin and pps_hout calls are synchronous and may run on any
 *     thread, but calls on one handle must be serialised by the caller.
 *   - No function unwinds across the ABI; errors are returned as codes plus
 *     an optional caller-provided message buffer.
 */
#ifndef PARPAR_SHIM_H
#define PARPAR_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
# ifdef PARPAR_SHIM_BUILD
#  define PPS_API __declspec(dllexport)
# else
#  define PPS_API __declspec(dllimport)
# endif
#else
# define PPS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pps_ctx pps_ctx;
typedef struct pps_gfp pps_gfp;
typedef struct pps_hin pps_hin;
typedef struct pps_hout pps_hout;

/* ---------------- context ---------------- */

/* Create a context: one libuv loop running on a dedicated thread.
 * Returns NULL on failure. */
PPS_API pps_ctx* pps_ctx_new(void);

/* Destroy the context. Any remaining GfProc handles are closed and freed
 * first (blocking). Safe to call with live gfps; not safe concurrently
 * with other calls on the same ctx. */
PPS_API void pps_ctx_free(pps_ctx* ctx);

/* ---------------- events ---------------- */

enum pps_event_kind {
	PPS_EV_INPUT_DONE  = 1, /* input consumed; value = input index */
	PPS_EV_ENDED       = 2, /* end_input processing finished */
	PPS_EV_OUTPUT_DONE = 3, /* get_output done; value = cksum valid (0/1) */
	PPS_EV_PROGRESS    = 4, /* value = number of inputs currently staged */
	PPS_EV_ERROR       = 5  /* asynchronous error; errbuf set */
};

struct pps_event {
	pps_gfp* proc;          /* owning GfProc handle */
	int kind;
	uint64_t token;         /* caller token passed to the originating call */
	int64_t value;
	int err;
	char errbuf[192];
};

/* Block until an event is available.
 * timeout_ms < 0 waits forever.
 * Returns 1 if *out was filled, 0 on timeout, -1 if the context is torn
 * down (no more events will ever arrive). */
PPS_API int pps_wait_event(pps_ctx* ctx, struct pps_event* out, int timeout_ms);

/* ---------------- GfProc ---------------- */

struct pps_cpu_spec {
	int method;                    /* Galois16Methods; 0 = GF16_AUTO */
	uint32_t input_batchsize;      /* 0 = default */
	uint32_t input_minbatchsize;   /* 0 = default */
	uint32_t chunk_size;           /* 0 = default */
	uint64_t slice_offset;         /* byte offset into each slice */
	uint64_t slice_size;           /* bytes of each slice handled by CPU */
};

struct pps_ocl_spec {
	int platform;                  /* -1 = default platform */
	int device;                    /* -1 = default device */
	uint64_t slice_size;
	uint64_t slice_offset;
	int method;                    /* Galois16OCLMethods; 0 = GF16OCL_AUTO */
	int cksum_method;              /* Galois16Methods; 0 = inherit cpu method */
	int input_batchsize;
	int input_minbatchsize;
	uint32_t target_iters;
	uint32_t target_grouping;
};

/* Create a GfProc. Validation mirrors GfProc::New in src/gf.cc.
 * cpu may be NULL to disable CPU processing.
 * staging_areas <= 0 selects the default (2).
 * On success returns 0 and *out is set; on failure returns -1 and, when
 * errbuf != NULL, writes the JS-equivalent error message. */
PPS_API int pps_gfp_new(pps_ctx* ctx, uint64_t slice_size,
			const struct pps_cpu_spec* cpu,
			const struct pps_ocl_spec* ocl, size_t n_ocl,
			int staging_areas,
			pps_gfp** out, char* errbuf, size_t errlen);

/* Returns actual thread count, or -1 on error (errbuf set). */
PPS_API int pps_gfp_set_num_threads(pps_gfp* g, int threads, char* errbuf, size_t errlen);

PPS_API int pps_gfp_set_current_slice_size(pps_gfp* g, uint64_t slice_size, char* errbuf, size_t errlen);

/* exponents may be NULL when n == the desired count (all exponents default to 1). */
PPS_API int pps_gfp_set_recovery_slices(pps_gfp* g, const uint16_t* exponents, size_t n, char* errbuf, size_t errlen);

PPS_API int pps_gfp_free_mem(pps_gfp* g, char* errbuf, size_t errlen);

/* Submit one input slice. buf is borrowed until PPS_EV_INPUT_DONE arrives
 * with the same token. Returns:
 *   0  accepted
 *   1  staging full (caller should wait for progress events and retry)
 *  -1  error (errbuf set) */
PPS_API int pps_gfp_add_input(pps_gfp* g, const void* buf, size_t len,
			      int input_index, uint64_t token, char* errbuf, size_t errlen);

/* Signal end of input. Returns 0 once queued; PPS_EV_ENDED arrives when
 * all processing completes. */
PPS_API int pps_gfp_end_input(pps_gfp* g, uint64_t token, char* errbuf, size_t errlen);

/* Fetch one recovery output slice into out; out_len is its size in bytes
 * (validated against the current slice size like the JS Buffer check).
 * Returns 0 once queued; PPS_EV_OUTPUT_DONE arrives with value = cksumValid. */
PPS_API int pps_gfp_get_output(pps_gfp* g, int output_index, void* out, size_t out_len,
			       uint64_t token, char* errbuf, size_t errlen);

/* Enable/disable progress events (PPS_EV_PROGRESS). */
PPS_API int pps_gfp_set_progress(pps_gfp* g, int on, char* errbuf, size_t errlen);

struct pps_gf_cpu_info {
	int threads;
	char method_desc[64];
	uint64_t chunk_size;
	uint32_t staging_count;
	uint64_t staging_size;
	uint32_t alignment;
	uint64_t stride;
	uint64_t slice_mem;
	int num_output_slices;
};
struct pps_gf_ocl_info {
	char device_name[128];
	char method_desc[64];
	uint32_t staging_count;
	uint64_t staging_size;
	uint64_t chunk_size;
	uint32_t output_chunks;
	uint64_t slice_mem;
	int num_output_slices;
};
/* Returns 0; *cpu set only if has_cpu, *n_ocl gets the ocl backend count. */
PPS_API int pps_gfp_info(pps_gfp* g, int* has_cpu, struct pps_gf_cpu_info* cpu,
			int* n_ocl, char* errbuf, size_t errlen);
PPS_API int pps_gfp_ocl_info(pps_gfp* g, int index, struct pps_gf_ocl_info* out,
			char* errbuf, size_t errlen);

/* Close the processor: waits for in-flight work, runs deinit and releases
 * native resources. Blocks until deinit completes. The handle itself must
 * then be released with pps_gfp_free. */
PPS_API int pps_gfp_close(pps_gfp* g, char* errbuf, size_t errlen);

/* Free the handle. Only valid after pps_gfp_close succeeded. */
PPS_API void pps_gfp_free(pps_gfp* g);

/* ---------------- method / device info ---------------- */

struct pps_method_info {
	int id;
	char name[64];
	uint64_t alignment;
	uint64_t stride;
	uint64_t target_chunk;
	uint32_t target_grouping;
};

/* method == 0 (GF16_AUTO) resolves to the platform default first. */
PPS_API int pps_gf_info(int method, struct pps_method_info* out);

PPS_API int pps_set_inhash(int method);      /* returns 1 ok, 0 invalid */
PPS_API void pps_set_outhash(int level);
PPS_API const char* pps_inhash_method(void);
PPS_API const char* pps_outhash_method(void);

struct pps_ocl_dev {
	int id;
	char name[256];
	int32_t vendor_id;
	int available;
	int supported;
	char type[16];
	uint64_t memory_global;
	uint64_t cache_global;
	int memory_unified;
	uint64_t memory_constant;
	uint64_t memory_local;
	int has_memory_local;
	uint64_t memory_max_alloc;
	uint32_t workgroup_limit;
	uint32_t workgroup_multiple;
	uint32_t compute_units;
};

PPS_API int pps_ocl_load(char* errbuf, size_t errlen);            /* 0 ok, -1 no runtime */
PPS_API int pps_ocl_default_platform(void);                       /* -1 if none */
PPS_API int pps_ocl_default_device(int platform);                 /* -1 if none */
PPS_API int pps_ocl_platform_count(void);
PPS_API int pps_ocl_platform_name(int platform, char* buf, size_t len); /* 0 ok */
PPS_API int pps_ocl_device_count(int platform);
PPS_API int pps_ocl_device_info(int platform, int device, struct pps_ocl_dev* out);

/* ---------------- input hasher (synchronous; caller serialises) ---------------- */

/* slice_size: PAR2 slice size. block_out: caller buffer receiving 20-byte
 * (MD5+CRC32) per-slice records as slices complete; block_out_len is its
 * size in bytes (capacity in records = len/20, mirroring gf.cc).
 * The buffer must outlive the handle. Returns NULL on failure. */
PPS_API pps_hin* pps_hin_new(uint64_t slice_size, uint8_t* block_out, size_t block_out_len);

/* Feed data; internally splits at slice boundaries and writes completed
 * 20-byte records into block_out. Must be serialised per handle. */
PPS_API int pps_hin_update(pps_hin* h, const void* buf, size_t len, char* errbuf, size_t errlen);

/* Finalise: writes the trailing partial slice's 20-byte record (zero-padded
 * to slice_size) if capacity remains, then the 16-byte file MD5 to out.
 * out must be >= 16 bytes. Destroys the hasher; the handle must then be
 * freed with pps_hin_free. */
PPS_API int pps_hin_end(pps_hin* h, uint8_t* out, size_t out_len, char* errbuf, size_t errlen);

PPS_API int pps_hin_reset(pps_hin* h, char* errbuf, size_t errlen);
PPS_API void pps_hin_free(pps_hin* h);

/* ---------------- output hasher (synchronous) ---------------- */

/* regions: number of equal-length buffers hashed in parallel (1..65534). */
PPS_API pps_hout* pps_hout_new(unsigned regions);

/* bufs: array of `regions` pointers, each region `len` bytes. */
PPS_API int pps_hout_update(pps_hout* h, const void* const* bufs, size_t num_bufs,
			    size_t len, char* errbuf, size_t errlen);

/* Finalise and write regions*16 bytes of MD5s to out. */
PPS_API int pps_hout_get(pps_hout* h, uint8_t* out, size_t out_len, char* errbuf, size_t errlen);

PPS_API int pps_hout_reset(pps_hout* h, char* errbuf, size_t errlen);
PPS_API void pps_hout_free(pps_hout* h);

#ifdef __cplusplus
}
#endif
#endif /* PARPAR_SHIM_H */
