/* parpar_shim.cc - implementation of the C ABI declared in parpar_shim.h.
 *
 * Mirrors src/gf.cc semantics with Node/V8 replaced by an owned libuv loop
 * thread plus a polled event queue. Validation order, state transitions and
 * error strings follow the addon exactly; see header for the threading
 * contract.
 */
#include "parpar_shim.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <uv.h>

#include "controller.h"
#include "controller_cpu.h"
#include "hasher_input.h"
#include "hasher_md5mb.h"
#include "hasher.h"

#ifdef PARPAR_OPENCL_SUPPORT
# include "controller_ocl.h"
#endif

/* The opaque typedefs in the header are completed by defining the same
 * struct names at global scope below. */

struct pps_ctx {
	std::thread thr;
	uv_loop_t* loop = nullptr;
	uv_async_t* cmd_async = nullptr;

	std::mutex mtx;
	std::condition_variable cv;
	std::deque<std::function<void()>> cmds;
	bool ready = false;
	bool cmd_handle_closed = false;

	std::mutex evmtx;
	std::condition_variable evcv;
	std::deque<pps_event> events;
	bool events_closed = false;

	std::mutex gmtx;
	std::vector<pps_gfp*> gfps;
};

namespace {

/* ---------- small helpers ---------- */

void set_err(char* eb, size_t ebl, const char* msg) {
	if(eb && ebl) {
		std::snprintf(eb, ebl, "%s", msg ? msg : "Unknown error");
	}
}

void push_event(pps_ctx* c, pps_gfp* g, int kind, uint64_t token, int64_t value,
		int err = 0, const char* msg = nullptr) {
	pps_event ev;
	std::memset(&ev, 0, sizeof(ev));
	ev.proc = g;
	ev.kind = kind;
	ev.token = token;
	ev.value = value;
	ev.err = err;
	if(msg) set_err(ev.errbuf, sizeof(ev.errbuf), msg);
	{
		std::lock_guard<std::mutex> lk(c->evmtx);
		if(c->events_closed) return;
		c->events.push_back(ev);
	}
	c->evcv.notify_one();
}

void ctx_thread_main(pps_ctx* c) {
	uv_loop_t loop;
	uv_loop_init(&loop);
	uv_async_t async;
	uv_async_init(&loop, &async, [](uv_async_t* h) {
		pps_ctx* c = static_cast<pps_ctx*>(h->data);
		std::deque<std::function<void()>> jobs;
		{
			std::lock_guard<std::mutex> lk(c->mtx);
			jobs.swap(c->cmds);
		}
		for(auto& j : jobs) j();
	});
	async.data = c;
	{
		std::lock_guard<std::mutex> lk(c->mtx);
		c->loop = &loop;
		c->cmd_async = &async;
		c->ready = true;
	}
	c->cv.notify_all();
	uv_run(&loop, UV_RUN_DEFAULT);
	uv_loop_close(&loop);
	{
		std::lock_guard<std::mutex> lk(c->mtx);
		c->loop = nullptr;
		c->cmd_async = nullptr;
	}
	// wake any waiters in pps_wait_event
	{
		std::lock_guard<std::mutex> lk(c->evmtx);
		c->events_closed = true;
	}
	c->evcv.notify_all();
}

/* Post work to the loop thread. Must not be called after ctx_free begins. */
void post(pps_ctx* c, std::function<void()> f) {
	{
		std::lock_guard<std::mutex> lk(c->mtx);
		c->cmds.push_back(std::move(f));
	}
	uv_async_send(c->cmd_async);
}

/* Post work and block until it completes; returns the job's result. */
template<typename T>
T call_sync(pps_ctx* c, std::function<T()> job) {
	struct Waiter {
		std::mutex m;
		std::condition_variable cv;
		bool done = false;
		T result{};
	};
	auto w = std::make_shared<Waiter>();
	post(c, [w, job = std::move(job)]() mutable {
		w->result = job();
		std::lock_guard<std::mutex> lk(w->m);
		w->done = true;
		w->cv.notify_one();
	});
	std::unique_lock<std::mutex> ul(w->m);
	w->cv.wait(ul, [&] { return w->done; });
	return w->result;
}

/* ---------- GfProc ---------- */

bool load_ocl() {
	static bool oclLoaded = false;
	if(!oclLoaded) {
#ifdef PARPAR_OPENCL_SUPPORT
		if(PAR2ProcOCL::load_runtime()) {
			return false;
		}
#else
		return false;
#endif
		oclLoaded = true;
	}
	return true;
}

} // namespace

struct pps_gfp {
	pps_ctx* ctx;
	PAR2Proc par2;
	std::unique_ptr<PAR2ProcCPU> cpu;
#ifdef PARPAR_OPENCL_SUPPORT
	std::vector<std::unique_ptr<PAR2ProcOCL>> ocl;
#endif
	bool isRunning = false;
	bool isClosed = false;
	bool pendingDiscardOutput = true;
	bool hasOutput = false;
	bool progressOn = false;
};

static inline size_t ocl_count(const pps_gfp* g) {
#ifdef PARPAR_OPENCL_SUPPORT
	return g->ocl.size();
#else
	(void)g; return 0;
#endif
}

/* ---------- context API ---------- */

pps_ctx* pps_ctx_new(void) {
	try {
		auto* c = new pps_ctx;
		c->thr = std::thread(ctx_thread_main, c);
		std::unique_lock<std::mutex> ul(c->mtx);
		c->cv.wait(ul, [&] { return c->ready; });
		return c;
	} catch(...) {
		return nullptr;
	}
}

void pps_ctx_free(pps_ctx* c) {
	if(!c) return;
	try {
		// close + free any remaining gfps (close waits for deinit)
		std::vector<pps_gfp*> live;
		{
			std::lock_guard<std::mutex> lk(c->gmtx);
			live = c->gfps;
		}
		for(auto* g : live) {
			pps_gfp_close(g, nullptr, 0);
			pps_gfp_free(g);
		}
		// stop the loop: close the command async handle then stop
		call_sync<int>(c, [c] {
			uv_close(reinterpret_cast<uv_handle_t*>(c->cmd_async),
				 [](uv_handle_t*) {});
			uv_stop(c->loop);
			return 0;
		});
		c->thr.join();
	} catch(...) {
		// best-effort teardown; nothing should throw, but never unwind out
	}
	delete c;
}

int pps_wait_event(pps_ctx* c, pps_event* out, int timeout_ms) {
	if(!c || !out) return -1;
	std::unique_lock<std::mutex> ul(c->evmtx);
	auto pred = [&] { return !c->events.empty() || c->events_closed; };
	if(timeout_ms < 0) {
		c->evcv.wait(ul, pred);
	} else if(!c->evcv.wait_for(ul, std::chrono::milliseconds(timeout_ms), pred)) {
		return 0;
	}
	if(c->events.empty()) return -1;
	*out = c->events.front();
	c->events.pop_front();
	return 1;
}

/* ---------- GfProc API ---------- */

namespace {

struct GfpCreateResult {
	pps_gfp* g = nullptr;
	const char* err = nullptr;
};

} // namespace

int pps_gfp_new(pps_ctx* ctx, uint64_t slice_size,
		const struct pps_cpu_spec* cpu,
		const struct pps_ocl_spec* ocl, size_t n_ocl,
		int staging_areas,
		pps_gfp** out, char* errbuf, size_t errlen) {
	if(!ctx || !out) { set_err(errbuf, errlen, "Invalid argument"); return -1; }
	*out = nullptr;
	try {
		/* --- validation: mirrors GfProc::New in src/gf.cc --- */
		size_t sliceSize = (size_t)slice_size;
		if(sliceSize < 2 || sliceSize & 1) {
			set_err(errbuf, errlen, "Slice size is invalid");
			return -1;
		}

		bool useCpu = cpu != nullptr;
		Galois16Methods cpuMethod = GF16_AUTO;
		unsigned cpuInputGrouping = 0, cpuInputMinGrouping = 0;
		size_t cpuChunkLen = 0;
		size_t cpuOffset = 0, cpuSliceSize = 0;
		if(useCpu) {
			cpuMethod = (Galois16Methods)cpu->method;
			cpuInputGrouping = cpu->input_batchsize;
			if(cpuInputGrouping > 32768) {
				set_err(errbuf, errlen, "Input batchsize is invalid");
				return -1;
			}
			cpuInputMinGrouping = cpu->input_minbatchsize;
			cpuChunkLen = cpu->chunk_size;
			cpuOffset = (size_t)cpu->slice_offset;
			if(cpuOffset & 1 || cpuOffset > sliceSize) {
				set_err(errbuf, errlen, "Invalid CPU slice offset");
				return -1;
			}
			cpuSliceSize = (size_t)(cpu->slice_size ? cpu->slice_size : slice_size);
			if(cpuSliceSize < 2 || cpuSliceSize & 1) {
				set_err(errbuf, errlen, "CPU slice size must be a multiple of 2");
				return -1;
			}
			if(cpuOffset + cpuSliceSize > sliceSize) {
				set_err(errbuf, errlen, "CPU slice offset+size cannot exceed the slice size");
				return -1;
			}
		}
		if(staging_areas <= 0) staging_areas = 2;
		if(staging_areas < 1 || staging_areas > 32768) {
			set_err(errbuf, errlen, "Staging area count is invalid");
			return -1;
		}
		for(size_t i = 0; i < n_ocl; i++) {
			if(ocl[i].input_batchsize > 32768) {
				set_err(errbuf, errlen, "OpenCL input batchsize is invalid");
				return -1;
			}
			if(ocl[i].target_grouping > 65535) {
				set_err(errbuf, errlen, "OpenCL target grouping is invalid");
				return -1;
			}
		}
		if(cpuInputGrouping * (unsigned)staging_areas > 65536) {
			set_err(errbuf, errlen, "Staging area too large");
			return -1;
		}
		if(!n_ocl) {
			if(!useCpu) {
				set_err(errbuf, errlen, "At least the CPU or one OpenCL device must be enabled");
				return -1;
			}
		} else if(!load_ocl()) {
			set_err(errbuf, errlen, "Could not load OpenCL runtime");
			return -1;
		}

		/* --- construction on the loop thread (uv handles must be made there) --- */
		auto res = call_sync<GfpCreateResult>(ctx, [&]() -> GfpCreateResult {
			GfpCreateResult r;
			auto* g = new pps_gfp;
			g->ctx = ctx;

			std::vector<struct PAR2ProcBackendAlloc> procs;
			procs.reserve(n_ocl + (useCpu ? 1 : 0));
#ifdef PARPAR_OPENCL_SUPPORT
			for(size_t i = 0; i < n_ocl; i++) {
				auto* proc = new PAR2ProcOCL(ctx->loop, ocl[i].platform, ocl[i].device, staging_areas);
				g->ocl.emplace_back(proc);
				procs.push_back({static_cast<IPAR2ProcBackend*>(proc),
						 (size_t)ocl[i].slice_offset, (size_t)ocl[i].slice_size});
			}
#endif
			if(useCpu) {
				g->cpu.reset(new PAR2ProcCPU(ctx->loop, staging_areas));
				procs.push_back({static_cast<IPAR2ProcBackend*>(g->cpu.get()),
						 cpuOffset, cpuSliceSize});
			}
			g->par2.init(sliceSize, procs, [g](unsigned numInputs) {
				if(g->progressOn)
					push_event(g->ctx, g, PPS_EV_PROGRESS, 0, (int64_t)numInputs);
			});

			if(useCpu) {
				if(!g->cpu->init(cpuMethod, cpuInputGrouping, cpuChunkLen)) {
					delete g;
					r.err = "Failed to allocate memory";
					return r;
				}
				g->cpu->setMinInputBatchSize(cpuInputMinGrouping);
			}
#ifdef PARPAR_OPENCL_SUPPORT
			size_t usedSliceSize = 0;
			for(size_t i = 0; i < n_ocl; i++) {
				usedSliceSize += (size_t)ocl[i].slice_size;
				if(ocl[i].slice_size == 0 || (ocl[i].slice_size & 1)) {
					delete g;
					r.err = "Invalid slice size allocated to OpenCL device";
					return r;
				}
				if(((uint64_t)ocl[i].slice_offset) & 1 || ocl[i].slice_offset > slice_size) {
					delete g;
					r.err = "Invalid slice offset allocated to OpenCL device";
					return r;
				}
				if((uint64_t)ocl[i].slice_offset + ocl[i].slice_size > slice_size) {
					delete g;
					r.err = "Invalid slice offset+size allocated to OpenCL device";
					return r;
				}
				if(!g->ocl[i]->init((Galois16OCLMethods)ocl[i].method,
						    ocl[i].input_batchsize, ocl[i].target_iters,
						    ocl[i].target_grouping,
						    (Galois16Methods)ocl[i].cksum_method)) {
					delete g;
					r.err = "Failed to initialise OpenCL device";
					return r;
				}
				g->ocl[i]->setMinInputBatchSize(ocl[i].input_minbatchsize);
			}
			if((useCpu && usedSliceSize >= sliceSize) || (!useCpu && usedSliceSize != sliceSize)) {
				delete g;
				r.err = "Slice portions allocated to OpenCL devices is invalid";
				return r;
			}
#endif
			r.g = g;
			return r;
		});

		if(!res.g) {
			set_err(errbuf, errlen, res.err ? res.err : "Failed to allocate memory");
			return -1;
		}
		{
			std::lock_guard<std::mutex> lk(ctx->gmtx);
			ctx->gfps.push_back(res.g);
		}
		*out = res.g;
		return 0;
	} catch(...) {
		set_err(errbuf, errlen, "Internal error");
		return -1;
	}
}

int pps_gfp_set_num_threads(pps_gfp* g, int threads, char* errbuf, size_t errlen) {
	if(!g) { set_err(errbuf, errlen, "Invalid handle"); return -1; }
	try {
		return call_sync<int>(g->ctx, [&]() -> int {
			if(g->isRunning) { set_err(errbuf, errlen, "Cannot change params whilst running"); return -1; }
			if(g->isClosed)  { set_err(errbuf, errlen, "Already closed"); return -1; }
			if(!g->cpu.get()){ set_err(errbuf, errlen, "CPU processing not enabled"); return -1; }
			g->cpu->setNumThreads(threads);
			return g->cpu->getNumThreads();
		});
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

int pps_gfp_set_current_slice_size(pps_gfp* g, uint64_t slice_size, char* errbuf, size_t errlen) {
	if(!g) { set_err(errbuf, errlen, "Invalid handle"); return -1; }
	try {
		return call_sync<int>(g->ctx, [&]() -> int {
			if(g->isRunning) { set_err(errbuf, errlen, "Cannot change params whilst running"); return -1; }
			if(g->isClosed)  { set_err(errbuf, errlen, "Already closed"); return -1; }
			size_t sz = (size_t)slice_size;
			if(sz < 2 || sz & 1) { set_err(errbuf, errlen, "Slice size is invalid"); return -1; }
			g->hasOutput = false;
			if(!g->par2.setCurrentSliceSize(sz)) {
				set_err(errbuf, errlen, "Failed to allocate memory");
				return -1;
			}
			return 0;
		});
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

int pps_gfp_set_recovery_slices(pps_gfp* g, const uint16_t* exps, size_t n, char* errbuf, size_t errlen) {
	if(!g) { set_err(errbuf, errlen, "Invalid handle"); return -1; }
	try {
		std::vector<uint16_t> outputs;
		if(exps) outputs.assign(exps, exps + n);
		return call_sync<int>(g->ctx, [&]() -> int {
			// validation order mirrors GfProc::SetRecoverySlices
			if(g->isRunning) { set_err(errbuf, errlen, "Cannot change params whilst running"); return -1; }
			if(g->isClosed)  { set_err(errbuf, errlen, "Already closed"); return -1; }
			if(n > 65534) { set_err(errbuf, errlen, "Too many recovery indicies specified"); return -1; }
			if(n < 1)     { set_err(errbuf, errlen, "At least one recovery index must be supplied"); return -1; }
			if(exps) {
				for(size_t i = 0; i < n; i++) {
					if(exps[i] > 65534) {
						set_err(errbuf, errlen, "Invalid recovery index supplied");
						return -1;
					}
				}
			}
			g->hasOutput = false;
			bool ok = exps ? g->par2.setRecoverySlices(outputs)
				       : g->par2.setRecoverySlices((unsigned)n);
			if(!ok) { set_err(errbuf, errlen, "Failed to allocate memory"); return -1; }
			return 0;
		});
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

int pps_gfp_free_mem(pps_gfp* g, char* errbuf, size_t errlen) {
	if(!g) { set_err(errbuf, errlen, "Invalid handle"); return -1; }
	try {
		return call_sync<int>(g->ctx, [&]() -> int {
			if(g->isRunning) { set_err(errbuf, errlen, "Cannot free memory whilst running"); return -1; }
			if(g->isClosed)  { set_err(errbuf, errlen, "Already closed"); return -1; }
			g->par2.freeProcessingMem();
			g->hasOutput = false;
			return 0;
		});
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

int pps_gfp_add_input(pps_gfp* g, const void* buf, size_t len,
		      int input_index, uint64_t token, char* errbuf, size_t errlen) {
	if(!g) { set_err(errbuf, errlen, "Invalid handle"); return -1; }
	try {
		return call_sync<int>(g->ctx, [&]() -> int {
			if(g->isClosed) { set_err(errbuf, errlen, "Already closed"); return -1; }
			if(!g->par2.getNumRecoverySlices()) {
				set_err(errbuf, errlen, "setRecoverySlices not yet called");
				return -1;
			}
			if(input_index < 0 || input_index > 32767) {
				set_err(errbuf, errlen, "Input index not valid");
				return -1;
			}
			if(len > g->par2.getCurrentSliceSize()) {
				set_err(errbuf, errlen, "Input buffer too large");
				return -1;
			}
			g->isRunning = true;
			g->hasOutput = false;
			if(g->pendingDiscardOutput) {
				g->pendingDiscardOutput = false;
				g->par2.discardOutput();
			}
			bool added = g->par2.addInput(buf, len, (uint16_t)input_index, false,
				[g, token, input_index]() {
					push_event(g->ctx, g, PPS_EV_INPUT_DONE, token, input_index);
				});
			return added ? 0 : 1;
		});
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

int pps_gfp_end_input(pps_gfp* g, uint64_t token, char* errbuf, size_t errlen) {
	if(!g) { set_err(errbuf, errlen, "Invalid handle"); return -1; }
	try {
		return call_sync<int>(g->ctx, [&]() -> int {
			if(g->isClosed) { set_err(errbuf, errlen, "Already closed"); return -1; }
			if(!g->par2.getNumRecoverySlices()) {
				set_err(errbuf, errlen, "setRecoverySlices not yet called");
				return -1;
			}
			if(g->pendingDiscardOutput)
				g->par2.discardOutput();
			g->par2.endInput([g, token]() {
				g->isRunning = false;
				g->hasOutput = true;
				g->pendingDiscardOutput = true;
				push_event(g->ctx, g, PPS_EV_ENDED, token, 0);
			});
			return 0;
		});
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

int pps_gfp_get_output(pps_gfp* g, int output_index, void* out, size_t out_len,
		       uint64_t token, char* errbuf, size_t errlen) {
	if(!g) { set_err(errbuf, errlen, "Invalid handle"); return -1; }
	try {
		return call_sync<int>(g->ctx, [&]() -> int {
			if(g->isRunning) { set_err(errbuf, errlen, "Cannot get output whilst running"); return -1; }
			if(g->isClosed)  { set_err(errbuf, errlen, "Already closed"); return -1; }
			if(!g->hasOutput) { set_err(errbuf, errlen, "No finalized output to retrieve"); return -1; }
			if(out_len < g->par2.getCurrentSliceSize()) {
				set_err(errbuf, errlen, "Output buffer too small");
				return -1;
			}
			if(output_index < 0 || output_index >= g->par2.getNumRecoverySlices()) {
				set_err(errbuf, errlen, "Recovery index is not valid");
				return -1;
			}
			g->par2.getOutput((unsigned)output_index, out, [g, token](bool cksumValid) {
				push_event(g->ctx, g, PPS_EV_OUTPUT_DONE, token, cksumValid ? 1 : 0);
			});
			return 0;
		});
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

int pps_gfp_set_progress(pps_gfp* g, int on, char* errbuf, size_t errlen) {
	if(!g) { set_err(errbuf, errlen, "Invalid handle"); return -1; }
	try {
		return call_sync<int>(g->ctx, [&]() -> int {
			if(g->isClosed) { set_err(errbuf, errlen, "Already closed"); return -1; }
			g->progressOn = on ? true : false;
			return 0;
		});
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

int pps_gfp_info(pps_gfp* g, int* has_cpu, struct pps_gf_cpu_info* cpu,
		 int* n_ocl, char* errbuf, size_t errlen) {
	if(!g || !has_cpu || !n_ocl) { set_err(errbuf, errlen, "Invalid argument"); return -1; }
	try {
		return call_sync<int>(g->ctx, [&]() -> int {
			if(g->isClosed) { set_err(errbuf, errlen, "Already closed"); return -1; }
			if(!g->par2.getNumRecoverySlices()) {
				set_err(errbuf, errlen, "setRecoverySlices not yet called");
				return -1;
			}
			*has_cpu = g->cpu.get() ? 1 : 0;
			if(*has_cpu && cpu) {
				std::memset(cpu, 0, sizeof(*cpu));
				cpu->threads = g->cpu->getNumThreads();
				set_err(cpu->method_desc, sizeof(cpu->method_desc), g->cpu->getMethodName());
				cpu->chunk_size = g->cpu->getChunkLen();
				cpu->staging_count = g->cpu->getStagingAreas();
				cpu->staging_size = g->cpu->getInputBatchSize();
				cpu->alignment = g->cpu->getAlignment();
				cpu->stride = g->cpu->getStride();
				cpu->slice_mem = g->cpu->getAllocSliceSize();
				cpu->num_output_slices = g->cpu->getNumRecoverySlices();
			}
			*n_ocl = (int)ocl_count(g);
			return 0;
		});
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

int pps_gfp_ocl_info(pps_gfp* g, int index, struct pps_gf_ocl_info* out,
			char* errbuf, size_t errlen) {
	if(!g || !out) { set_err(errbuf, errlen, "Invalid argument"); return -1; }
	try {
		return call_sync<int>(g->ctx, [&]() -> int {
			if(g->isClosed) { set_err(errbuf, errlen, "Already closed"); return -1; }
			if(!g->par2.getNumRecoverySlices()) {
				set_err(errbuf, errlen, "setRecoverySlices not yet called");
				return -1;
			}
			if(index < 0 || index >= (int)ocl_count(g)) {
				set_err(errbuf, errlen, "Invalid OpenCL device index");
				return -1;
			}
#ifdef PARPAR_OPENCL_SUPPORT
			auto* proc = g->ocl[index].get();
			auto devInfo = proc->deviceInfo();
			std::memset(out, 0, sizeof(*out));
			set_err(out->device_name, sizeof(out->device_name), devInfo.name.c_str());
			set_err(out->method_desc, sizeof(out->method_desc), proc->getMethodName());
			out->staging_count = proc->getStagingAreas();
			out->staging_size = proc->getInputBatchSize();
			out->chunk_size = proc->getChunkLen();
			out->output_chunks = proc->getOutputGrouping();
			out->slice_mem = proc->getAllocSliceSize();
			out->num_output_slices = proc->getNumRecoverySlices();
			return 0;
#else
			set_err(errbuf, errlen, "OpenCL support not built");
			return -1;
#endif
		});
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

int pps_gfp_close(pps_gfp* g, char* errbuf, size_t errlen) {
	if(!g) { set_err(errbuf, errlen, "Invalid handle"); return -1; }
	try {
		// The deinit callback only fires after the loop closes the backend
		// async handles, so the caller must wait off-thread rather than
		// inside a marshalled job (which would deadlock the loop).
		struct Waiter {
			std::mutex m; std::condition_variable cv;
			bool done = false; int result = 0; std::string err;
		};
		auto w = std::make_shared<Waiter>();
		post(g->ctx, [g, w] {
			auto finish = [w](int r, const char* e) {
				std::lock_guard<std::mutex> lk(w->m);
				w->result = r;
				if(e) w->err = e;
				w->done = true;
				w->cv.notify_one();
			};
			if(g->isRunning) { finish(-1, "Cannot close whilst running"); return; }
			if(g->isClosed)  { finish(-1, "Already closed"); return; }
			g->isClosed = true;
			g->par2.deinit([g, finish] {
				// backend uv handles are closed; safe to destruct here
				g->cpu.reset();
#ifdef PARPAR_OPENCL_SUPPORT
				g->ocl.clear();
#endif
				finish(0, nullptr);
			});
		});
		std::unique_lock<std::mutex> ul(w->m);
		w->cv.wait(ul, [&] { return w->done; });
		if(w->result != 0) {
			set_err(errbuf, errlen, w->err.empty() ? "Internal error" : w->err.c_str());
			return -1;
		}
		return 0;
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

void pps_gfp_free(pps_gfp* g) {
	if(!g) return;
	try {
		pps_ctx* ctx = g->ctx;
		if(!g->isClosed) {
			// refuse to free a live handle: in-flight callbacks capture g
			pps_gfp_close(g, nullptr, 0);
			if(!g->isClosed) return;
		}
		{
			std::lock_guard<std::mutex> lk(ctx->gmtx);
			auto& v = ctx->gfps;
			for(auto it = v.begin(); it != v.end(); ++it) {
				if(*it == g) { v.erase(it); break; }
			}
		}
		delete g;
	} catch(...) {}
}

/* ---------- method / device info ---------- */

int pps_gf_info(int method, struct pps_method_info* out) {
	if(!out) return -1;
	try {
		Galois16Methods m = (Galois16Methods)method;
		if(m == GF16_AUTO)
			m = PAR2ProcCPU::default_method();
		if(m < GF16_AUTO || m > GF16_CLMUL_RVV) return -1;
		auto info = PAR2ProcCPU::info(m);
		std::memset(out, 0, sizeof(*out));
		out->id = info.id;
		set_err(out->name, sizeof(out->name), info.name);
		out->alignment = info.alignment;
		out->stride = info.stride;
		out->target_chunk = info.idealChunkSize;
		out->target_grouping = info.idealInputMultiple;
		return 0;
	} catch(...) { return -1; }
}

static void ensure_hasher_setup();

int pps_set_inhash(int method) {
	try {
		ensure_hasher_setup();
		return set_hasherInput((HasherInputMethods)method) ? 1 : 0;
	} catch(...) { return 0; }
}
void pps_set_outhash(int level) {
	try {
		ensure_hasher_setup();
		set_hasherMD5MultiLevel((MD5MultiLevels)level);
	} catch(...) {}
}
const char* pps_inhash_method(void) {
	try { ensure_hasher_setup(); return hasherInput_methodName(); } catch(...) { return ""; }
}
const char* pps_outhash_method(void) {
	try { ensure_hasher_setup(); return hasherMD5Multi_methodName(); } catch(...) { return ""; }
}

int pps_ocl_load(char* errbuf, size_t errlen) {
	try {
		if(load_ocl()) return 0;
		set_err(errbuf, errlen, "Could not load OpenCL runtime");
		return -1;
	} catch(...) { set_err(errbuf, errlen, "Could not load OpenCL runtime"); return -1; }
}

int pps_ocl_default_platform(void) {
#ifdef PARPAR_OPENCL_SUPPORT
	if(!load_ocl()) return -1; // mirrors Ocl* entry points' runtime check
	try { return PAR2ProcOCL::defaultPlatformId(); } catch(...) { return -1; }
#else
	return -1;
#endif
}
int pps_ocl_default_device(int platform) {
#ifdef PARPAR_OPENCL_SUPPORT
	if(!load_ocl()) return -1;
	try { return PAR2ProcOCL::defaultDeviceId(platform); } catch(...) { return -1; }
#else
	(void)platform; return -1;
#endif
}
int pps_ocl_platform_count(void) {
#ifdef PARPAR_OPENCL_SUPPORT
	try { return (int)PAR2ProcOCL::getPlatforms().size(); } catch(...) { return 0; }
#else
	return 0;
#endif
}
int pps_ocl_platform_name(int platform, char* buf, size_t len) {
#ifdef PARPAR_OPENCL_SUPPORT
	try {
		if(!buf || !len) return -1;
		auto name = PAR2ProcOCL::getPlatform(platform);
		if(name.empty()) return -1;
		set_err(buf, len, name.c_str());
		return 0;
	} catch(...) { return -1; }
#else
	(void)platform; (void)buf; (void)len; return -1;
#endif
}
int pps_ocl_device_count(int platform) {
#ifdef PARPAR_OPENCL_SUPPORT
	try { return (int)PAR2ProcOCL::getDevices(platform).size(); } catch(...) { return 0; }
#else
	(void)platform; return 0;
#endif
}
int pps_ocl_device_info(int platform, int device, struct pps_ocl_dev* out) {
#ifdef PARPAR_OPENCL_SUPPORT
	if(!out) return -1;
	if(!load_ocl()) return -1; // mirrors OclDeviceInfo's runtime check
	try {
		auto d = PAR2ProcOCL::getDevice(platform, device);
		if(!d.vendorId) return -1; // invalid device, mirrors RETURN_UNDEF
		std::memset(out, 0, sizeof(*out));
		out->id = d.id;
		set_err(out->name, sizeof(out->name), d.name.c_str());
		out->vendor_id = (int32_t)d.vendorId;
		out->available = d.available ? 1 : 0;
		out->supported = d.supported ? 1 : 0;
		const char* type = "Unknown";
		switch(d.type) {
			case CL_DEVICE_TYPE_DEFAULT:     type = "Default"; break;
			case CL_DEVICE_TYPE_CPU:         type = "CPU"; break;
			case CL_DEVICE_TYPE_GPU:         type = "GPU"; break;
			case CL_DEVICE_TYPE_ACCELERATOR: type = "Accelerator"; break;
			case CL_DEVICE_TYPE_CUSTOM:      type = "Custom"; break;
			default: break;
		}
		set_err(out->type, sizeof(out->type), type);
		out->memory_global = d.memory;
		out->cache_global = d.globalCache;
		out->memory_unified = d.unifiedMemory ? 1 : 0;
		out->memory_constant = d.constantMemory;
		out->memory_local = d.localMemory;
		out->has_memory_local = (!d.localMemoryIsGlobal || d.memory != d.localMemory) ? 1 : 0;
		out->memory_max_alloc = d.maxAllocation;
		out->workgroup_limit = d.maxWorkGroup;
		out->workgroup_multiple = d.workGroupMultiple;
		out->compute_units = d.computeUnits;
		return 0;
	} catch(...) { return -1; }
#else
	(void)platform; (void)device; (void)out; return -1;
#endif
}

/* ---------- input hasher ---------- */

struct pps_hin {
	IHasherInput* hasher;
	uint64_t blockSize;
	uint64_t pos;
	int64_t count;
	uint8_t* ptr;
};

static void ensure_hasher_setup() {
	static std::once_flag f;
	std::call_once(f, [] { setup_hasher(); });
}

pps_hin* pps_hin_new(uint64_t slice_size, uint8_t* block_out, size_t block_out_len) {
	try {
		ensure_hasher_setup();
		auto* h = new pps_hin;
		h->hasher = HasherInput_Create();
		if(!h->hasher) { delete h; return nullptr; }
		h->blockSize = slice_size;
		h->pos = 0;
		h->count = (int64_t)(block_out_len / 20);
		h->ptr = block_out;
		return h;
	} catch(...) { return nullptr; }
}

int pps_hin_update(pps_hin* h, const void* buf, size_t len, char* errbuf, size_t errlen) {
	if(!h) { set_err(errbuf, errlen, "Invalid handle"); return -1; }
	try {
		if(!h->hasher) { set_err(errbuf, errlen, "Process already ended"); return -1; }
		// block-split logic ported from HasherInput::thread_func (src/gf.cc)
		const char* src = (const char*)buf;
		size_t l = len;
		uint64_t blockLeft = h->blockSize - h->pos;
		while(l >= blockLeft) {
			h->hasher->update(src, (size_t)blockLeft);
			src += blockLeft;
			l -= (size_t)blockLeft;
			blockLeft = h->blockSize;
			h->pos = 0;
			if(h->count) {
				h->hasher->getBlock(h->ptr, 0);
				h->ptr += 20;
				h->count--;
			} // else overflow: matches JS behaviour of dropping extra records
		}
		if(l) h->hasher->update(src, l);
		h->pos += l;
		return 0;
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

int pps_hin_end(pps_hin* h, uint8_t* out, size_t out_len, char* errbuf, size_t errlen) {
	if(!h) { set_err(errbuf, errlen, "Invalid handle"); return -1; }
	try {
		if(!h->hasher) { set_err(errbuf, errlen, "Process already ended"); return -1; }
		if(out_len < 16) { set_err(errbuf, errlen, "Buffer must be at least 16 bytes long"); return -1; }
		// finish final partial block (zero-padded), as in HasherInput::End
		if(h->count)
			h->hasher->getBlock(h->ptr, h->blockSize - h->pos);
		h->hasher->end(out);
		h->hasher->destroy();
		h->hasher = nullptr;
		return 0;
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

int pps_hin_reset(pps_hin* h, char* errbuf, size_t errlen) {
	if(!h) { set_err(errbuf, errlen, "Invalid handle"); return -1; }
	try {
		if(!h->hasher) { set_err(errbuf, errlen, "Process already ended"); return -1; }
		h->hasher->reset();
		return 0;
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

void pps_hin_free(pps_hin* h) {
	if(!h) return;
	try {
		if(h->hasher) { h->hasher->destroy(); h->hasher = nullptr; }
		delete h;
	} catch(...) {}
}

/* ---------- output hasher ---------- */

struct pps_hout {
	MD5Multi* hasher;
	unsigned regions;
};

pps_hout* pps_hout_new(unsigned regions) {
	try {
		if(regions < 1 || regions > 65534) return nullptr;
		ensure_hasher_setup();
		auto* h = new pps_hout;
		h->hasher = new MD5Multi((int)regions);
		h->regions = regions;
		return h;
	} catch(...) { return nullptr; }
}

int pps_hout_update(pps_hout* h, const void* const* bufs, size_t num_bufs,
		    size_t len, char* errbuf, size_t errlen) {
	if(!h) { set_err(errbuf, errlen, "Invalid handle"); return -1; }
	try {
		if(num_bufs != h->regions) {
			set_err(errbuf, errlen, "Invalid number of array items given");
			return -1;
		}
		h->hasher->update(bufs, len);
		return 0;
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

int pps_hout_get(pps_hout* h, uint8_t* out, size_t out_len, char* errbuf, size_t errlen) {
	if(!h) { set_err(errbuf, errlen, "Invalid handle"); return -1; }
	try {
		if(out_len < (size_t)h->regions * 16) {
			set_err(errbuf, errlen, "Buffer must be large enough to hold all hashes");
			return -1;
		}
		h->hasher->end();
		h->hasher->get(out);
		return 0;
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

int pps_hout_reset(pps_hout* h, char* errbuf, size_t errlen) {
	if(!h) { set_err(errbuf, errlen, "Invalid handle"); return -1; }
	try {
		h->hasher->reset();
		return 0;
	} catch(...) { set_err(errbuf, errlen, "Internal error"); return -1; }
}

void pps_hout_free(pps_hout* h) {
	if(!h) return;
	try {
		delete h->hasher;
		delete h;
	} catch(...) {}
}
