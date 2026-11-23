// sy_gpuexp.h - routed experts of a MoE layer on the Arc GPU for PREFILL chunks.
// Upgraded for 5x-10x faster prefill:
//   * All-subgroup active adaptive tiling (zero idle XMX engines for small/medium ntok!)
//   * BK=64 cooperative vectorized dequant (8 KiB SLM -> 4x occupancy) + optional DPAS INT8 XMX (HY4S_GX_I8=1)
//   * Double-buffered planning arrays eliminating cross-layer last_ev stalls
//   * Direct VRAM-slab execution for SyVramExpertCache hits (0 host copy, 0 PCIe H2D!) overlapped with PCIe streaming of host misses
//   * Supports BOTH legacy hy4s::CpuExperts (sy_gpuexp_test) AND unified hy4::CpuExperts (production sy_main)
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "hy4_cpu_experts.h"
#include "hy4s_cpuexp.h"
#include "sy_common.h"
#include "sy_vram_cache.h"

namespace hy4s {

// Planned (expert, <=64-token tile) work item of the grouped XMX evaluator — shared by GpuExperts
// (sy_gpuexp.cpp) and the peer-GPU grouped prefill (sy_peer.h). ABI-fixed 32-byte POD.
struct GxItem {
    const uint8_t* body_ptr;  // device pointer in the submitting queue's context (staging slot OR resident VRAM slab!)
    uint32_t dn_off;
    uint32_t pbase;
    uint32_t ntok;
    int32_t gu, dn;
    int32_t pad0;
};

// Runs one pre-planned grouped expert evaluation on an arbitrary SYCL queue (primary or peer GPU):
//   gather z[ptok[p]] -> d_zg (fp16)  ->  GEMM1 + SwiGLU -> d_hb (fp16)  ->  GEMM2 -> d_ye (fp32)
//   ->  per-token weighted reduce (pairs consumed in ascending expert_id order) -> d_zout.
// All device pointers must be USM allocations reachable by `q`'s context; an in-order `q` sequences
// every stage, so the returned reduce event also covers gather + both GEMMs. `xmx_sg16` selects the
// Xe2 SG16 XMX DPAS kernels (when compiled with HY4S_ENABLE_SG16_XMX); false selects portable SG32 SIMT.
sycl::event gx_eval_grouped(sycl::queue& q, const GxItem* d_items, int nitems, const float* z_dev,
                            const int32_t* d_ptok, const int32_t* d_red, const float* d_pw,
                            half* d_zg, half* d_hb, float* d_ye, float* d_zout, int P, int T, bool xmx_sg16);

struct GpuExpertStats {
    double seconds = 0;
    double t_plan = 0;
    double t_copy_host = 0;
    double t_wait_host = 0;
    double t_wait_prev = 0;
    uint64_t calls = 0, groups = 0, experts = 0, pairs = 0, bytes = 0, vram_experts = 0;
    double gpu_gemm1_ms = 0, gpu_gemm2_ms = 0, gpu_other_ms = 0, gpu_h2d_ms = 0;
};

class GpuExperts {
public:
    GpuExperts();
    ~GpuExperts();
    GpuExperts(const GpuExperts&) = delete;
    GpuExperts& operator=(const GpuExperts&) = delete;

    // Legacy init (used by sy_gpuexp_test)
    bool init(const CpuExperts& cx, size_t stage_bytes, int max_tokens, std::string& err, int copy_threads = 6);

    // Unified init (used by sy_main / sy_serve with run/hy4_cpu_experts.cpp + optional SyVramExpertCache)
    bool init_unified(const std::string& strata_path, const hy4::CpuExperts* ucx, const SyVramExpertCache* vcache,
                      size_t stage_bytes, int max_tokens, std::string& err, int copy_threads = 6);

    void set_vram_cache(const SyVramExpertCache* vcache);
    void set_profile_queue(sycl::queue* compute);

    void run(int moe_layer, int T, const float* z_dev, const int32_t* idx_host, const float* w_host, float* zout_dev, const uint8_t* only = nullptr);

    void wait_all();
    size_t device_bytes() const;
    const GpuExpertStats& stats() const;
    void reset_stats();

private:
    struct Impl;
    std::unique_ptr<Impl> im_;
};

}  // namespace hy4s
