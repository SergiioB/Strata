// sy_peer.h - Multi-GPU Peer Expert Evaluator for Hy4-Flash Intel Arc SYCL runner (--gpus 0,1 / --gpus 2 / --devices / HY4_GPUS).
// MIT License - Copyright (c) 2026 Strata contributors
//
// Architecture:
//   * Primary GPU (ids[0]) owns trunk, MLA attention, KV cache, Engram, Vision, DFly2, and hot VRAM experts (SyVramExpertCache).
//   * Secondary GPU(s) (ids[1..]) own the next hottest routed experts in Section 3 rank order across multi-slab (<= 1.4 GiB/slab)
//     USM device allocations, each with its own in-order sycl::queue.
//   * Transfers use direct P2P USM copy when ext_oneapi_can_access_peer + 16-byte roundtrip hardware probe succeed, and cleanly
//     fall back to USM host-pinned bounce buffers otherwise.
//   * Decode/Verify (T <= 16): runs in parallel with GPU 0 SyVramExpertCache AND hy4::CpuExperts, with identical SG32 Q8/FP32
//     math and ascending expert_id reduction order (bitwise reproducible, compatible with command_graph & DFly2).
//   * Prefill (T > 16): runs in parallel with GPU 0 GpuExperts (which skips peer experts via `only` mask) and accumulates partial
//     sums into zout_f on GPU 0.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "hy4_io.h"
#include "strata/artifact/strata_reader.hpp"
#include "sy_common.h"
#include "sy_gpuexp.h"
#include "sy_vram_cache.h"

namespace hy4s {

class SyPeerExperts {
public:
    static constexpr int MAX_DEC_T = SyVramExpertCache::MAX_T;
    static constexpr int MAX_DEC_PAIRS = SyVramExpertCache::MAX_PAIRS;
    static constexpr int MAX_PF_TILE_T = 256;
    static constexpr int MAX_PF_PAIRS = MAX_PF_TILE_T * vcache_detail::VC_NTOP;

    struct Meta {
        uint64_t file_off = 0;
        uint32_t dn_off = 0;
        uint32_t body_bytes = 0;
        uint8_t gu = 0, dn = 0;
    };

    struct Slab {
        uint8_t* d = nullptr;
        size_t bytes = 0;
    };

    struct PeerDev {
        int dev_idx = -1;
        std::string name;
        size_t global_mem = 0;
        size_t dev_bytes = 0;
        int n_experts = 0;
        bool p2p_ok = false;

        sycl::queue* q = nullptr;
        std::vector<Slab> slabs;

        // Decode scratch (on peer device + pinned host bounce)
        float* d_z = nullptr;           // [MAX_PF_TILE_T * 1536] (shared between decode & prefill!)
        float* d_zout = nullptr;        // [MAX_PF_TILE_T * 1536]
        float* h_zout = nullptr;        // pinned host [MAX_PF_TILE_T * 1536]
        float* d_zpeer0 = nullptr;      // on primary GPU 0 [MAX_PF_TILE_T * 1536]
        vcache_detail::VramPair* d_pairs = nullptr;  // [MAX_PF_PAIRS]
        vcache_detail::VramPair* h_pairs = nullptr;  // [MAX_PF_PAIRS]
        int32_t* d_tok_off = nullptr;   // [MAX_PF_TILE_T + 1]
        int32_t* h_tok_off = nullptr;   // [MAX_PF_TILE_T + 1]
        half* d_hmid = nullptr;         // [MAX_PF_PAIRS * 1536]
        float* d_ye = nullptr;          // [MAX_PF_PAIRS * 1536]

        // Native SG16 dedup decode scratch (same kernels as the primary SyVramExpertCache on this
        // peer's own queue; HY4S_PEER_DEC_SG32=1 keeps the legacy SG32 sg_dot_row path).
        vcache_detail::VramExpertGroup* d_grps = nullptr;  // [MAX_DEC_PAIRS]
        vcache_detail::VramExpertGroup* h_grps = nullptr;
        float* d_pair_w = nullptr;                         // [MAX_DEC_PAIRS]
        float* h_pair_w = nullptr;
        int8_t* d_zq = nullptr;                            // [MAX_DEC_T * 1536] Q8_1_G64 quants
        float* d_zd = nullptr;                             // [MAX_DEC_T * VC_NG] Q8_1_G64 scales
        int8_t* d_hq = nullptr;                            // [MAX_DEC_PAIRS * 1536]
        float* d_hd = nullptr;                             // [MAX_DEC_PAIRS * VC_NG]
        int planned_groups = 0;
        int planned_max_ntok = 0;
        bool sg16_ok = false;   // device accepts reqd_sub_group_size(16) kernels
        bool xmx_sg16 = false;  // device is Xe2-class with SG16 XMX (grouped prefill kernels)

        // Grouped XMX SG16 prefill scratch, grown lazily to the largest (T, pairs) seen
        // (HY4S_PEER_PF_LEGACY=1 restores the serialized 256-token SG32 tile path).
        float* pf_z = nullptr;       // [pf_cap_t * 1536] input activations on peer
        half* pf_zg = nullptr;       // [pf_cap_p * 1536] gathered fp16 rows
        half* pf_hb = nullptr;       // [pf_cap_p * 1536] GEMM1+SwiGLU out
        float* pf_ye = nullptr;      // [pf_cap_p * 1536] GEMM2 out
        float* pf_zout = nullptr;    // [pf_cap_t * 1536] peer partial sums on peer
        float* pf_zpeer0 = nullptr;  // [pf_cap_t * 1536] on primary GPU (p2p dst / !p2p staging)
        float* pf_h_zout = nullptr;  // [pf_cap_t * 1536] pinned host (!p2p dst)
        GxItem* pf_items_d = nullptr;  // [pf_cap_items]
        GxItem* pf_items_h = nullptr;
        int32_t* pf_ptok_d = nullptr;  // [pf_cap_p]
        int32_t* pf_ptok_h = nullptr;
        int32_t* pf_red_d = nullptr;   // [pf_cap_t * VC_NTOP]
        int32_t* pf_red_h = nullptr;
        float* pf_pw_d = nullptr;      // [pf_cap_p]
        float* pf_pw_h = nullptr;
        int pf_cap_t = 0, pf_cap_p = 0, pf_cap_items = 0;
        bool pf_launch = false;        // an XMX prefill is in flight on this peer's queue

        int planned_pairs = 0;
        int planned_T = 0;
        bool active_launch = false;
        sycl::event done_ev;
    };

    ~SyPeerExperts() { release(); }

    bool active() const { return !peers_.empty() && total_experts_ > 0; }
    int num_peers() const { return (int)peers_.size(); }
    int total_experts() const { return total_experts_; }
    size_t total_device_bytes() const { return total_dev_bytes_; }
    uint64_t hits() const { return hits_; }
    void reset_stats() { hits_ = 0; }
    void set_q8_act(bool on) { q8_act_ = on; }
    bool has_expert(int L, int e) const {
        if ((unsigned)L >= (unsigned)nlayers_ || (unsigned)e >= 512u) return false;
        return owner_[(size_t)L * 512 + e] >= 0;
    }

    std::string describe() const {
        if (peers_.empty()) return "";
        std::string s;
        for (size_t i = 0; i < peers_.size(); ++i) {
            if (!s.empty()) s += ", ";
            char buf[256];
            snprintf(buf, sizeof buf, "GPU%d(%s, %d exp, %.2f GiB, %s)",
                     peers_[i].dev_idx, peers_[i].name.c_str(), peers_[i].n_experts,
                     peers_[i].dev_bytes / 1073741824.0, peers_[i].p2p_ok ? "p2p" : "pinned");
            s += buf;
        }
        return s;
    }

    void release() {
        for (auto& p : peers_) {
            if (p.q) {
                try { p.q->wait(); } catch (...) {}
                for (auto& sl : p.slabs) if (sl.d) sycl::free(sl.d, *p.q);
                if (p.d_z) sycl::free(p.d_z, *p.q);
                if (p.d_zout) sycl::free(p.d_zout, *p.q);
                if (p.h_zout) sycl::free(p.h_zout, *p.q);
                if (p.d_pairs) sycl::free(p.d_pairs, *p.q);
                if (p.h_pairs) sycl::free(p.h_pairs, *p.q);
                if (p.d_tok_off) sycl::free(p.d_tok_off, *p.q);
                if (p.h_tok_off) sycl::free(p.h_tok_off, *p.q);
                if (p.d_hmid) sycl::free(p.d_hmid, *p.q);
                if (p.d_ye) sycl::free(p.d_ye, *p.q);
                if (p.d_grps) sycl::free(p.d_grps, *p.q);
                if (p.h_grps) sycl::free(p.h_grps, *p.q);
                if (p.d_pair_w) sycl::free(p.d_pair_w, *p.q);
                if (p.h_pair_w) sycl::free(p.h_pair_w, *p.q);
                if (p.d_zq) sycl::free(p.d_zq, *p.q);
                if (p.d_zd) sycl::free(p.d_zd, *p.q);
                if (p.d_hq) sycl::free(p.d_hq, *p.q);
                if (p.d_hd) sycl::free(p.d_hd, *p.q);
                if (p.pf_z) sycl::free(p.pf_z, *p.q);
                if (p.pf_zg) sycl::free(p.pf_zg, *p.q);
                if (p.pf_hb) sycl::free(p.pf_hb, *p.q);
                if (p.pf_ye) sycl::free(p.pf_ye, *p.q);
                if (p.pf_zout) sycl::free(p.pf_zout, *p.q);
                if (p.pf_h_zout) sycl::free(p.pf_h_zout, *p.q);
                if (p.pf_items_d) sycl::free(p.pf_items_d, *p.q);
                if (p.pf_items_h) sycl::free(p.pf_items_h, *p.q);
                if (p.pf_ptok_d) sycl::free(p.pf_ptok_d, *p.q);
                if (p.pf_ptok_h) sycl::free(p.pf_ptok_h, *p.q);
                if (p.pf_red_d) sycl::free(p.pf_red_d, *p.q);
                if (p.pf_red_h) sycl::free(p.pf_red_h, *p.q);
                if (p.pf_pw_d) sycl::free(p.pf_pw_d, *p.q);
                if (p.pf_pw_h) sycl::free(p.pf_pw_h, *p.q);
                delete p.q;
                p.q = nullptr;
            }
            if (p.d_zpeer0) { dfree(p.d_zpeer0); p.d_zpeer0 = nullptr; }
            if (p.pf_zpeer0) { dfree(p.pf_zpeer0); p.pf_zpeer0 = nullptr; }
            p.slabs.clear();
            p.pf_cap_t = p.pf_cap_p = p.pf_cap_items = 0;
            p.pf_launch = false;
        }
        peers_.clear();
        if (h_pf_z_) { sycl::free(h_pf_z_, Q()); h_pf_z_ = nullptr; }
        h_pf_z_cap_ = 0;
        owner_.clear();
        d_ptr_.clear();
        meta_.clear();
        total_experts_ = 0;
        total_dev_bytes_ = 0;
        nlayers_ = 0;
    }

    // Probe optional P2P support between primary device and peer device with a 16-byte hardware roundtrip check.
    static bool probe_p2p(sycl::queue& q0, sycl::queue& qp) {
        const char* force_pinned = std::getenv("HY4S_PEER_PINNED");
        if (force_pinned && force_pinned[0] == '1') return false;
#if defined(SYCL_EXT_ONEAPI_PEER_ACCESS)
        try {
            auto d0 = q0.get_device();
            auto dp = qp.get_device();
            if (!d0.ext_oneapi_can_access_peer(dp, sycl::ext::oneapi::peer_access::access_supported) ||
                !dp.ext_oneapi_can_access_peer(d0, sycl::ext::oneapi::peer_access::access_supported)) {
                return false;
            }
            d0.ext_oneapi_enable_peer_access(dp);
            dp.ext_oneapi_enable_peer_access(d0);
            uint32_t h_tx[4] = {0x12345678u, 0xCAFEBABEu, 0xA5A55A5Au, 0xDEADBEEFu};
            uint32_t h_rx01[4] = {0, 0, 0, 0};
            uint32_t h_rx10[4] = {0, 0, 0, 0};
            uint32_t* p0 = sycl::malloc_device<uint32_t>(4, q0);
            uint32_t* p1 = sycl::malloc_device<uint32_t>(4, qp);
            bool ok = false;
            if (p0 && p1) {
                q0.memcpy(p0, h_tx, sizeof(h_tx)).wait();
                qp.memset(p1, 0, sizeof(h_tx)).wait();
                qp.memcpy(p1, p0, sizeof(h_tx)).wait();
                qp.memcpy(h_rx01, p1, sizeof(h_tx)).wait();
                q0.memset(p0, 0, sizeof(h_tx)).wait();
                q0.memcpy(p0, p1, sizeof(h_tx)).wait();
                q0.memcpy(h_rx10, p0, sizeof(h_tx)).wait();
                ok = (std::memcmp(h_tx, h_rx01, sizeof(h_tx)) == 0 &&
                      std::memcmp(h_tx, h_rx10, sizeof(h_tx)) == 0);
            }
            if (p0) sycl::free(p0, q0);
            if (p1) sycl::free(p1, qp);
            return ok;
        } catch (...) {
            return false;
        }
#else
        (void)q0; (void)qp;
        return false;
#endif
    }

    // Initialize peer GPU expert storage for `peer_devs` (indices into enum_gpu_devices()), skipping experts already in `vram0_mask`.
    // `per_peer_budget_bytes`: 0 => auto (90% of peer global_mem minus workspace).
    bool init(const std::string& strata_path, const std::vector<int>& peer_devs,
              const std::vector<std::vector<bool>>& vram0_mask,
              size_t per_peer_budget_bytes, std::string& err, size_t slab_mb = 1400) {
        release();
        if (peer_devs.empty()) return true;

        auto all_devs = enum_gpu_devices();
        if (all_devs.size() < 2) {
            fprintf(stderr, "[peer-sycl] only %zu GPU(s) visible; multi-GPU peer experts disabled\n", all_devs.size());
            return true;
        }

        strata::artifact::StrataFileReader rd;
        if (!rd.open(strata_path, err)) return false;
        const auto& h = rd.header();
        const auto& dir = rd.expert_directory();
        if (dir.empty() || (dir.size() % 512) != 0) {
            err = "peer-sycl: invalid expert directory";
            return false;
        }
        nlayers_ = (int)(dir.size() / 512);
        meta_.assign(dir.size(), Meta{});
        owner_.assign(dir.size(), -1);
        d_ptr_.assign(dir.size(), nullptr);
        std::vector<uint16_t> rank_eid(dir.size(), 0);
        uint32_t max_body = 0;
        for (size_t i = 0; i < dir.size(); ++i) {
            const int L = (int)(i / 512);
            const auto& d = dir[i];
            Meta& m = meta_[(size_t)L * 512 + d.expert_id];
            m.file_off = h.sections[4].file_offset + d.body_offset;
            m.dn_off = d.dn_offset;
            m.body_bytes = d.body_bytes;
            m.gu = d.gu_code_id;
            m.dn = d.dn_code_id;
            rank_eid[i] = d.expert_id;
            max_body = std::max(max_body, d.body_bytes);
        }
        const size_t slot_bytes = ((size_t)max_body + 4095ull) & ~4095ull;
        if (slot_bytes == 0) return true;

        // Scratch memory per peer (~42 MiB for up to MAX_PF_TILE_T=256 prefill tile + decode)
        const size_t ws_bytes =
            (size_t)MAX_PF_TILE_T * vcache_detail::VC_K * sizeof(float) * 2 +
            (size_t)MAX_PF_PAIRS * (sizeof(vcache_detail::VramPair) + vcache_detail::VC_K * (sizeof(half) + sizeof(float))) +
            (size_t)(MAX_PF_TILE_T + 1) * sizeof(int32_t) + (64ull << 20);

        // Create queues & query budgets for each valid peer device
        std::vector<size_t> peer_max_slots;
        for (int pidx : peer_devs) {
            if (pidx < 0 || pidx >= (int)all_devs.size() || pidx == primary_gpu_index()) continue;
            PeerDev pd;
            pd.dev_idx = pidx;
            auto dev = all_devs[pidx];
            pd.name = dev.get_info<sycl::info::device::name>();
            pd.global_mem = dev.get_info<sycl::info::device::global_mem_size>();
            try {
                pd.q = new sycl::queue(dev, sycl::property_list{sycl::property::queue::in_order()});
            } catch (const sycl::exception& e) {
                fprintf(stderr, "[peer-sycl] skipping GPU%d (%s): queue creation failed (%s)\n", pidx, pd.name.c_str(), e.what());
                continue;
            }
            size_t budget = per_peer_budget_bytes;
            if (budget == 0) {
                const size_t safe_cap = (size_t)((double)pd.global_mem * 0.90);
                budget = (safe_cap > ws_bytes + (256ull << 20)) ? (safe_cap - ws_bytes - (256ull << 20)) : 0;
            }
            const size_t nslots = budget / slot_bytes;
            if (nslots == 0) {
                delete pd.q;
                continue;
            }
            pd.p2p_ok = probe_p2p(Q(), *pd.q);
            {
                const DevCaps pcaps = probe_dev_caps(pd.q->get_device());
                pd.sg16_ok = pcaps.has_sg16;
                pd.xmx_sg16 = pcaps.has_xmx_sg16;  // Xe2 (BMG) grouped-prefill kernel variant; non-XMX peers use grouped SIMT
            }
            peers_.push_back(std::move(pd));
            peer_max_slots.push_back(nslots);
        }
        if (peers_.empty()) return true;

        // Partition non-GPU0 experts in Section 3 rank-major order across peers (round-robin per rank across available peers
        // so every layer gets balanced multi-GPU parallelism across all peers!)
        std::vector<std::vector<std::pair<int, int>>> assign(peers_.size());
        size_t rr = 0;
        for (int rk = 0; rk < 512; ++rk) {
            for (int L = 0; L < nlayers_; ++L) {
                const int e = rank_eid[(size_t)L * 512 + rk];
                if (!vram0_mask.empty() && L < (int)vram0_mask.size() && e < (int)vram0_mask[L].size() && vram0_mask[L][e]) {
                    continue;  // already resident on GPU 0
                }
                int chosen = -1;
                for (size_t k = 0; k < peers_.size(); ++k) {
                    const size_t pi = (rr + k) % peers_.size();
                    if (assign[pi].size() < peer_max_slots[pi]) {
                        chosen = (int)pi;
                        rr = (pi + 1) % peers_.size();
                        break;
                    }
                }
                if (chosen < 0) break;  // all peers full
                assign[chosen].push_back({L, e});
            }
        }

        hy4::AlignedFile af;
        if (!af.open(strata_path, false)) {
            err = "peer-sycl: cannot open " + strata_path;
            return false;
        }
        const double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        const size_t slab_max = (slab_mb << 20);
        const size_t slots_per_slab_max = std::max<size_t>(1, slab_max / slot_bytes);

        for (size_t pi = 0; pi < peers_.size(); ++pi) {
            PeerDev& pd = peers_[pi];
            const auto& list = assign[pi];
            if (list.empty()) continue;

            // Allocate peer scratch & primary GPU 0 receive buffer
            pd.d_z = sycl::malloc_device<float>((size_t)MAX_PF_TILE_T * vcache_detail::VC_K, *pd.q);
            pd.d_zout = sycl::malloc_device<float>((size_t)MAX_PF_TILE_T * vcache_detail::VC_K, *pd.q);
            pd.h_zout = sycl::malloc_host<float>((size_t)MAX_PF_TILE_T * vcache_detail::VC_K, *pd.q);
            pd.d_pairs = sycl::malloc_device<vcache_detail::VramPair>(MAX_PF_PAIRS, *pd.q);
            pd.h_pairs = sycl::malloc_host<vcache_detail::VramPair>(MAX_PF_PAIRS, *pd.q);
            pd.d_tok_off = sycl::malloc_device<int32_t>(MAX_PF_TILE_T + 1, *pd.q);
            pd.h_tok_off = sycl::malloc_host<int32_t>(MAX_PF_TILE_T + 1, *pd.q);
            pd.d_hmid = sycl::malloc_device<half>((size_t)MAX_PF_PAIRS * vcache_detail::VC_K, *pd.q);
            pd.d_ye = sycl::malloc_device<float>((size_t)MAX_PF_PAIRS * vcache_detail::VC_K, *pd.q);
            pd.d_zpeer0 = dalloc<float>((size_t)MAX_PF_TILE_T * vcache_detail::VC_K);
            // SG16 dedup decode scratch (native SyVramExpertCache kernels on this peer's queue)
            if (pd.sg16_ok) {
                pd.d_grps = sycl::malloc_device<vcache_detail::VramExpertGroup>(MAX_DEC_PAIRS, *pd.q);
                pd.h_grps = sycl::malloc_host<vcache_detail::VramExpertGroup>(MAX_DEC_PAIRS, *pd.q);
                pd.d_pair_w = sycl::malloc_device<float>(MAX_DEC_PAIRS, *pd.q);
                pd.h_pair_w = sycl::malloc_host<float>(MAX_DEC_PAIRS, *pd.q);
                pd.d_zq = sycl::malloc_device<int8_t>((size_t)MAX_DEC_T * vcache_detail::VC_K, *pd.q);
                pd.d_zd = sycl::malloc_device<float>((size_t)MAX_DEC_T * vcache_detail::VC_NG, *pd.q);
                pd.d_hq = sycl::malloc_device<int8_t>((size_t)MAX_DEC_PAIRS * vcache_detail::VC_K, *pd.q);
                pd.d_hd = sycl::malloc_device<float>((size_t)MAX_DEC_PAIRS * vcache_detail::VC_NG, *pd.q);
            }
            if (!pd.d_z || !pd.d_zout || !pd.h_zout || !pd.d_pairs || !pd.h_pairs ||
                !pd.d_tok_off || !pd.h_tok_off || !pd.d_hmid || !pd.d_ye ||
                (pd.sg16_ok && (!pd.d_grps || !pd.h_grps || !pd.d_pair_w || !pd.h_pair_w ||
                                !pd.d_zq || !pd.d_zd || !pd.d_hq || !pd.d_hd))) {
                af.close();
                err = "peer-sycl: scratch allocation failed on GPU " + std::to_string(pd.dev_idx);
                return false;
            }
            uint8_t* h_stage = sycl::aligned_alloc_host<uint8_t>(4096, slot_bytes, *pd.q);
            if (!h_stage) {
                af.close();
                err = "peer-sycl: staging allocation failed";
                return false;
            }

            size_t idx = 0;
            while (idx < list.size()) {
                const size_t ns = std::min(list.size() - idx, slots_per_slab_max);
                const size_t sb = ns * slot_bytes;
                uint8_t* slab_ptr = sycl::malloc_device<uint8_t>(sb, *pd.q);
                if (!slab_ptr) break;
                pd.slabs.push_back(Slab{slab_ptr, sb});
                pd.dev_bytes += sb;
                total_dev_bytes_ += sb;

                for (size_t k = 0; k < ns; ++k) {
                    const int L = list[idx + k].first;
                    const int e = list[idx + k].second;
                    const Meta& m = meta_[(size_t)L * 512 + e];
                    if (af.read_at(m.file_off, h_stage, m.body_bytes) < m.body_bytes) {
                        sycl::free(h_stage, *pd.q);
                        af.close();
                        err = "peer-sycl: short read";
                        return false;
                    }
                    uint8_t* dst = slab_ptr + k * slot_bytes;
                    pd.q->memcpy(dst, h_stage, m.body_bytes).wait();
                    owner_[(size_t)L * 512 + e] = (int)pi;
                    d_ptr_[(size_t)L * 512 + e] = dst;
                    ++pd.n_experts;
                    ++total_experts_;
                }
                idx += ns;
            }
            sycl::free(h_stage, *pd.q);
        }
        af.close();

        if (total_experts_ > 0) {
            h_pf_z_ = halloc<float>((size_t)MAX_PF_TILE_T * vcache_detail::VC_K);
            h_pf_z_cap_ = h_pf_z_ ? (size_t)MAX_PF_TILE_T * vcache_detail::VC_K : 0;
        }

        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - t0;
        fprintf(stderr, "[peer-sycl] loaded %d peer experts across %zu peer GPU(s) (%.2f GiB total) in %.1f s: %s\n",
                total_experts_, peers_.size(), total_dev_bytes_ / 1073741824.0, dt, describe().c_str());
        return true;
    }

    void mark_peer_mask(std::vector<std::vector<bool>>& mask) const {
        if (!active()) return;
        if ((int)mask.size() < nlayers_) mask.assign(nlayers_, std::vector<bool>(512, false));
        for (int L = 0; L < nlayers_; ++L)
            for (int e = 0; e < 512; ++e)
                if (owner_[(size_t)L * 512 + e] >= 0) mask[L][e] = true;
    }

    void build_non_peer_mask(int L, uint8_t only[512]) const {
        for (int e = 0; e < 512; ++e) {
            only[e] = (owner_[(size_t)L * 512 + e] >= 0) ? 0 : 1;
        }
    }

    // ---- Decode/Verify path (T <= 16): claim peer hits from `use_cpu`, launch peers concurrently, gather partial sums ----
    int plan_decode(int L, int T, const int32_t* idx_host, const float* w_host, uint8_t* use_cpu) {
        for (auto& pd : peers_) {
            pd.planned_pairs = 0;
            pd.planned_T = T;
            pd.active_launch = false;
        }
        if (!active() || T <= 0 || T > MAX_DEC_T) return 0;

        int total_hits = 0;
        for (size_t pi = 0; pi < peers_.size(); ++pi) {
            PeerDev& pd = peers_[pi];
            int npairs = 0;
            int ngroups = 0, max_ntok = 0;
            int16_t exp_grp[512];
            for (int i = 0; i < 512; ++i) exp_grp[i] = -1;
            pd.h_tok_off[0] = 0;
            for (int t = 0; t < T; ++t) {
                int hit_k[vcache_detail::VC_NTOP], nh = 0;
                for (int k = 0; k < vcache_detail::VC_NTOP; ++k) {
                    const int slot = t * vcache_detail::VC_NTOP + k;
                    if (use_cpu && !use_cpu[slot]) continue;  // already claimed by GPU 0
                    const int e = idx_host[slot];
                    if ((unsigned)e < 512u && owner_[(size_t)L * 512 + e] == (int)pi) {
                        if (use_cpu) use_cpu[slot] = 0;
                        ++hits_;
                        ++total_hits;
                        int j = nh++;
                        while (j > 0 && idx_host[t * vcache_detail::VC_NTOP + hit_k[j - 1]] > e) {
                            hit_k[j] = hit_k[j - 1];
                            --j;
                        }
                        hit_k[j] = k;
                    }
                }
                for (int a = 0; a < nh; ++a) {
                    const int k = hit_k[a];
                    const int e = idx_host[t * vcache_detail::VC_NTOP + k];
                    const Meta& m = meta_[(size_t)L * 512 + e];
                    const int p = npairs++;
                    vcache_detail::VramPair& vp = pd.h_pairs[p];
                    vp.body = d_ptr_[(size_t)L * 512 + e];
                    vp.dn_off = m.dn_off;
                    vp.tok = t;
                    vp.gu = m.gu;
                    vp.dn = m.dn;
                    vp.weight = w_host[t * vcache_detail::VC_NTOP + k];
                    // Dedup groups for the SG16 path: same O(npairs) expert->group map as SyVramExpertCache.
                    // Pairs stay in ascending expert_id order per token, so reduction order is unchanged.
                    if (pd.h_grps) {
                        pd.h_pair_w[p] = vp.weight;
                        int gi = exp_grp[e];
                        if (gi < 0 || pd.h_grps[gi].ntok >= vcache_detail::VC_GRP_MAX_T) {
                            gi = ngroups++;
                            vcache_detail::VramExpertGroup& vg = pd.h_grps[gi];
                            vg.body = vp.body;
                            vg.dn_off = m.dn_off;
                            vg.gu = m.gu;
                            vg.dn = m.dn;
                            vg.ntok = 0;
                            vg.pad0 = 0;
                            exp_grp[e] = (int16_t)gi;
                        }
                        vcache_detail::VramExpertGroup& vg = pd.h_grps[gi];
                        const int slot = vg.ntok++;
                        vg.tok[slot] = t;
                        vg.pair_idx[slot] = p;
                        if (vg.ntok > max_ntok) max_ntok = vg.ntok;
                    }
                }
                pd.h_tok_off[t + 1] = npairs;
            }
            pd.planned_pairs = npairs;
            pd.planned_groups = ngroups;
            pd.planned_max_ntok = max_ntok;
        }
        return total_hits;
    }

    void launch_decode(const float* z_dev_g0, const float* z_host_pinned) {
        if (!active()) return;
        // Self-defend primary-queue ordering only when a peer will read z_dev_g0/d_zpeer0 over P2P:
        // the peer queue cannot see Q()'s in-order guarantees. With the pinned-host bounce the copy
        // source (z_host_pinned) was already produced before run_routed_experts was entered (evw.wait),
        // and all !p2p folding runs on Q() itself, so a full drain here only stalls the host 41x/token.
        bool any_p2p = false;
        for (const auto& pd : peers_) if (pd.planned_pairs > 0 && pd.p2p_ok) any_p2p = true;
        if (any_p2p || env_dec_wait()) { try { Q().wait(); } catch (...) {} }
        for (auto& pd : peers_) {
            if (pd.planned_pairs <= 0) continue;
            const int T = pd.planned_T;
            const size_t nz = (size_t)T * vcache_detail::VC_K;
            sycl::queue& pq = *pd.q;
            if (pd.p2p_ok && z_dev_g0) {
                pq.memcpy(pd.d_z, z_dev_g0, nz * sizeof(float));
            } else {
                pq.memcpy(pd.d_z, z_host_pinned, nz * sizeof(float));
            }
            // Default: native SG16 + dedup + Q8_1_G64 pre-quantized activations (same kernels as the
            // primary SyVramExpertCache, identical math). HY4S_PEER_DEC_SG32=1 restores the legacy
            // per-pair SG32 evaluator; non-SG16-capable peers keep it automatically.
            if (pd.sg16_ok && !env_dec_sg32()) enqueue_peer_eval_sg16(pd, T);
            else enqueue_peer_eval(pd, T, pd.planned_pairs);
            if (pd.p2p_ok) {
                pd.done_ev = pq.memcpy(pd.d_zpeer0, pd.d_zout, nz * sizeof(float));
            } else {
                pd.done_ev = pq.memcpy(pd.h_zout, pd.d_zout, nz * sizeof(float));
            }
            pd.active_launch = true;
        }
    }

    // Wait for all launched peer GPUs and accumulate partial sums into zout_f_g0 on primary GPU 0.
    void finish_decode(int T, float* zout_f_g0, bool& zout_initialized) {
        if (!active()) return;
        const size_t nz = (size_t)T * vcache_detail::VC_K;
        for (auto& pd : peers_) {
            if (!pd.active_launch) continue;
            pd.done_ev.wait();
            pd.active_launch = false;
            accumulate_peer_zout(pd, pd.d_zpeer0, pd.h_zout, nz, zout_f_g0, zout_initialized);
        }
    }

    // ---- Prefill path (T > 16): evaluate all peer-resident experts across T tokens (tiled by MAX_PF_TILE_T) ----
    // Note: called alongside GpuExperts::run(..., only_non_peer); accumulates partial sums into zout_f_g0 after gx->run!
    void run_prefill_tile_and_add(int L, int T, const float* z_dev_g0, const int32_t* idx_host, const float* w_host, float* zout_f_g0) {
        if (!active() || T <= 0) return;
        bool any_p2p = true;
        for (const auto& pd : peers_) if (!pd.p2p_ok) any_p2p = false;

        for (int t0 = 0; t0 < T; t0 += MAX_PF_TILE_T) {
            const int tn = std::min(MAX_PF_TILE_T, T - t0);
            const size_t nz = (size_t)tn * vcache_detail::VC_K;
            const float* z_tile_g0 = z_dev_g0 + (size_t)t0 * vcache_detail::VC_K;
            float* zout_tile_g0 = zout_f_g0 + (size_t)t0 * vcache_detail::VC_K;

            int tile_hits = 0;
            for (size_t pi = 0; pi < peers_.size(); ++pi) {
                PeerDev& pd = peers_[pi];
                int npairs = 0;
                pd.h_tok_off[0] = 0;
                for (int t = 0; t < tn; ++t) {
                    const int gt = t0 + t;
                    int hit_k[vcache_detail::VC_NTOP], nh = 0;
                    for (int k = 0; k < vcache_detail::VC_NTOP; ++k) {
                        const int e = idx_host[gt * vcache_detail::VC_NTOP + k];
                        if ((unsigned)e < 512u && owner_[(size_t)L * 512 + e] == (int)pi) {
                            int j = nh++;
                            while (j > 0 && idx_host[gt * vcache_detail::VC_NTOP + hit_k[j - 1]] > e) {
                                hit_k[j] = hit_k[j - 1];
                                --j;
                            }
                            hit_k[j] = k;
                        }
                    }
                    for (int a = 0; a < nh; ++a) {
                        const int k = hit_k[a];
                        const int e = idx_host[gt * vcache_detail::VC_NTOP + k];
                        const Meta& m = meta_[(size_t)L * 512 + e];
                        vcache_detail::VramPair& vp = pd.h_pairs[npairs++];
                        vp.body = d_ptr_[(size_t)L * 512 + e];
                        vp.dn_off = m.dn_off;
                        vp.tok = t;
                        vp.gu = m.gu;
                        vp.dn = m.dn;
                        vp.weight = w_host[gt * vcache_detail::VC_NTOP + k];
                    }
                    pd.h_tok_off[t + 1] = npairs;
                }
                pd.planned_pairs = npairs;
                pd.planned_T = tn;
                pd.active_launch = false;
                tile_hits += npairs;
            }
            if (tile_hits == 0) continue;

            if (!any_p2p) {
                Q().memcpy(h_pf_z_, z_tile_g0, nz * sizeof(float)).wait();
            } else {
                Q().wait();
            }

            // Launch all peers in parallel for this tile
            for (auto& pd : peers_) {
                if (pd.planned_pairs <= 0) continue;
                sycl::queue& pq = *pd.q;
                if (pd.p2p_ok) {
                    pq.memcpy(pd.d_z, z_tile_g0, nz * sizeof(float));
                } else {
                    pq.memcpy(pd.d_z, h_pf_z_, nz * sizeof(float));
                }
                enqueue_peer_eval(pd, tn, pd.planned_pairs);
                if (pd.p2p_ok) {
                    pd.done_ev = pq.memcpy(pd.d_zpeer0, pd.d_zout, nz * sizeof(float));
                } else {
                    pd.done_ev = pq.memcpy(pd.h_zout, pd.d_zout, nz * sizeof(float));
                }
                pd.active_launch = true;
            }
            // Wait & accumulate into zout_tile_g0
            bool inited = true;
            finish_decode(tn, zout_tile_g0, inited);
        }
    }

    // ---- Grouped prefill (default; HY4S_PEER_PF_LEGACY=1 keeps the serialized 256-token SG32 tiles) ----
    // One launch gathers all T tokens' activations, evaluates every peer-resident expert ONCE across
    // all its tokens with the same grouped gather/GEMM/reduce kernels GpuExperts runs on the primary
    // GPU (Xe2 SG16 XMX DPAS when `xmx_sg16`, grouped SIMT otherwise), and writes a T*1536 partial-sum
    // tensor — zero 256-token tile syncs, zero host-RAM weight streaming.
    bool prefill_grouped() const { return active() && !env_pf_legacy(); }

    // Plan + enqueue on every hit peer; returns false if scratch could not be allocated (caller then
    // falls back to run_prefill_tile_and_add). Launch BEFORE the primary GpuExperts::run(...) so peer
    // GPUs and the primary GPU execute in true hardware overlap.
    bool launch_prefill_grouped(int L, int T, const float* z_dev_g0, const int32_t* idx_host, const float* w_host) {
        if (!active() || T <= 0) return false;
        const size_t nz = (size_t)T * vcache_detail::VC_K;
        bool any = false, need_bounce = false;
        for (auto& pd : peers_) pd.pf_launch = false;

        // Ensure the shared !p2p host bounce fits T tokens before any peer copy uses it.
        for (auto& pd : peers_) {
            int32_t cnt[512];
            std::memset(cnt, 0, sizeof(cnt));
            for (int i = 0; i < T * vcache_detail::VC_NTOP; ++i) {
                const int e = idx_host[i];
                if ((unsigned)e < 512u && owner_[(size_t)L * 512 + e] == (int)(&pd - &peers_[0])) cnt[e]++;
            }
            int P = 0, nitems = 0;
            for (int e = 0; e < 512; ++e) {
                P += cnt[e];
                nitems += (cnt[e] + 63) / 64;
            }
            if (P == 0) continue;
            any = true;
            if (!pd.p2p_ok) need_bounce = true;
            if (!ensure_pf_capacity(pd, T, P, nitems)) return false;
        }
        if (!any) return true;  // nothing for the peers this layer; finish_prefill_grouped is a no-op
        if (need_bounce) {
            if (!ensure_h_pf_z(nz)) return false;
            Q().memcpy(h_pf_z_, z_dev_g0, nz * sizeof(float)).wait();  // orders all prior Q() work too
        } else {
            Q().wait();  // z_dev_g0 must be complete before peers read it via p2p
        }

        pf_pos_.assign((size_t)T * vcache_detail::VC_NTOP, -1);
        for (size_t pi = 0; pi < peers_.size(); ++pi) {
            PeerDev& pd = peers_[pi];
            int32_t cnt[512], st[513], cur[512];
            std::memset(cnt, 0, sizeof(cnt));
            for (int i = 0; i < T * vcache_detail::VC_NTOP; ++i) {
                const int e = idx_host[i];
                if ((unsigned)e < 512u && owner_[(size_t)L * 512 + e] == (int)pi) cnt[e]++;
            }
            int acc = 0;
            for (int e = 0; e < 512; ++e) { st[e] = acc; cur[e] = acc; acc += cnt[e]; }
            st[512] = acc;
            const int P = acc;
            if (P == 0) continue;

            // Pair table: pairs grouped by ascending expert (identical layout to GpuExperts::run).
            std::fill(pf_pos_.begin(), pf_pos_.end(), -1);
            for (int i = 0; i < T * vcache_detail::VC_NTOP; ++i) {
                const int e = idx_host[i];
                if ((unsigned)e < 512u && owner_[(size_t)L * 512 + e] == (int)pi) {
                    const int p = cur[e]++;
                    pd.pf_ptok_h[p] = i / vcache_detail::VC_NTOP;
                    pd.pf_pw_h[p] = w_host[i];
                    pf_pos_[i] = p;
                }
            }
            // Per-token reduction list in ascending expert_id order (bitwise-identical reduce order).
            for (int t = 0; t < T; ++t) {
                int ks[vcache_detail::VC_NTOP], n = 0;
                for (int k = 0; k < vcache_detail::VC_NTOP; ++k)
                    if (pf_pos_[(size_t)t * vcache_detail::VC_NTOP + k] >= 0) {
                        int j = n++;
                        while (j > 0 && idx_host[t * vcache_detail::VC_NTOP + ks[j - 1]] > idx_host[t * vcache_detail::VC_NTOP + k]) {
                            ks[j] = ks[j - 1];
                            --j;
                        }
                        ks[j] = k;
                    }
                for (int a = 0; a < vcache_detail::VC_NTOP; ++a)
                    pd.pf_red_h[(size_t)t * vcache_detail::VC_NTOP + a] =
                        a < n ? pf_pos_[(size_t)t * vcache_detail::VC_NTOP + ks[a]] : -1;
            }
            // Work items: one per (expert, <=64-token tile), expert weights read ONCE across all T.
            int nitems = 0;
            for (int e = 0; e < 512; ++e) {
                for (int tt = 0; tt < cnt[e]; tt += 64) {
                    const Meta& m = meta_[(size_t)L * 512 + e];
                    GxItem& it = pd.pf_items_h[nitems++];
                    it.body_ptr = d_ptr_[(size_t)L * 512 + e];
                    it.dn_off = m.dn_off;
                    it.pbase = (uint32_t)(st[e] + tt);
                    it.ntok = (uint32_t)std::min(64, cnt[e] - tt);
                    it.gu = m.gu;
                    it.dn = m.dn;
                    it.pad0 = 0;
                }
            }

            sycl::queue& pq = *pd.q;
            pq.memcpy(pd.pf_items_d, pd.pf_items_h, (size_t)nitems * sizeof(GxItem));
            pq.memcpy(pd.pf_ptok_d, pd.pf_ptok_h, (size_t)P * sizeof(int32_t));
            pq.memcpy(pd.pf_pw_d, pd.pf_pw_h, (size_t)P * sizeof(float));
            pq.memcpy(pd.pf_red_d, pd.pf_red_h, (size_t)T * vcache_detail::VC_NTOP * sizeof(int32_t));
            if (pd.p2p_ok) {
                pq.memcpy(pd.pf_z, z_dev_g0, nz * sizeof(float));
            } else {
                pq.memcpy(pd.pf_z, h_pf_z_, nz * sizeof(float));
            }
            gx_eval_grouped(pq, pd.pf_items_d, nitems, pd.pf_z, pd.pf_ptok_d, pd.pf_red_d, pd.pf_pw_d,
                            pd.pf_zg, pd.pf_hb, pd.pf_ye, pd.pf_zout, P, T, pd.xmx_sg16);
            if (pd.p2p_ok) {
                pd.done_ev = pq.memcpy(pd.pf_zpeer0, pd.pf_zout, nz * sizeof(float));
            } else {
                pd.done_ev = pq.memcpy(pd.pf_h_zout, pd.pf_zout, nz * sizeof(float));
            }
            pd.pf_launch = true;
        }
        return true;
    }

    // Wait for launched grouped prefills and fold each peer's partial sums into zout_f_g0 on the primary GPU.
    void finish_prefill_grouped(int T, float* zout_f_g0, bool& zout_initialized) {
        if (!active()) return;
        const size_t nz = (size_t)T * vcache_detail::VC_K;
        for (auto& pd : peers_) {
            if (!pd.pf_launch) continue;
            pd.done_ev.wait();
            pd.pf_launch = false;
            accumulate_peer_zout(pd, pd.pf_zpeer0, pd.pf_h_zout, nz, zout_f_g0, zout_initialized);
        }
    }

private:
    // Fold one peer's completed partial sums into zout_f_g0: device-resident `d_src0` (primary GPU)
    // when p2p is up, pinned host `h_src` otherwise. First contributor may overwrite the accumulator.
    void accumulate_peer_zout(PeerDev& pd, float* d_src0, const float* h_src, size_t nz, float* zout_f_g0, bool& zout_initialized) {
        if (pd.p2p_ok) {
            if (!zout_initialized) {
                d2d(zout_f_g0, d_src0, nz * sizeof(float));
                zout_initialized = true;
            } else {
                SyVramExpertCache::add_zout(zout_f_g0, d_src0, nz);
            }
        } else {
            if (!zout_initialized) {
                h2d(zout_f_g0, h_src, nz * sizeof(float));
                zout_initialized = true;
            } else {
                h2d(d_src0, h_src, nz * sizeof(float));
                SyVramExpertCache::add_zout(zout_f_g0, d_src0, nz);
            }
        }
    }

    // Grow the grouped-prefill scratch of peer `pd` to cover (T tokens, P pairs, nitems work items).
    // Returns false when the peer device refuses the allocation (caller falls back to legacy tiles).
    bool ensure_pf_capacity(PeerDev& pd, int T, int P, int nitems) {
        if (pd.pf_cap_t >= T && pd.pf_cap_p >= P && pd.pf_cap_items >= nitems) return true;
        // Rare growth path: drain before freeing buffers a previous launch may still be reading
        // (peer queue covers device buffers; Q() covers pf_zpeer0 staged accumulates).
        try { pd.q->wait(); Q().wait(); } catch (...) {}
        try {
            if (pd.pf_cap_t < T) {
                if (pd.pf_z) { sycl::free(pd.pf_z, *pd.q); pd.pf_z = nullptr; }
                if (pd.pf_zout) { sycl::free(pd.pf_zout, *pd.q); pd.pf_zout = nullptr; }
                if (pd.pf_h_zout) { sycl::free(pd.pf_h_zout, *pd.q); pd.pf_h_zout = nullptr; }
                if (pd.pf_red_d) { sycl::free(pd.pf_red_d, *pd.q); pd.pf_red_d = nullptr; }
                if (pd.pf_red_h) { sycl::free(pd.pf_red_h, *pd.q); pd.pf_red_h = nullptr; }
                if (pd.pf_zpeer0) { dfree(pd.pf_zpeer0); pd.pf_zpeer0 = nullptr; }
                pd.pf_z = sycl::malloc_device<float>((size_t)T * vcache_detail::VC_K, *pd.q);
                pd.pf_zout = sycl::malloc_device<float>((size_t)T * vcache_detail::VC_K, *pd.q);
                pd.pf_h_zout = sycl::malloc_host<float>((size_t)T * vcache_detail::VC_K, *pd.q);
                pd.pf_red_d = sycl::malloc_device<int32_t>((size_t)T * vcache_detail::VC_NTOP, *pd.q);
                pd.pf_red_h = sycl::malloc_host<int32_t>((size_t)T * vcache_detail::VC_NTOP, *pd.q);
                // raw malloc_device on the PRIMARY context (not dalloc): OOM must fall back to the
                // legacy tiled path, not exit(3). Track the bytes for telemetry parity with dalloc.
                pd.pf_zpeer0 = sycl::malloc_device<float>((size_t)T * vcache_detail::VC_K, Q());
                if (pd.pf_zpeer0) dev_bytes_allocated() += (size_t)T * vcache_detail::VC_K * sizeof(float);
                pd.pf_cap_t = (pd.pf_z && pd.pf_zout && pd.pf_h_zout && pd.pf_red_d && pd.pf_red_h && pd.pf_zpeer0) ? T : 0;
                if (!pd.pf_cap_t) return false;
            }
            if (pd.pf_cap_p < P) {
                if (pd.pf_zg) { sycl::free(pd.pf_zg, *pd.q); pd.pf_zg = nullptr; }
                if (pd.pf_hb) { sycl::free(pd.pf_hb, *pd.q); pd.pf_hb = nullptr; }
                if (pd.pf_ye) { sycl::free(pd.pf_ye, *pd.q); pd.pf_ye = nullptr; }
                if (pd.pf_ptok_d) { sycl::free(pd.pf_ptok_d, *pd.q); pd.pf_ptok_d = nullptr; }
                if (pd.pf_ptok_h) { sycl::free(pd.pf_ptok_h, *pd.q); pd.pf_ptok_h = nullptr; }
                if (pd.pf_pw_d) { sycl::free(pd.pf_pw_d, *pd.q); pd.pf_pw_d = nullptr; }
                if (pd.pf_pw_h) { sycl::free(pd.pf_pw_h, *pd.q); pd.pf_pw_h = nullptr; }
                // Reserve +64 padded tail rows beyond logical capacity: launch_gemm_xmx_sg16
                // issues 8-row joint_matrix_load tiles at m0 = pbase + sg_m*8, which reads up
                // to 7 rows past pbase + ntok when ntok % 8 != 0 (avoids page fault at P end).
                const int alloc_p = std::max(P + 128, P + (P >> 2) + 128);
                pd.pf_zg = sycl::malloc_device<half>((size_t)alloc_p * vcache_detail::VC_K, *pd.q);
                pd.pf_hb = sycl::malloc_device<half>((size_t)alloc_p * vcache_detail::VC_K, *pd.q);
                pd.pf_ye = sycl::malloc_device<float>((size_t)alloc_p * vcache_detail::VC_K, *pd.q);
                pd.pf_ptok_d = sycl::malloc_device<int32_t>(alloc_p, *pd.q);
                pd.pf_ptok_h = sycl::malloc_host<int32_t>(alloc_p, *pd.q);
                pd.pf_pw_d = sycl::malloc_device<float>(alloc_p, *pd.q);
                pd.pf_pw_h = sycl::malloc_host<float>(alloc_p, *pd.q);
                const bool ok_p = (pd.pf_zg && pd.pf_hb && pd.pf_ye && pd.pf_ptok_d && pd.pf_ptok_h && pd.pf_pw_d && pd.pf_pw_h);
                pd.pf_cap_p = ok_p ? (alloc_p - 64) : 0;
                if (!pd.pf_cap_p) return false;
            }
            if (pd.pf_cap_items < nitems) {
                if (pd.pf_items_d) { sycl::free(pd.pf_items_d, *pd.q); pd.pf_items_d = nullptr; }
                if (pd.pf_items_h) { sycl::free(pd.pf_items_h, *pd.q); pd.pf_items_h = nullptr; }
                pd.pf_items_d = sycl::malloc_device<GxItem>(nitems, *pd.q);
                pd.pf_items_h = sycl::malloc_host<GxItem>(nitems, *pd.q);
                pd.pf_cap_items = (pd.pf_items_d && pd.pf_items_h) ? nitems : 0;
                if (!pd.pf_cap_items) return false;
            }
        } catch (const sycl::exception& e) {
            fprintf(stderr, "[peer-sycl] grouped-prefill scratch allocation failed on GPU %d (%s); using legacy tiled path\n",
                    pd.dev_idx, e.what());
            return false;
        }
        return true;
    }

    // Grow the shared pinned host bounce used by non-p2p peers to hold T*1536 activations.
    bool ensure_h_pf_z(size_t nfloats) {
        if (h_pf_z_cap_ >= nfloats && h_pf_z_) return true;
        try {
            // Allocate before freeing old buffer so OOM preserves the <=256-tok fallback buffer.
            float* fresh = sycl::malloc_host<float>(nfloats, Q());
            if (!fresh) return false;
            if (h_pf_z_) sycl::free(h_pf_z_, Q());
            h_pf_z_ = fresh;
            h_pf_z_cap_ = nfloats;
            return true;
        } catch (const sycl::exception& e) {
            fprintf(stderr, "[peer-sycl] host prefill bounce allocation failed (%s)\n", e.what());
            return false;
        }
    }

    static bool env_dec_sg32() {
        static const int v = [] { const char* e = std::getenv("HY4S_PEER_DEC_SG32"); return e && e[0] == '1'; }();
        return v;
    }
    static bool env_pf_legacy() {
        static const int v = [] { const char* e = std::getenv("HY4S_PEER_PF_LEGACY"); return e && e[0] == '1'; }();
        return v;
    }
    static bool env_dec_wait() {
        static const int v = [] { const char* e = std::getenv("HY4S_PEER_DEC_WAIT"); return e && e[0] == '1'; }();
        return v;
    }

    // Native SG16 + dedup + Q8_1_G64 + SLM decode on the peer's own queue — the exact kernel sequence
    // SyVramExpertCache::launch_decode runs on the primary GPU (see sy_vram_cache.h), producing
    // d_zout[t][0..1535] partial sums from the peer-resident deduplicated expert groups.
    void enqueue_peer_eval_sg16(PeerDev& pd, int T) {
        sycl::queue& pq = *pd.q;
        const int npairs = pd.planned_pairs;
        const int ngrps = pd.planned_groups;
        const int max_nt = pd.planned_max_ntok;
        if (npairs <= 0 || ngrps <= 0) return;
        pq.memcpy(pd.d_grps, pd.h_grps, (size_t)ngrps * sizeof(vcache_detail::VramExpertGroup));
        pq.memcpy(pd.d_pair_w, pd.h_pair_w, (size_t)npairs * sizeof(float));
        pq.memcpy(pd.d_tok_off, pd.h_tok_off, (size_t)(T + 1) * sizeof(int32_t));
        if (q8_act_) {
            vcache_detail::k_quant_q8_1_g64<float>(pd.d_z, T, pd.d_zq, pd.d_zd, pq);
            if (max_nt <= 1) vcache_detail::k_vcache_gemm1_q8<1>(pd.d_grps, ngrps, pd.d_zq, pd.d_zd, pd.d_hmid, pq);
            else if (max_nt <= 2) vcache_detail::k_vcache_gemm1_q8<2>(pd.d_grps, ngrps, pd.d_zq, pd.d_zd, pd.d_hmid, pq);
            else if (max_nt <= 4) vcache_detail::k_vcache_gemm1_q8<4>(pd.d_grps, ngrps, pd.d_zq, pd.d_zd, pd.d_hmid, pq);
            else vcache_detail::k_vcache_gemm1_q8<8>(pd.d_grps, ngrps, pd.d_zq, pd.d_zd, pd.d_hmid, pq);
            vcache_detail::k_quant_q8_1_g64<half>(pd.d_hmid, npairs, pd.d_hq, pd.d_hd, pq);
            if (max_nt <= 1) vcache_detail::k_vcache_gemm2_q8<1>(pd.d_grps, ngrps, pd.d_hq, pd.d_hd, pd.d_ye, pq);
            else if (max_nt <= 2) vcache_detail::k_vcache_gemm2_q8<2>(pd.d_grps, ngrps, pd.d_hq, pd.d_hd, pd.d_ye, pq);
            else if (max_nt <= 4) vcache_detail::k_vcache_gemm2_q8<4>(pd.d_grps, ngrps, pd.d_hq, pd.d_hd, pd.d_ye, pq);
            else vcache_detail::k_vcache_gemm2_q8<8>(pd.d_grps, ngrps, pd.d_hq, pd.d_hd, pd.d_ye, pq);
        } else {
            if (max_nt <= 1) vcache_detail::k_vcache_gemm1_fp<1>(pd.d_grps, ngrps, pd.d_z, pd.d_hmid, pq);
            else if (max_nt <= 2) vcache_detail::k_vcache_gemm1_fp<2>(pd.d_grps, ngrps, pd.d_z, pd.d_hmid, pq);
            else if (max_nt <= 4) vcache_detail::k_vcache_gemm1_fp<4>(pd.d_grps, ngrps, pd.d_z, pd.d_hmid, pq);
            else vcache_detail::k_vcache_gemm1_fp<8>(pd.d_grps, ngrps, pd.d_z, pd.d_hmid, pq);
            if (max_nt <= 1) vcache_detail::k_vcache_gemm2_fp<1>(pd.d_grps, ngrps, pd.d_hmid, pd.d_ye, pq);
            else if (max_nt <= 2) vcache_detail::k_vcache_gemm2_fp<2>(pd.d_grps, ngrps, pd.d_hmid, pd.d_ye, pq);
            else if (max_nt <= 4) vcache_detail::k_vcache_gemm2_fp<4>(pd.d_grps, ngrps, pd.d_hmid, pd.d_ye, pq);
            else vcache_detail::k_vcache_gemm2_fp<8>(pd.d_grps, ngrps, pd.d_hmid, pd.d_ye, pq);
        }
        // Per-token weighted reduce over this token's pair range (ascending expert_id) — identical to primary.
        const float* ye = pd.d_ye;
        const float* pw = pd.d_pair_w;
        const int32_t* toff = pd.d_tok_off;
        float* zout_dev = pd.d_zout;
        pq.parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) REQD_SG16 {
            const int t = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            const int p0 = toff[t], p1 = toff[t + 1];
            for (int d = tid; d < vcache_detail::VC_K; d += 256) {
                float acc = 0.0f;
                for (int p = p0; p < p1; ++p) acc += ye[(size_t)p * vcache_detail::VC_K + d] * pw[p];
                zout_dev[(size_t)t * vcache_detail::VC_K + d] = acc;
            }
        });
    }

    void enqueue_peer_eval(PeerDev& pd, int T, int npairs) {
        sycl::queue& pq = *pd.q;
        pq.memcpy(pd.d_pairs, pd.h_pairs, (size_t)npairs * sizeof(vcache_detail::VramPair));
        pq.memcpy(pd.d_tok_off, pd.h_tok_off, (size_t)(T + 1) * sizeof(int32_t));

        const vcache_detail::VramPair* dp = pd.d_pairs;
        const float* z_dev = pd.d_z;
        half* hmid = pd.d_hmid;
        float* ye = pd.d_ye;
        float* zout_dev = pd.d_zout;
        const int32_t* toff = pd.d_tok_off;

        auto launch_k = [&](auto q8_tag) {
            constexpr bool Q8A = decltype(q8_tag)::value;
            pq.parallel_for(NR((size_t)npairs * 192, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
                const int gid = (int)it.get_group(0);
                const int p = gid / 192, rb = gid % 192;
                const int wave = (int)it.get_local_id(0) >> 5, lane = (int)it.get_local_id(0) & 31;
                const int d = rb * 8 + wave;
                const vcache_detail::VramPair vp = dp[p];
                const size_t row_b = (size_t)6 * vcache_detail::vc_sb_bytes(vp.gu);
                const uint8_t* wgu = vp.body + 3072;
                const float* zt = z_dev + (size_t)vp.tok * vcache_detail::VC_K;
                const float g = vcache_detail::sg_dot_row<float, Q8A>(it, vp.gu, wgu + (size_t)d * row_b, zt);
                const float u = vcache_detail::sg_dot_row<float, Q8A>(it, vp.gu, wgu + (size_t)(vcache_detail::VC_K + d) * row_b, zt);
                if (lane == 0) {
                    const float sq = h2f_bits(ld16(vp.body + 2 * d));
                    const float s = sycl::fmin(g / (1.0f + sycl::exp(-g)), 10.0f);
                    const float uc = sycl::fmax(-10.0f, sycl::fmin(u, 10.0f));
                    hmid[(size_t)p * vcache_detail::VC_K + d] = (half)((s * uc) / sq);
                }
            });

            pq.parallel_for(NR((size_t)npairs * 192, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
                const int gid = (int)it.get_group(0);
                const int p = gid / 192, rb = gid % 192;
                const int wave = (int)it.get_local_id(0) >> 5, lane = (int)it.get_local_id(0) & 31;
                const int d = rb * 8 + wave;
                const vcache_detail::VramPair vp = dp[p];
                const size_t row_b = (size_t)6 * vcache_detail::vc_sb_bytes(vp.dn);
                const uint8_t* wdn = vp.body + vp.dn_off;
                const float y = vcache_detail::sg_dot_row<half, Q8A>(it, vp.dn, wdn + (size_t)d * row_b, hmid + (size_t)p * vcache_detail::VC_K);
                if (lane == 0) ye[(size_t)p * vcache_detail::VC_K + d] = y;
            });
        };
        if (q8_act_) launch_k(std::true_type{});
        else launch_k(std::false_type{});

        pq.parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) {
            const int t = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            const int p0 = toff[t], p1 = toff[t + 1];
            for (int d = tid; d < vcache_detail::VC_K; d += 256) {
                float acc = 0.0f;
                for (int p = p0; p < p1; ++p) acc += ye[(size_t)p * vcache_detail::VC_K + d] * dp[p].weight;
                zout_dev[(size_t)t * vcache_detail::VC_K + d] = acc;
            }
        });
    }

    int nlayers_ = 0;
    bool q8_act_ = true;
    int total_experts_ = 0;
    size_t total_dev_bytes_ = 0;
    uint64_t hits_ = 0;
    std::vector<Meta> meta_;
    std::vector<int> owner_;      // [nlayers * 512]: -1 or peer index in peers_
    std::vector<uint8_t*> d_ptr_; // [nlayers * 512]: device ptr on owning peer
    std::vector<PeerDev> peers_;
    float* h_pf_z_ = nullptr;     // pinned host bounce for non-P2P prefill (grown by ensure_h_pf_z)
    size_t h_pf_z_cap_ = 0;       // capacity of h_pf_z_ in floats
    std::vector<int32_t> pf_pos_; // per-token pair-index scratch for grouped prefill planning
};

}  // namespace hy4s
