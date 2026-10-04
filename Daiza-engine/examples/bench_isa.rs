//! ISA 能力检测 + AVX2 vs AVX-512 FMA 峰值吞吐测试 (功耗墙下的可持续算力)
//!
//! 目的: 225H (Arrow Lake-H) P 核 Lion Cove 理论支持 AVX-512 (2×512b FMA),
//!       E 核 Skymont 也支持 AVX-512 ISA。Daiza 当前全部内核为 AVX2。
//!       本 bench 回答:
//!       1) 本机是否真的开放 AVX-512F/BW/VL/DQ (OS + 固件支持)?
//!       2) AVX-512 vs AVX2 的 FMA 峰值/可持续吞吐比 (功耗墙折损后)?
//!
//! Run: cargo build --release --example bench_isa && target/release/examples/bench_isa

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use std::time::Instant;

/// AVX2 峰值: 8 条独立 FMA 依赖链 (打破端口依赖), 256-bit
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_avx2(iters: usize) -> f32 {
    let mut acc = [_mm256_set1_ps(1.0f32); 8];
    let b = _mm256_set1_ps(1.0000001);
    let c = _mm256_set1_ps(0.0000001);
    for _ in 0..iters {
        acc[0] = _mm256_fmadd_ps(b, acc[0], c);
        acc[1] = _mm256_fmadd_ps(b, acc[1], c);
        acc[2] = _mm256_fmadd_ps(b, acc[2], c);
        acc[3] = _mm256_fmadd_ps(b, acc[3], c);
        acc[4] = _mm256_fmadd_ps(b, acc[4], c);
        acc[5] = _mm256_fmadd_ps(b, acc[5], c);
        acc[6] = _mm256_fmadd_ps(b, acc[6], c);
        acc[7] = _mm256_fmadd_ps(b, acc[7], c);
    }
    let mut s = 0.0f32;
    for a in &acc {
        let t = _mm256_hadd_ps(*a, *a);
        let t = _mm256_hadd_ps(t, t);
        let arr: [f32; 8] = std::mem::transmute(t);
        s += arr.iter().sum::<f32>();
    }
    s
}

/// AVX-512 峰值: 8 条独立 FMA 链, 512-bit
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fma_avx512(iters: usize) -> f32 {
    let mut acc = [_mm512_set1_ps(1.0f32); 8];
    let b = _mm512_set1_ps(1.0000001);
    let c = _mm512_set1_ps(0.0000001);
    for _ in 0..iters {
        for i in 0..8 {
            acc[i] = _mm512_fmadd_ps(b, acc[i], c);
        }
    }
    let mut s = 0.0f32;
    for a in &acc {
        let arr: [f32; 16] = std::mem::transmute(*a);
        s += arr.iter().sum::<f32>();
    }
    s
}

fn bench_one(name: &str, flops_per_iter: f64, f: impl Fn(usize) -> f32 + Send + Sync + Copy + 'static) -> f64 {
    let iters = 30_000_000usize;
    // 单线程
    let t = Instant::now();
    let sink = f(iters);
    let dt1 = t.elapsed().as_secs_f64();
    let gflops1 = flops_per_iter * iters as f64 / dt1 / 1e9;
    if sink == f32::MAX {
        println!(".");
    }

    // 多线程 (各线程独立 FMA 链, 纯寄存器, 考验功耗墙)
    let mut best_mt = 0.0f64;
    for &nt in &[4usize, 8, 9, 14] {
        let t = Instant::now();
        let per = iters / 2 / nt;
        let handles: Vec<_> = (0..nt)
            .map(|_| {
                let ff = f;
                std::thread::spawn(move || {
                    let s = ff(per);
                    if s == f32::MAX {
                        println!(".");
                    }
                })
            })
            .collect();
        for h in handles {
            let _ = h.join();
        }
        let dt = t.elapsed().as_secs_f64();
        let gflops = flops_per_iter * (per * nt) as f64 / dt / 1e9;
        if gflops > best_mt {
            best_mt = gflops;
        }
        println!("  {name} threads={nt:>2}: {gflops:7.0} GFLOPS");
    }
    println!("  {name} single-thread: {gflops1:7.0} GFLOPS");
    best_mt
}

fn main() {
    println!("=== ISA detection (CPUID) ===");
    #[cfg(target_arch = "x86_64")]
    {
        println!("  avx2           : {}", std::is_x86_feature_detected!("avx2"));
        println!("  fma            : {}", std::is_x86_feature_detected!("fma"));
        println!("  avx512f        : {}", std::is_x86_feature_detected!("avx512f"));
        println!("  avx512dq       : {}", std::is_x86_feature_detected!("avx512dq"));
        println!("  avx512bw       : {}", std::is_x86_feature_detected!("avx512bw"));
        println!("  avx512vl       : {}", std::is_x86_feature_detected!("avx512vl"));
        // AMX: std 的 is_x86_feature_detected 不含 amx_int8 (rust 1.75+ 有 "amx_int8"?)
        // 用 cpuid 直接查 leaf 7 edx bit 24 (amx-int8) / 22 (amx-bf16)
        let (a, b, c, d) = cpuid(7, 0);
        let amx_bf16 = (d >> 22) & 1 == 1;
        let amx_int8 = (d >> 24) & 1 == 1;
        println!("  amx-bf16 (cpuid): {amx_bf16}");
        println!("  amx-int8 (cpuid): {amx_int8}");
        // avx512f 硅片位: leaf7 EBX bit16 (区别于 OS XCR0 屏蔽)
        let avx512f_silicon = (b >> 16) & 1 == 1;
        println!("  avx512f silicon (cpuid leaf7.ebx[16]): {avx512f_silicon}");
        let _ = (a, c);
        // CPU brand string (leaf 0x80000002..04)
        let mut brand = String::new();
        for leaf in [0x8000_0002u32, 0x8000_0003, 0x8000_0004] {
            let (a, b, c, d) = cpuid(leaf, 0);
            for v in [a, b, c, d] {
                let bytes = v.to_le_bytes();
                for ch in bytes {
                    if ch.is_ascii() && ch != 0 {
                        brand.push(ch as char);
                    }
                }
            }
        }
        println!("  cpu brand        : {}", brand.trim());
    }

    println!("\n=== FMA sustained throughput (register-resident, power-wall limited) ===");
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // 8 chains × 8 lanes × 2 flops = 128 flops/iter
            bench_one("AVX2 ", 8.0 * 8.0 * 2.0, |it| fma_avx2(it));
        }
        if std::is_x86_feature_detected!("avx512f") {
            // 8 chains × 16 lanes × 2 flops = 256 flops/iter
            bench_one("AVX512", 8.0 * 16.0 * 2.0, |it| fma_avx512(it));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn cpuid(leaf: u32, sub: u32) -> (u32, u32, u32, u32) {
    let r = unsafe { std::arch::x86_64::__cpuid_count(leaf, sub) };
    (r.eax, r.ebx, r.ecx, r.edx)
}
