//! 纯内存流带宽带宽天花板测试 (read / write / copy)
//!
//! 目的: 判定 LPDDR5X-8400 (128-bit, 理论 ~134GB/s) 在本机的真实可持续带宽,
//! 区分 "引擎受计算吞吐限制" vs "平台带宽天花板"。
//! 若纯流读 >> matvec 有效带宽 (~20-24GB/s), 则 decode/prefill 的瓶颈是
//! Q1_0 内核计算吞吐 (功耗墙), 而非内存带宽。
//!
//! Run: cargo build --release --example bench_mem_bw && target/release/examples/bench_mem_bw

use std::time::Instant;

const GB: usize = 1024 * 1024 * 1024;
/// 读/写缓冲 2GB (远超 LLC 24MB), copy 用 2×1GB
const READ_BYTES: usize = 2 * GB;
const COPY_BYTES: usize = GB;

/// 裸指针转 usize 传递 (整数天然 Copy+Send, 绕开精确捕获/借用冲突)
fn ptr_addr(p: *mut u64) -> usize {
    p as usize
}

fn main() {
    let n = READ_BYTES / 8;
    let mut buf = vec![0u64; n];
    // 预热 + 触碰全部页
    for chunk in buf.chunks_mut(1 << 18) {
        chunk.fill(0x5A5A_5A5A_5A5A_5A5A);
    }
    let mut copy_dst = vec![0u64; COPY_BYTES / 8];
    let cn = copy_dst.len();
    // 裸指针转地址 (整数) 绕开闭包借用冲突
    let src_addr = ptr_addr(buf.as_ptr() as *mut u64);
    let wptr_addr = ptr_addr(buf.as_mut_ptr());
    let dst_addr = ptr_addr(copy_dst.as_mut_ptr());

    println!(
        "buffer: read {} GB / copy 2x{} GB (LPDDR5X-8400 128-bit, theoretical {:.0} GB/s)",
        READ_BYTES / GB,
        COPY_BYTES / GB,
        8400.0 * 16.0 / 1000.0
    );

    for &nt in &[1usize, 4, 8, 10, 12, 14] {
        // ---- read stream (sum, auto-vectorized paddq) ----
        let best_read = run_best(3, || {
            let per = n / nt;
            let handles: Vec<_> = (0..nt)
                .map(|t| {
                    let lo = t * per;
                    let hi = if t == nt - 1 { n } else { lo + per };
                    let s = unsafe { std::slice::from_raw_parts(src_addr as *const u64, hi - lo) };
                    std::thread::spawn(move || {
                        let mut acc = 0u64;
                        for c in s.chunks_exact(4) {
                            // 4 路链式加, 打破依赖链, 利于向量化
                            acc = acc
                                .wrapping_add(c[0])
                                .wrapping_add(c[1])
                                .wrapping_add(c[2])
                                .wrapping_add(c[3]);
                        }
                        acc
                    })
                })
                .collect();
            let mut total = 0u64;
            for h in handles {
                total = total.wrapping_add(h.join().unwrap());
            }
            total
        });

        // ---- write stream (fill, memset-able) ----
        let best_write = run_best(3, || {
            let per = n / nt;
            let handles: Vec<_> = (0..nt)
                .map(|t| {
                    let lo = t * per;
                    let hi = if t == nt - 1 { n } else { lo + per };
                    let v = 0x1234_5678_9ABC_DEF0u64.wrapping_add(t as u64);
                    let base = wptr_addr + lo * 8;
                    std::thread::spawn(move || unsafe {
                        let s = std::slice::from_raw_parts_mut(base as *mut u64, hi - lo);
                        s.fill(v);
                        0u64
                    })
                })
                .collect();
            let mut sink = 0u64;
            for h in handles {
                sink |= h.join().unwrap_or(0);
            }
            sink
        });

        // ---- copy stream (1GB src -> 1GB dst, memcpy) ----
        let best_copy = run_best(3, || {
            let per = cn / nt;
            let handles: Vec<_> = (0..nt)
                .map(|t| {
                    let lo = t * per;
                    let hi = if t == nt - 1 { cn } else { lo + per };
                    let sbase = src_addr + lo * 8;
                    let dbase = dst_addr + lo * 8;
                    std::thread::spawn(move || unsafe {
                        let s = std::slice::from_raw_parts(sbase as *const u64, hi - lo);
                        let d = std::slice::from_raw_parts_mut(dbase as *mut u64, hi - lo);
                        d.copy_from_slice(s);
                        0u64
                    })
                })
                .collect();
            let mut sink = 0u64;
            for h in handles {
                sink |= h.join().unwrap_or(0);
            }
            sink
        });

        println!(
            "threads={:>2} | read {:>6.1} GB/s | write {:>6.1} GB/s | copy {:>6.1} GB/s",
            nt,
            READ_BYTES as f64 / 1e9 / best_read,
            READ_BYTES as f64 / 1e9 / best_write,
            (2.0 * COPY_BYTES as f64 / 1e9) / best_copy, // copy = 读+写各 1GB
        );
    }
}

/// 多次迭代取最优 (返回秒); 返回值由调用方消耗防优化
fn run_best(iters: usize, mut f: impl FnMut() -> u64) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let sink = f();
        let dt = t.elapsed().as_secs_f64();
        if sink == u64::MAX {
            println!("unreachable");
        }
        if dt < best {
            best = dt;
        }
    }
    best
}
