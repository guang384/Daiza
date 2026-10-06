//! Standalone benchmark: Q1_0 matvec batch4 (speculative-decode verify kernel)
//! vs 4x single-token matvec (current decode path).
//!
//! Motivation: batch verify for speculative decoding amortizes the DRAM weight
//! stream across k drafted tokens. The prefill batch path (n_batch=128) is slow
//! because x spans 2.5MB (L1/L2 thrash), but verify uses k=4 -> x = 80KB (fits L2).
//! This bench measures the real k=4 amortization with the existing kernels:
//!   A) 4 x matvec_into_slice        (current decode, 4x weight stream)
//!   B) 1 x matvec_batch_into_slice  (k=4 batch, 1x weight stream)
//!
//! Run: cargo build --release --example bench_batch4 && target/release/examples/bench_batch4
//!
//! DRAM mode: rotate over N_MATRICES (150MB total, >> L3 18MB) so every pass
//! streams weights from DRAM, matching real decode (12GB weights, L3 useless).

use daiza_engine::model::weights::Q1_0Matrix;
use daiza_engine::model::workspace;
use std::time::Instant;

const ROWS: usize = 17408; // MLP gate/up row count (largest matvec in the model)
const COLS: usize = 5120; // hidden
const GROUP: usize = 128;
const BLOCK_BYTES: usize = 18; // 2B f16 scale + 16B sign bits
const N_THREADS: usize = 14;
const N_MATRICES: usize = 12; // 12 x 12.55MB = 150MB, flushes L3 between visits

fn make_random_q10(rows: usize, cols: usize) -> Q1_0Matrix {
    let groups = cols / GROUP;
    let mut bytes = vec![0u8; rows * groups * BLOCK_BYTES];
    // xorshift PRNG, cheap and deterministic
    let mut s: u64 = 0x9E3779B97F4A7C15;
    for chunk in bytes.chunks_exact_mut(BLOCK_BYTES) {
        // scale: random f16 bits in a sane range (0.001..0.05) to keep magnitudes similar
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        let scale_f: f32 = 0.001 + (s & 0xFF) as f32 / 255.0 * 0.049;
        let scale_bits = scale_f.to_bits(); // f32 bits; take upper 16 as f16 approx
        // simple f16 encode (round-to-nearest-even not needed for a bench)
        let f16_bits: u16 = ((scale_bits >> 16) & 0x7FFF) as u16 | (if scale_bits & 0x8000_0000 != 0 { 0x8000 } else { 0 });
        chunk[0] = (f16_bits & 0xFF) as u8;
        chunk[1] = (f16_bits >> 8) as u8;
        for b in chunk[2..18].iter_mut() {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            *b = (s >> 32) as u8;
        }
    }
    Q1_0Matrix { bytes, rows, cols }
}

fn main() {
    // Clean env: no DAIZA_* overrides (fresh process anyway)
    workspace::init_thread_pool(N_THREADS);

    // DRAM-resident workload: rotate over N_MATRICES distinct weight matrices.
    // Each iter touches 150MB total, so every matrix visit is a cold DRAM stream
    // (matches decode: 64 blocks x ~40MB weights >> L3).
    let mats: Vec<Q1_0Matrix> = (0..N_MATRICES)
        .map(|i| make_random_q10(ROWS, COLS + i)) // vary cols slightly to avoid page aliasing
        .collect();
    let x_single = vec![0.017f32; COLS + N_MATRICES];
    let mut y_single = vec![0.0f32; ROWS];
    let mut y_batch = vec![0.0f32; 4 * ROWS];
    let mut x4 = vec![0.0f32; 4 * (COLS + N_MATRICES)];
    x4.iter_mut().for_each(|v| *v = 0.017);

    let total_mb: f64 = mats.iter().map(|m| m.bytes.len()).sum::<usize>() as f64 / 1024.0 / 1024.0;

    // Warmup: one full rotation (page-in 150MB) + pool spin state
    for m in &mats {
        for _ in 0..4 {
            m.matvec_into_slice(&x_single[..m.cols], &mut y_single);
        }
        let xb = &x4[..4 * m.cols];
        m.matvec_batch_into_slice(xb, 4, &mut y_batch);
    }

    // ---- A: 4 rounds of rotation, single matvec each (true decode pattern:
    //        each visit is a cold DRAM stream, no L3 reuse between visits) ----
    let iters = 6;
    let mut times_a: Vec<f64> = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        for _round in 0..4 {
            for m in &mats {
                m.matvec_into_slice(&x_single[..m.cols], &mut y_single);
            }
        }
        times_a.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times_a.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_a = times_a[iters / 2];

    // ---- B: per-matrix 1x batch4 matvec ----
    let mut times_b: Vec<f64> = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        for m in &mats {
            m.matvec_batch_into_slice(&x4[..4 * m.cols], 4, &mut y_batch);
        }
        times_b.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times_b.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_b = times_b[iters / 2];

    // ---- C: per-matrix 1x verify4 (new kernel: zero-spill, LUT shared) ----
    let mut y_verify = vec![0.0f32; 4 * ROWS];
    let mut times_c: Vec<f64> = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        for m in &mats {
            m.matvec_verify4_into_slice(&x4[..4 * m.cols], &mut y_verify[..4 * m.rows]);
        }
        times_c.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times_c.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_c = times_c[iters / 2];

    // ---- D: per-matrix 1x verify4t (interleaved x, L1-resident loads) ----
    let mut y_verify_t = vec![0.0f32; 4 * ROWS];
    let mut times_d: Vec<f64> = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        for m in &mats {
            m.matvec_verify4t_into_slice(&x4[..4 * m.cols], &mut y_verify_t[..4 * m.rows]);
        }
        times_d.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times_d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_d = times_d[iters / 2];

    // Correctness: verify4t vs single (token0 lanes must match single output)
    let mut max_err_c = 0.0f32;
    if let Some(m) = mats.first() {
        m.matvec_into_slice(&x_single[..m.cols], &mut y_single);
        for (i, &ys) in y_single[..m.rows].iter().enumerate() {
            let e = (ys - y_verify_t[i]).abs();
            if e > max_err_c { max_err_c = e; }
        }
    }

    println!("DRAM-resident workload: {} matrices x {} rows (total {:.0} MB)", N_MATRICES, ROWS, total_mb);
    println!("A) 4x single matvec   : median {:>8.2} ms/iter  -> per-4-token {:.2} ms", med_a, med_a / N_MATRICES as f64);
    println!("B) 1x batch4 matvec   : median {:>8.2} ms/iter  -> per-4-token {:.2} ms", med_b, med_b / N_MATRICES as f64);
    println!("C) 1x verify4 kernel  : median {:>8.2} ms/iter  -> per-4-token {:.2} ms", med_c, med_c / N_MATRICES as f64);
    println!("D) 1x verify4t kernel  : median {:>8.2} ms/iter  -> per-4-token {:.2} ms", med_d, med_d / N_MATRICES as f64);
    println!("amortization C vs A   : {:.2}x faster (A/C)", med_a / med_c);
    println!("amortization D vs A   : {:.2}x faster (A/D)", med_a / med_d);
    println!("effective BW A        : {:.1} GB/s (weights read 4x per iter)", total_mb * 4.0 / 1024.0 / (med_a / 1000.0));
    println!("effective BW C        : {:.1} GB/s (weights read 1x per iter)", total_mb / 1024.0 / (med_c / 1000.0));
    println!("verify4t max err vs single: {:.6}", max_err_c);
}
