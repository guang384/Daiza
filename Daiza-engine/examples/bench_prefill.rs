//! Prefill GEMM bench: 块 GEMM (k-lane) vs 数值参照 (n_tokens = 22..128)
//!
//! dispatch 条件 (weights.rs): n_batch >= 64 && rows >= 4096 → GEMM,
//! n_tokens < 64 时本 bench 实际走原 batch kernel 路径 (作 baseline 对照)。
//!
//! Run: cargo build --release --example bench_prefill && target/release/examples/bench_prefill

use daiza_engine::model::weights::Q1_0Matrix;
use daiza_engine::model::workspace;
use std::time::Instant;

const ROWS: usize = 17408;
const COLS: usize = 5120;
const GROUP: usize = 128;
const BLOCK_BYTES: usize = 18;
const N_THREADS: usize = 14;
const N_MATRICES: usize = 12;

fn make_random_q10(rows: usize, cols: usize) -> Q1_0Matrix {
    let groups = cols / GROUP;
    let mut bytes = vec![0u8; rows * groups * BLOCK_BYTES];
    let mut s: u64 = 0x9E3779B97F4A7C15;
    for chunk in bytes.chunks_exact_mut(BLOCK_BYTES) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        let scale_f: f32 = 0.001 + (s & 0xFF) as f32 / 255.0 * 0.049;
        let scale_bits = scale_f.to_bits();
        let f16_bits: u16 = ((scale_bits >> 16) & 0x7FFF) as u16
            | (if scale_bits & 0x8000_0000 != 0 { 0x8000 } else { 0 });
        chunk[0] = (f16_bits & 0xFF) as u8;
        chunk[1] = (f16_bits >> 8) as u8;
        for b in chunk[2..18].iter_mut() {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            *b = (s >> 32) as u8;
        }
    }
    Q1_0Matrix { bytes, rows, cols }
}

fn bench(n_tokens: usize, iters: usize, mats: &[Q1_0Matrix]) {
    let rows = mats[0].rows;
    let cols = mats[0].cols;
    let x = vec![0.017f32; n_tokens * cols];
    let mut y_batch = vec![0.0f32; n_tokens * rows];
    let mut y_gemm = vec![0.0f32; n_tokens * rows];

    // Warmup
    for m in mats {
        m.matvec_batch_into_slice(&x, n_tokens, &mut y_batch);
    }
    // 数值验证：用 dot_q1_0_row_avx2 单行单 token 做参照 (前 64 行)
    let mut y_ref = vec![0.0f32; n_tokens * rows];
    if let Some(m) = mats.first() {
        for t in 0..n_tokens {
            for i in 0..rows.min(64) {
                unsafe {
                    y_ref[t * rows + i] = daiza_engine::tensor::quant::dot_q1_0_row_avx2(
                        &m.bytes, i, cols, &x[t * cols..(t + 1) * cols],
                    );
                }
            }
        }
    }

    // 实测路径 (与 weights.rs dispatch 条件一致): >=64 走 GEMM, 其余走 batch kernel
    let path = if n_tokens >= 64 { "gemm" } else { "batch" };
    let mut times = Vec::new();
    for _ in 0..iters {
        let t0 = Instant::now();
        for m in mats {
            m.matvec_batch_into_slice(&x, n_tokens, &mut y_gemm);
        }
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[iters / 2];

    // 数值校验: GEMM vs ref (前 64 行, 所有 token)
    let mut max_err = 0.0f32;
    for t in 0..n_tokens {
        for i in 0..rows.min(64) {
            let e = (y_gemm[t * rows + i] - y_ref[t * rows + i]).abs();
            if e > max_err { max_err = e; }
        }
    }

    let total_mb: f64 = mats.iter().map(|m| m.bytes.len()).sum::<usize>() as f64 / 1024.0 / 1024.0;
    println!("n_tokens={:>3} [{}] | median {:>8.2} ms/iter -> {:>7.2} ms/token | max_err vs single-row: {:.6} | weights {:.0} MB",
        n_tokens, path, med, med / N_MATRICES as f64 / n_tokens as f64, max_err, total_mb);
}

/// 交错轮转 worker 数配置 (热漂移公平对比): 同进程内 9/12/13 轮流跑,
/// 单进程单配置的跨 run 对比在本机热噪声 (±20%) 下不可信。
fn bench_workers(mats: &[Q1_0Matrix], saved_active: usize) {
    let rows = mats[0].rows;
    let cols = mats[0].cols;
    let n_tokens = 128usize;
    let x = vec![0.017f32; n_tokens * cols];
    let mut y = vec![0.0f32; n_tokens * rows];
    let configs = [9usize, 12, 13];
    const ROUNDS: usize = 3;
    let mut times = [[0f64; 3]; ROUNDS];
    for row in times.iter_mut() {
        for (ci, &n) in configs.iter().enumerate() {
            workspace::set_active_workers(n);
            let t0 = Instant::now();
            for m in mats {
                m.matvec_batch_into_slice(&x, n_tokens, &mut y);
            }
            row[ci] = t0.elapsed().as_secs_f64() * 1000.0;
        }
    }
    workspace::set_active_workers(saved_active);
    for (ci, &n) in configs.iter().enumerate() {
        let mut med: Vec<f64> = (0..ROUNDS).map(|r| times[r][ci]).collect();
        med.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("workers={:>2} [{}x{}] median {:>8.2} ms/iter (128t)",
            n, rows, cols, med[ROUNDS / 2]);
    }
}

fn main() {
    workspace::init_thread_pool(N_THREADS);
    // 形状 1: MLP gate/up (17408×5120, r_block=32)
    let mats: Vec<Q1_0Matrix> = (0..N_MATRICES)
        .map(|_| make_random_q10(ROWS, COLS))
        .collect();
    // 形状 2: MLP down (5120×17408, 宽列 → r_block=10, 验证自适应 blocking)
    let mats_down: Vec<Q1_0Matrix> = (0..N_MATRICES)
        .map(|_| make_random_q10(5120, 17408))
        .collect();
    // ★ 交错 worker 轮转模式 (DAIZA_BENCH_WORKERS=1): 单进程内公平对比 9/12/13
    if std::env::var("DAIZA_BENCH_WORKERS").is_ok() {
        let saved = workspace::active_workers();
        println!("=== worker-count sweep (interleaved, {} matrices) ===", N_MATRICES);
        bench_workers(&mats, saved);
        bench_workers(&mats_down, saved);
        return;
    }
    // warmup page-in (两种形状)
    for mats in [&mats, &mats_down] {
        let cols = mats[0].cols;
        let rows = mats[0].rows;
        let x_warm = vec![0.017f32; 128 * cols];
        let mut y_warm = vec![0.0f32; 128 * rows];
        for m in mats {
            m.matvec_batch_into_slice(&x_warm, 128, &mut y_warm);
        }
    }
    println!("=== prefill GEMM bench ({} matrices, DRAM-resident rotation) ===", N_MATRICES);
    // 128t 最先测 (run 内自热会让尾部测量偏差 30%+); 轻负载放后面
    println!("--- shape 17408x5120 (gate/up) ---");
    bench(128, 6, &mats);
    bench(64, 6, &mats);
    bench(32, 6, &mats);
    bench(22, 6, &mats);
    println!("--- shape 5120x17408 (down_proj) ---");
    bench(128, 6, &mats_down);
    bench(64, 6, &mats_down);
    bench(32, 6, &mats_down);
    bench(22, 6, &mats_down);
}
