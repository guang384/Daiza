//! 跨 token 复用的工作区缓冲区与持久线程池
//!
//! ## 设计动机
//!
//! 原实现每 token 在 block/attention/ssm/mlp 内部产生 ~1000+ 次 `vec!`/`to_vec()` 堆分配,
//! 每次 alloc ~20μs,合计 ~20ms/token,占单 token 时延的显著比例。
//!
//! 本 Workspace 在 `ForwardContext` 创建时一次性预分配所有热路径 buffer,
//! 跨 token 复用 —— 同一个 buffer 在每次 `forward_single_token` 中被覆盖写入。
//!
//! 主残差流 `h_buf` 不在本 Workspace 中,而在 `ForwardContext::h_buf`,
//! 避免与 `&mut Workspace` 的借用冲突(两者需同时传给 `block::forward_single_inplace`)。
//!
//! ## 持久线程池
//!
//! 原实现每个 matvec 调用都创建 `std::thread::scope`,每 token 产生 ~369 次 scope 创建
//! 和 ~2952 次 thread spawn (Windows CreateThread 开销 ~50μs/次)。
//!
//! `ThreadPool` 在引擎启动时一次性创建 N 个 worker 线程,通过 mpsc channel 分发任务,
//! 消除热路径上的所有 thread spawn 和 scope 创建开销。
//!
//! ## Buffer 划分原则
//!
//! - 同一阶段不会同时使用的 buffer 可以共享(本实现保守,全部独立)
//! - 大小固定(由 Config 决定),`attn_scores` 例外(随序列长度增长)
//! - 最终输出仍返回 owned `Vec`(每 token 1 个 alloc,可接受),内部 buffer 全部复用

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::cell::Cell;

use crate::model::config::Config;

// ---------------------------------------------------------------------------
// 全局持久线程池(OnceLock, 引擎启动时初始化一次, 全局复用)
// ---------------------------------------------------------------------------

static GLOBAL_POOL: OnceLock<ThreadPool> = OnceLock::new();

// ---------------------------------------------------------------------------
// Worker 等待模式 (OnceLock 缓存, worker_loop 与 scatter_wait 共享读取)
//   yield 模式 (默认): active worker spin + yield_now, 永不 park
//     → scatter_wait 无需 unpark (worker 自旋检测 generation 变化)
//   park 模式 (DAIZA_WAIT_MODE=park): active worker spin + park
//     → scatter_wait 必须 unpark 唤醒 park 中的 worker
// ---------------------------------------------------------------------------
static WAIT_MODE_PARK: OnceLock<bool> = OnceLock::new();

/// 当前是否为 park 等待模式 (yield 模式下 scatter_wait 可跳过 unpark syscall)
fn use_park_mode() -> bool {
    *WAIT_MODE_PARK.get_or_init(|| std::env::var("DAIZA_WAIT_MODE").as_deref() == Ok("park"))
}

// ---------------------------------------------------------------------------
// 嵌套并行检测: 防止 matvec_into_slice 在 par_for_patches 闭包内再次调度
// ---------------------------------------------------------------------------
// thread_local 标志: 当前线程是否正在执行 scatter_wait 分发的任务.
// - trampoline / stride_trampoline / steal_trampoline 进入时置位, 退出时清除
// - scatter_wait 主线程执行 chunk 时置位, 完成后清除
// - matvec_into_slice 检查此标志: 若在并行区内, 走 serial 路径, 避免嵌套调度死等
thread_local! {
    static IN_PARALLEL: Cell<bool> = const { Cell::new(false) };
}

/// 当前线程是否在并行区内 (正在执行 scatter_wait 分发的任务)
pub fn in_parallel_region() -> bool {
    IN_PARALLEL.with(|f| f.get())
}

fn set_in_parallel(v: bool) {
    IN_PARALLEL.with(|f| f.set(v));
}

/// 初始化全局线程池(引擎启动时调用一次)
///
/// ★ 热降频根治: 读取 DAIZA_ACTIVE_WORKERS 环境变量, 限制活跃 worker 数
///   未设置时默认全核利用; main.rs 根据 max_tokens 自动设置
pub fn init_thread_pool(n_threads: usize) {
    GLOBAL_POOL.get_or_init(|| {
        let pool = ThreadPool::new(n_threads);
        // ★ 读取 DAIZA_ACTIVE_WORKERS: 长跑时减少活跃核心, 避免热降频
        if let Ok(s) = std::env::var("DAIZA_ACTIVE_WORKERS") {
            if let Ok(n) = s.parse::<usize>() {
                if n > 0 && n < n_threads {
                    pool.set_active_workers(n);
                    eprintln!("[threadpool] active workers limited to {n} (DAIZA_ACTIVE_WORKERS)");
                }
            }
        }
        pool
    });
}

/// 获取全局线程池引用(若已初始化)
pub fn get_thread_pool() -> Option<&'static ThreadPool> {
    GLOBAL_POOL.get()
}

// ===========================================================================
// P0-C: park/unpark 零分配线程池
// ===========================================================================
//
// 原实现 (mpsc channel) 每次 scatter_wait 产生:
//   - 2 个 Arc 分配 (闭包 + AtomicUsize done 计数器)
//   - N 个 Box 分配 (per-worker job 闭包, N=13)
//   - N 次 channel send + N 次 channel recv
//   - N 次 Arc::clone + N 次 Arc drop
//   每 token ~257 barriers × (2 Arc + 13 Box + 13 channel) = ~514 Arc + ~3341 Box
//   dispatch 开销 ~0.5-1.8ms/token, 是 14 线程效率仅 40% 的主因。
//
// 新实现 (park/unpark + atomic generation):
//   - 零堆分配 (所有状态预分配在 Shared 中)
//   - worker 空闲时 park (不消耗 CPU), 主线程 unpark 唤醒
//   - main thread 设置 func/ctx/n_dispatch 后 fetch_add(Release) generation
//   - worker 被 unpark 唤醒后读 func/ctx, 调用 trampoline(ctx, tid)
//   - worker 完成后 fetch_add(Release) done 计数器
//   - main spin on done (Acquire) 等待完成
//
// park/unpark 语义: unpark 是 "sticky" 的 — 若在 park 之前调用,
// 下一次 park 立即返回。因此 main 可以安全地在 worker park 之前调用 unpark。
//
// 类型擦除: trampoline<F> 把 *const () 转回 &F 并调用, F 在 scatter_wait 栈上
// (调用方保证 scatter_wait 阻塞至所有 worker 完成, 故 F 生命周期安全)

/// 共享状态 (Arc 包裹, worker 和 main 共享)
///
/// ★ Cache line 优化: 6 个 atomic 原本挤在同一 cache line, 每 barrier 产生
/// ~26 次跨核 cache line transfer (~50ns/次 × 257 barriers = ~13ms/token)。
/// 拆成 4 个 cache line, 按"写者"分组:
///   Line 1 (main 写, worker 读): func, ctx, n_dispatch
///   Line 2 (main 写 Release, worker 读 Acquire): generation
///   Line 3 (worker 写, main 读): done
///   Line 4 (Drop 写): shutdown
#[repr(C, align(64))]
struct Shared {
    // --- Line 1: main 写, worker 读 (dispatch 时一次写入) ---
    func: AtomicUsize,
    ctx: AtomicPtr<()>,
    n_dispatch: AtomicUsize,
    // ★ 热降频根治: 限制活跃 worker 数 (tid >= n_active_workers 的 worker 深度 park)
    //   14 线程全负载 → 热积累 → 降频; 限制到 10 active → 4 核空闲散热 → 不降频
    n_active_workers: AtomicUsize,
    // ★ Work-stealing: 下一个待抢的 chunk 起始 (worker fetch_add 抢)
    next_chunk: AtomicUsize,
    // ★ Work-stealing: 总工作单元数 (抢到 >= total 即结束)
    total_work: AtomicUsize,
    // ★ Work-stealing: chunk 大小
    chunk_size: AtomicUsize,
    _pad1: [u8; 64 - 56], // 7×8=56B, 补到 64B

    // --- Line 2: main 写 Release, worker 读 Acquire (hot signal) ---
    generation: AtomicU64,
    _pad2: [u8; 64 - 8],

    // --- Line 3: worker 写 Release, main 读 Acquire (hot signal) ---
    done: AtomicUsize,
    _pad3: [u8; 64 - 8],

    // --- Line 4: shutdown (cold path) ---
    shutdown: AtomicBool,
    _pad4: [u8; 64 - 1], // AtomicBool 实际 1 字节, 补满 64B cache line
}

/// 类型擦除 trampoline: 把 *const () 转回 &F 并调用 F(i)
///
/// # Safety
/// ctx 必须指向有效的 F 实例, 且在调用期间保持存活
#[allow(unsafe_code)]
unsafe fn trampoline<F: Fn(usize)>(ctx: *const (), i: usize) {
    set_in_parallel(true);
    (&*(ctx as *const F))(i);
    set_in_parallel(false);
}

/// Worker 主循环: 自适应 wait 策略 (彻底解决热降频)
///
/// ★ 核心问题: 14 线程全负载 → 热积累 → 降频 (PERF% 309%→250%, decode 96→235ms)
/// ★ llama.cpp 方案: spin 6.5M + cond_wait, 持续中等功耗 (PERF% 稳定 260%)
/// ★ Daiza 方案: 7 active workers + park + main yield (稳定 212ms)
///
/// 三种 worker 状态:
/// 1. active worker (tid < n_active_workers=7) + park 模式:
///    spin 4096 + park: 零调度开销, worker park 期间 0% CPU
/// 2. idle worker (tid >= n_active_workers=7): 深度 park, 0% CPU
///    → 6 核空闲散热, 避免 14 核全负载降频
/// 3. main thread: scatter_wait 中 spin 4096 + yield, 避免持续 spin 占满 main 核
///    → main 核有散热间隙, 功耗平滑, 性能稳定 (212ms ±1ms)
///
/// ★ 实测 (Meteor Lake 14 核, 256t):
///   - 14 active + park: 235ms (降频, 不稳定)
///   - 7 active + park (main spin): 200-257ms (不降频, 但波动 57ms)
///   - 7 active + park (main yield): 212-213ms (不降频, 稳定 ±1ms) ← 当前方案
///   - llama.cpp 14 核: 154ms (稳定, 参考)
///
/// 为什么不能全 14 线程 spin 6.5M (llama.cpp 方案):
///   Daiza 每 token 257 barriers, 间隔 0.78ms < spin 2ms
///   spin 6.5M 期间 14 worker 占满 14 核, main thread 无法调度 (P0-F 教训)
///
/// ★ 实验: DAIZA_SPIN_ROUNDS env var 控制 spin 轮数 (默认 4096)
///   DAIZA_WAIT_MODE=park 回退到 park 方案
#[allow(unsafe_code)]
fn worker_loop(shared: Arc<Shared>, tid: usize) {
    const SPIN_ROUNDS_DEFAULT: usize = 4096;
    static SPIN_ROUNDS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let spin_rounds = *SPIN_ROUNDS.get_or_init(|| {
        std::env::var("DAIZA_SPIN_ROUNDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(SPIN_ROUNDS_DEFAULT)
    });
    let use_park = use_park_mode();
    let mut last_gen: u64 = 0;
    loop {
        if shared.shutdown.load(Ordering::Acquire) {
            break;
        }
        // ★ 读取 n_active_workers (Relaxed: 即使读到旧值也安全, 最多多 spin 一轮)
        let n_active = shared.n_active_workers.load(Ordering::Relaxed);
        // Acquire: 看到 generation 变化后, 读到 main 的 func/ctx/n_dispatch 写入
        let gen = shared.generation.load(Ordering::Acquire);
        if gen != last_gen {
            last_gen = gen;
            let n_dispatch = shared.n_dispatch.load(Ordering::Relaxed);
            if tid < n_dispatch {
                // Relaxed: generation 的 Acquire 已保证可见性
                let ctx = shared.ctx.load(Ordering::Relaxed);
                let func_ptr = shared.func.load(Ordering::Relaxed);
                let f: unsafe fn(*const (), usize) =
                    unsafe { std::mem::transmute(func_ptr) };
                unsafe { f(ctx, tid); }
                // Release: 确保 f(tid) 的内存写对 main 的 Acquire 可见
                shared.done.fetch_add(1, Ordering::Release);
            }
            // gen 变化但 tid >= n_dispatch: 不是给我的 dispatch, 继续循环
        } else if tid >= n_active {
            // ★ idle worker: 深度 park, 0% CPU (避免 14 核全负载降频)
            //   被 unpark 唤醒后不更新 last_gen: 让 loop 顶部的 gen != last_gen 检查
            //   决定是否执行任务 (scatter_wait 设置新 generation 后 unpark 会触发执行)
            //   spurious wakeup 时 gen == last_gen, 再次 park (安全)
            std::thread::park();
        } else if use_park {
            // ★ active worker + park 模式 (短跑): spin + park
            let mut spun = 0;
            while spun < spin_rounds {
                let g = shared.generation.load(Ordering::Acquire);
                if g != last_gen { break; }
                std::hint::spin_loop();
                spun += 1;
            }
            if spun >= spin_rounds {
                let g = shared.generation.load(Ordering::Acquire);
                if g == last_gen { std::thread::park(); }
            }
        } else {
            // ★ active worker + yield 模式 (长跑默认): spin + yield
            // spin spin_rounds cycles 检查 generation, 期满 yield_now 让出核心
            // yield 后立即回到 loop 顶部重新检查, 无 unpark 唤醒延迟
            let mut spun = 0;
            while spun < spin_rounds {
                let g = shared.generation.load(Ordering::Acquire);
                if g != last_gen { break; }
                std::hint::spin_loop();
                spun += 1;
            }
            if spun >= spin_rounds {
                std::thread::yield_now();
            }
        }
    }
}

/// 持久线程池: park/unpark 零分配 dispatch
///
/// ★ P0-C: 替代 mpsc channel, 消除每 barrier 的 Arc/Box/channel 开销
/// ★ P0-D: 主线程参与计算 (N-1 worker + main = N 线程匹配 N 核)
pub struct ThreadPool {
    shared: Arc<Shared>,
    threads: Vec<std::thread::Thread>,
    handles: Vec<std::thread::JoinHandle<()>>,
    n_workers: usize,
}

impl ThreadPool {
    /// 总线程数 (workers + 主线程)
    pub fn n_threads(&self) -> usize {
        self.n_workers + 1
    }
    /// 创建 n_threads-1 个 worker 线程 (主线程作为第 n_threads 个 worker)
    pub fn new(n_threads: usize) -> Self {
        let n_workers = n_threads.saturating_sub(1);
        let shared = Arc::new(Shared {
            func: AtomicUsize::new(0),
            ctx: AtomicPtr::new(std::ptr::null_mut()),
            n_dispatch: AtomicUsize::new(0),
            // 默认全部 worker 活跃 (向后兼容; main.rs 会根据 max_tokens 调整)
            n_active_workers: AtomicUsize::new(n_workers),
            next_chunk: AtomicUsize::new(0),
            total_work: AtomicUsize::new(0),
            chunk_size: AtomicUsize::new(0),
            _pad1: [0; 64 - 56],
            generation: AtomicU64::new(0),
            _pad2: [0; 64 - 8],
            done: AtomicUsize::new(0),
            _pad3: [0; 64 - 8],
            shutdown: AtomicBool::new(false),
            _pad4: [0; 64 - 1],
        });

        let mut threads = Vec::with_capacity(n_workers);
        let mut handles = Vec::with_capacity(n_workers);
        for tid in 0..n_workers {
            let shared = Arc::clone(&shared);
            let handle = std::thread::spawn(move || {
                worker_loop(shared, tid);
            });
            threads.push(handle.thread().clone());
            handles.push(handle);
        }

        Self { shared, threads, handles, n_workers }
    }

    /// 提交 N 个任务并等待全部完成 (零堆分配)
    ///
    /// 前 N-1 个任务分发给 worker 线程, 主线程执行第 N 个任务 (最后一个 chunk),
    /// 然后自旋等待所有 worker 完成。
    ///
    /// # Safety
    /// 本函数阻塞直到所有任务完成, 因此调用方在调用期间持有的借用仍然有效。
    /// 闭包应通过 raw pointer 访问外部数据, 避免生命周期冲突。
    pub fn scatter_wait<F>(&self, n: usize, f: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        if n == 0 {
            return;
        }
        if n == 1 {
            f(0);
            return;
        }

        let n_active = self.shared.n_active_workers.load(Ordering::Relaxed);

        // ★ dspark 性能修复: n > n_active+1 时用 stride 分块
        //   原方案 (n_dispatch = n-1) 会唤醒 idle worker (tid >= n_active), 导致:
        //   (1) park/unpark 开销 ~5μs × 5 idle workers = ~25μs/barrier
        //   (2) idle worker 多为 E-core, 执行慢, 成为 stride 瓶颈
        //   (3) dspark 每 forward ~257 barriers, 累积开销使 target forward 165→200ms (+21%)
        //   stride 分块: n_active+1 个 executor (active workers + main) 按 stride 分担 n 个 chunk
        //     - worker tid: 执行 f(tid), f(tid+stride), f(tid+2*stride), ...
        //     - main (tid=n_active): 执行 f(n_active), f(n_active+stride), ...
        //   优势: (1) 不唤醒 idle worker (避免 park/unpark + E-core 慢)
        //         (2) 无 chunk 跳过 (stride 覆盖全部 i=0..n-1)
        //         (3) 无 false sharing (f(i) 内部连续写 y[i*chunk..(i+1)*chunk], 不同 i 不重叠)
        //         (4) 负载匹配 P/E core: P-core (tid 0..5) 做 2 个 chunk, E-core (tid 6..8) 做 1 个
        //   注: f(i) 语义不变 — i 仍是虚拟 executor id, f 内部用 i 计算 start=i*chunk
        if n > n_active + 1 {
            let stride = n_active + 1;
            let n_dispatch = n_active; // 只唤醒 n_active 个 active worker

            self.shared.done.store(0, Ordering::Relaxed);

            // 用 StrideCtx 把 shared 和 f 打包 (stride_trampoline 从 shared 读 n_total/stride)
            let stride_ctx = StrideCtx {
                shared: self.shared.as_ref() as *const Shared,
                f: &f as *const F,
            };
            let ctx_ptr = &stride_ctx as *const StrideCtx<F> as *const ();
            let tramp = stride_trampoline::<F> as *const () as usize;
            self.shared.ctx.store(ctx_ptr as *mut (), Ordering::Relaxed);
            self.shared.func.store(tramp, Ordering::Relaxed);
            self.shared.n_dispatch.store(n_dispatch, Ordering::Relaxed);
            self.shared.total_work.store(n, Ordering::Relaxed); // n_total = n

            self.shared.generation.fetch_add(1, Ordering::Release);

            // ★ yield 模式下 active worker 自旋检测 generation, 无需 unpark (省 syscall)
            //   park 模式下 worker 可能已 park, 必须 unpark 唤醒
            if use_park_mode() {
                for i in 0..n_dispatch {
                    self.threads[i].unpark();
                }
            }

            // main 执行 stride 循环: f(n_active), f(n_active+stride), ...
            set_in_parallel(true);
            let mut i = n_active;
            while i < n {
                f(i);
                i += stride;
            }
            set_in_parallel(false);

            if n_dispatch > 0 {
                let mut spun = 0;
                while self.shared.done.load(Ordering::Acquire) < n_dispatch {
                    std::hint::spin_loop();
                    spun += 1;
                    if spun >= 4096 {
                        std::thread::yield_now();
                        spun = 0;
                    }
                }
            }
            return;
        }

        // n <= n_active+1: 每个 executor 最多 1 个 chunk, 走原逻辑
        // n_dispatch = n-1 (≤ n_active), 只唤醒 active worker, main 执行 chunk n-1
        let n_dispatch = n - 1;

        // 重置完成计数 (上一轮 worker 已全部完成, 无竞争)
        self.shared.done.store(0, Ordering::Relaxed);

        // 设置任务 (Relaxed: generation 的 Release 会保证可见性)
        let f_ptr = &f as *const F as *const ();
        let tramp = trampoline::<F> as *const () as usize;
        self.shared.ctx.store(f_ptr as *mut (), Ordering::Relaxed);
        self.shared.func.store(tramp, Ordering::Relaxed);
        self.shared.n_dispatch.store(n_dispatch, Ordering::Relaxed);

        // 递增 generation (Release: 让 worker 看到 ctx/func/n_dispatch 写入)
        self.shared.generation.fetch_add(1, Ordering::Release);

        // ★ yield 模式下 active worker 自旋检测 generation, 无需 unpark (省 syscall)
        //   park 模式下 worker 可能已 park, 必须 unpark 唤醒
        if use_park_mode() {
            for i in 0..n_dispatch {
                self.threads[i].unpark();
            }
        }

        // 主线程执行最后一个 chunk (tid = n-1)
        set_in_parallel(true);
        f(n - 1);
        set_in_parallel(false);

        // 等待所有 worker 完成 (Acquire: 看到 done 后, 读到 worker 的内存写)
        if n_dispatch > 0 {
            let mut spun = 0;
            while self.shared.done.load(Ordering::Acquire) < n_dispatch {
                std::hint::spin_loop();
                spun += 1;
                if spun >= 4096 {
                    std::thread::yield_now();
                    spun = 0;
                }
            }
        }
    }

    /// ★ Work-stealing dispatch: 把 total_work 个工作单元按 chunk_size 切分,
    /// worker + 主线程通过 atomic fetch_add 抢 chunk, 直到全部抢完。
    ///
    /// 与 `scatter_wait` 的区别:
    /// - scatter_wait: 静态分块 (tid × chunk), 快线程完成自己的 chunk 后等待慢线程
    /// - scatter_wait_stealing: 动态抢 chunk, 快线程可多抢, 改善负载均衡
    ///
    /// 适用场景: 工作单元负载不均 (如 matvec_multi 跨矩阵边界), 或某 worker 被抢占。
    ///
    /// # Safety
    /// 本函数阻塞直到所有 chunk 完成, F 生命周期安全。
    pub fn scatter_wait_stealing<F>(&self, total_work: usize, chunk_size: usize, f: F)
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        if total_work == 0 {
            return;
        }
        if self.n_workers == 0 {
            // 无 worker, 主线程顺序执行所有 chunk
            let mut start = 0;
            while start < total_work {
                let end = (start + chunk_size).min(total_work);
                f(start, end);
                start = end;
            }
            return;
        }

        let n_dispatch = self.shared.n_active_workers.load(Ordering::Relaxed);

        // 重置完成计数
        self.shared.done.store(0, Ordering::Relaxed);

        // 设置 work-stealing 参数
        self.shared.next_chunk.store(0, Ordering::Relaxed);
        self.shared.total_work.store(total_work, Ordering::Relaxed);
        self.shared.chunk_size.store(chunk_size, Ordering::Relaxed);

        // 设置闭包 (worker 内 steal_loop 会调用)
        // 用 StealCtx 把 shared 指针和 f 指针打包, 通过 ctx 传给 worker
        let steal_ctx = StealCtx {
            shared: self.shared.as_ref() as *const Shared,
            f: &f as *const F,
        };
        let ctx_ptr = &steal_ctx as *const StealCtx<F> as *const ();
        let tramp = steal_trampoline::<F> as *const () as usize;
        self.shared.ctx.store(ctx_ptr as *mut (), Ordering::Relaxed);
        self.shared.func.store(tramp, Ordering::Relaxed);
        self.shared.n_dispatch.store(n_dispatch, Ordering::Relaxed);

        // 递增 generation (Release: 让 worker 看到 ctx/func/n_dispatch/next_chunk 写入)
        self.shared.generation.fetch_add(1, Ordering::Release);

        // ★ yield 模式下 active worker 自旋检测 generation, 无需 unpark (省 syscall)
        //   park 模式下 worker 可能已 park, 必须 unpark 唤醒
        if use_park_mode() {
            for i in 0..n_dispatch {
                self.threads[i].unpark();
            }
        }

        // 主线程也参与抢 chunk (像 llama.cpp)
        set_in_parallel(true);
        steal_loop(&self.shared, &f);
        set_in_parallel(false);

        // 等待所有 worker 完成 (Acquire: 看到 done 后, 读到 worker 的内存写)
        // ★ 热降频根治: main spin + yield 混合, 避免持续 spin 占满 main 核
        //   (与 scatter_wait 一致, 见 scatter_wait 注释)
        if n_dispatch > 0 {
            let mut spun = 0;
            while self.shared.done.load(Ordering::Acquire) < n_dispatch {
                std::hint::spin_loop();
                spun += 1;
                if spun >= 4096 {
                    std::thread::yield_now();
                    spun = 0;
                }
            }
        }
    }

    /// ★ 热降频根治: 设置活跃 worker 数量
    ///
    /// tid >= n_active 的 worker 进入深度 park (0% CPU), 不参与 dispatch。
    /// 用于长跑时减少活跃核心数, 避免 14 核全负载导致热降频。
    ///
    /// 调用后立即 unpark 所有 worker, 让 idle worker 重新检查 n_active 并 park。
    ///
    /// 典型用法:
    /// - 短跑 (≤64t): set_active_workers(n_workers) — 全核利用, 短跑不降频
    /// - 长跑 (>64t): set_active_workers(9) — 10 核活跃 (9 worker + main), 4 核散热
    pub fn set_active_workers(&self, n_active: usize) {
        let n = n_active.min(self.n_workers);
        self.shared.n_active_workers.store(n, Ordering::Relaxed);
        // unpark 所有 worker, 让 idle worker 重新检查 n_active 并 park
        // active worker 被 unpark 后无副作用 (park 是 sticky 的)
        for thread in &self.threads {
            thread.unpark();
        }
    }
}

/// Work-stealing 循环: 反复 fetch_add 抢 chunk, 直到 next_chunk >= total_work
#[inline]
fn steal_loop<F: Fn(usize, usize)>(shared: &Shared, f: &F) {
    let total = shared.total_work.load(Ordering::Relaxed);
    let chunk = shared.chunk_size.load(Ordering::Relaxed);
    if chunk == 0 || total == 0 {
        return;
    }
    loop {
        // Acquire: 与 main 的 Release store 同步, 确保读到正确的 total/chunk
        let start = shared.next_chunk.fetch_add(chunk, Ordering::Relaxed);
        if start >= total {
            return;
        }
        let end = (start + chunk).min(total);
        f(start, end);
    }
}

/// Work-stealing context: 把 shared 指针和闭包指针打包, 通过 ctx 传给 worker
#[repr(C)]
struct StealCtx<F> {
    shared: *const Shared,
    f: *const F,
}

/// ★ Stride context: scatter_wait 在 n > n_active+1 时用 stride 分块
///   worker tid 执行 f(tid), f(tid+stride), f(tid+2*stride), ... 直到 i >= n_total
///   stride = n_active + 1 (active workers + main), n_total 存在 total_work 中
#[repr(C)]
struct StrideCtx<F> {
    shared: *const Shared,
    f: *const F,
}

/// ★ Stride trampoline: worker 调用此函数, 内部按 stride 循环执行 f(i)
///   不 fetch_add done — done 由 worker_loop 统一处理 (与 trampoline 一致)
#[allow(unsafe_code)]
unsafe fn stride_trampoline<F: Fn(usize)>(ctx: *const (), tid: usize) {
    let sc = &*(ctx as *const StrideCtx<F>);
    let shared = &*sc.shared;
    let f = &*sc.f;
    // n_total 存在 total_work (复用 scatter_wait_stealing 的字段)
    let n_total = shared.total_work.load(Ordering::Relaxed);
    // stride = n_active + 1 (active workers + main)
    let stride = shared.n_active_workers.load(Ordering::Relaxed) + 1;
    set_in_parallel(true);
    let mut i = tid;
    while i < n_total {
        f(i);
        i += stride;
    }
    set_in_parallel(false);
}

/// Work-stealing trampoline: worker 调用此函数, 内部抢完所有 chunk 后返回
#[allow(unsafe_code)]
unsafe fn steal_trampoline<F: Fn(usize, usize)>(ctx: *const (), _tid: usize) {
    let sc = &*(ctx as *const StealCtx<F>);
    let shared = &*sc.shared;
    let f = &*sc.f;
    set_in_parallel(true);
    steal_loop(shared, f);
    set_in_parallel(false);
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        for thread in &self.threads {
            thread.unpark();
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// 工作线程数,由 `DAIZA_THREADS` env var 控制,默认所有逻辑核
///
/// 缓存结果避免每次调用 `std::env::var`。第一次调用时初始化。
///
/// ★ P2-4 修正: 原 `(logical/2).max(1)` 假设 SMT (logical=物理×2), 但无 SMT 的 CPU
///   (如 Intel Core Ultra 5 225H, 6P+8E=14核无SMT) 会浪费一半核数。
///   实测 14 threads vs 7 threads: decode -18% (261→214ms/tok)。
///   改为默认用所有逻辑核; 有 SMT 的 CPU 如需避开超线程可手动设 DAIZA_THREADS。
pub fn thread_count() -> usize {
    use std::sync::OnceLock;
    static N_THREADS: OnceLock<usize> = OnceLock::new();
    *N_THREADS.get_or_init(|| {
        if let Ok(s) = std::env::var("DAIZA_THREADS") {
            if let Ok(n) = s.parse::<usize>() {
                if n > 0 {
                    return n;
                }
            }
        }
        // 默认:所有逻辑核
        // 无 SMT CPU (如 Arrow Lake): logical = 物理核数,全部使用
        // 有 SMT CPU (如 Alder Lake P+HT): logical = 物理×2, SMT 线程在内存带宽场景仍能贡献
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    })
}

#[derive(Default)]
pub struct Workspace {
    // === block.rs 用 ===
    // ★ P0-3: block_normed 复用为 mlp 输入(attention/ssm 完成后即死,post_norm 可覆盖)
    //   原 block_mlp_in 已删除,省 hidden × 4B = 20KB
    /// `[hidden]` attn_norm 输出 / post_attention_norm 输出(分阶段复用)
    pub block_normed: Vec<f32>,

    // === attention.rs 用 ===
    /// `[n_q_heads * head_dim * 2]` = 12288,attn_q matvec 输出(Q + gate 交错)
    pub attn_q_total: Vec<f32>,
    /// `[n_q_heads * head_dim]` = 6144,解交错后的 Q
    pub attn_q: Vec<f32>,
    /// `[n_q_heads * head_dim]` = 6144,解交错后的 gate
    pub attn_gate: Vec<f32>,
    /// `[n_kv_heads * head_dim]` = 1024,K 投影
    pub attn_k: Vec<f32>,
    /// `[n_kv_heads * head_dim]` = 1024,V 投影
    pub attn_v: Vec<f32>,
    /// `[n_q_heads * head_dim]` = 6144,attention 输出(被 gate 调制后)
    pub attn_out: Vec<f32>,

    // === ssm.rs 用 ===
    /// `[qkv_full_len]` = 10240,attn_qkv matvec 输出
    pub ssm_qkv: Vec<f32>,
    /// `[qkv_dim]` = 2048,conv 后的 q
    pub ssm_q: Vec<f32>,
    /// `[qkv_dim]` = 2048,conv 后的 k
    pub ssm_k: Vec<f32>,
    /// `[ssm_inner]` = 6144,conv 后的 v
    pub ssm_v: Vec<f32>,
    /// `[qkv_full_len]` = 10240,conv1d 输出
    pub ssm_conv_out: Vec<f32>,
    /// `[ssm_inner]` = 6144,scan 输出 y
    pub ssm_y: Vec<f32>,
    /// `[num_v_heads]` = 48,alpha 投影
    pub ssm_alpha: Vec<f32>,
    /// `[num_v_heads]` = 48,beta 投影
    pub ssm_beta: Vec<f32>,
    /// `[ssm_inner]` = 6144,output gate z = attn_gate @ x
    pub ssm_z: Vec<f32>,

    // === mlp.rs 用 ===
    /// `[d_ff]` = 17408,gate 投影
    pub mlp_gate: Vec<f32>,
    /// `[d_ff]` = 17408,up 投影
    pub mlp_up: Vec<f32>,
    // mlp_down 结果通过 `matvec_add_into_slice` 直接累加到 h_buf,无需独立 buffer
}

impl Workspace {
    pub fn new(cfg: &Config) -> Self {
        let hidden = cfg.hidden;
        let head_dim = cfg.head_dim;
        let n_q_heads = cfg.head_count;
        let n_kv_heads = cfg.head_count_kv;
        let d_ff = cfg.feed_forward_length;
        let ssm_inner = cfg.ssm_inner_size;
        let qkv_dim = cfg.ssm_group_count * cfg.ssm_state_size;
        let qkv_full_len = 2 * qkv_dim + ssm_inner;
        let num_v_heads = cfg.ssm_time_step_rank;

        Self {
            block_normed: vec![0.0; hidden],

            attn_q_total: vec![0.0; n_q_heads * head_dim * 2],
            attn_q: vec![0.0; n_q_heads * head_dim],
            attn_gate: vec![0.0; n_q_heads * head_dim],
            attn_k: vec![0.0; n_kv_heads * head_dim],
            attn_v: vec![0.0; n_kv_heads * head_dim],
            attn_out: vec![0.0; n_q_heads * head_dim],

            ssm_qkv: vec![0.0; qkv_full_len],
            ssm_q: vec![0.0; qkv_dim],
            ssm_k: vec![0.0; qkv_dim],
            ssm_v: vec![0.0; ssm_inner],
            ssm_conv_out: vec![0.0; qkv_full_len],
            ssm_y: vec![0.0; ssm_inner],
            ssm_alpha: vec![0.0; num_v_heads],
            ssm_beta: vec![0.0; num_v_heads],
            ssm_z: vec![0.0; ssm_inner],

            mlp_gate: vec![0.0; d_ff],
            mlp_up: vec![0.0; d_ff],
        }
    }
}
