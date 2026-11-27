//! The time of full-batch updates, against the peaks of this GPU measured by
//! the kernels below: FP32 fused multiply-adds and DRAM copies.
use crate::{
    Result, metrics,
    runner::{Cell, FullBatch},
};
use kernite_cuda::{Cuda, Grid};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};

#[kernite_macros::kernels]
mod device {
    /// Eight independent chains of `16 * rounds` fused multiply-adds in each
    /// thread. The sum of the chains is stored, so that none is removed.
    pub unsafe extern "ptx-kernel" fn fma_peak(out: *mut f32, rounds: u32, a: f32, b: f32) {
        let (t, u, n, i, block, threads): (u32, u32, u32, u32, u32, u32);
        mov.u32(t, tid.x);
        let acc: [f32; 8];
        for j in 0..8 {
            add.u32(u, t, j);
            cvt.rn.f32.u32(acc[j], u);
        }
        mov.u32(n, rounds);
        let more: pred;
        label(again);
        for r in 0..16 {
            for j in 0..8 {
                fma.rn.f32(acc[j], acc[j], a, b);
            }
        }
        sub.u32(n, n, 1);
        setp.ne.u32(more, n, 0);
        if more {
            bra(again);
        }
        for j in 1..8 {
            add.rn.f32(acc[0], acc[0], acc[j]);
        }
        mov.u32(block, ctaid.x);
        mov.u32(threads, ntid.x);
        mad.lo.u32(i, block, threads, t);
        let p: u64;
        mad.wide.u32(p, i, 4, out);
        st.global.f32([p], acc[0]);
    }

    /// Copies `count` groups of four floats, each thread striding by the grid.
    pub unsafe extern "ptx-kernel" fn copy(dst: *mut f32, src: *const f32, count: u32) {
        let (t, block, threads, blocks, i, stride): (u32, u32, u32, u32, u32, u32);
        mov.u32(t, tid.x);
        mov.u32(block, ctaid.x);
        mov.u32(threads, ntid.x);
        mov.u32(blocks, nctaid.x);
        mad.lo.u32(i, block, threads, t);
        mul.lo.u32(stride, blocks, threads);
        let (x0, x1, x2, x3): (f32, f32, f32, f32);
        let (ps, pd): (u64, u64);
        let done: pred;
        label(again);
        setp.ge.u32(done, i, count);
        if done {
            ret();
        }
        mad.wide.u32(ps, i, 16, src);
        ld.global.v4.f32((x0, x1, x2, x3), [ps]);
        mad.wide.u32(pd, i, 16, dst);
        st.global.v4.f32([pd], (x0, x1, x2, x3));
        add.u32(i, i, stride);
        bra(again);
    }
}

/// The shortest of `trials` timings of `work` on the GPU, in seconds.
fn best(cuda: &Cuda, trials: usize, mut work: impl FnMut() -> Result<()>) -> Result<f64> {
    let (start, stop) = (cuda.timed_event()?, cuda.timed_event()?);
    let mut shortest = f64::INFINITY;
    for _ in 0..trials {
        start.record()?;
        work()?;
        stop.record()?;
        stop.sync()?;
        shortest = shortest.min(stop.duration_since(&start)?.as_secs_f64());
    }
    Ok(shortest)
}

/// FP32 operations per second, two per fused multiply-add, and DRAM bytes
/// per second, read and written, as this GPU reaches them. The clock of a
/// laptop GPU rises under load, so the multiply-adds first run for a second.
pub fn peaks(cuda: &Cuda) -> Result<(f64, f64)> {
    let kernels = device::Kernels::load(cuda)?;
    let (blocks, threads, rounds) = (8 * cuda.multiprocessors(), 256, 4096);
    let out = cuda.alloc::<f32>((blocks * threads) as usize)?;
    let mut fma = || {
        // SAFETY: one store per thread, inside `out`.
        unsafe {
            kernels.fma_peak(Grid::new(blocks, threads), out.ptr(), rounds, 0.999, 1e-3)?;
        }
        Ok(())
    };
    let warm = Instant::now();
    while warm.elapsed().as_secs_f64() < 1.0 {
        best(cuda, 1, &mut fma)?;
    }
    let fma = best(cuda, 20, fma)?;
    let flops = 2.0 * 128.0 * f64::from(blocks * threads) * f64::from(rounds) / fma;
    let floats = 1 << 25;
    let (src, dst) = (cuda.alloc::<f32>(floats)?, cuda.alloc::<f32>(floats)?);
    let copy = best(cuda, 20, || {
        // SAFETY: the groups of four floats lie inside both buffers.
        unsafe {
            kernels.copy(
                Grid::new(16 * cuda.multiprocessors(), 256),
                dst.ptr(),
                src.ptr(),
                (floats / 4) as u32,
            )?;
        }
        Ok(())
    })?;
    Ok((flops, 2.0 * 4.0 * floats as f64 / copy))
}

fn peak_report(cuda: &Cuda) -> Result<Value> {
    let (flops, bytes) = peaks(cuda)?;
    Ok(json!({"gpu":cuda.name(),"fma_tflops":flops / 1e12,"copy_gbps":bytes / 1e9}))
}

pub fn peak() -> Result<()> {
    println!("{}", peak_report(&Cuda::new()?)?);
    Ok(())
}

/// Times `updates` full-batch updates of seed 0 of `name`, after one update
/// that counts the matrix-product operations and one that warms up, then
/// measures the peaks on the GPU so warmed.
pub fn bench(name: &str, updates: usize) -> Result<()> {
    if updates == 0 {
        return Err("bench needs at least one update".into());
    }
    let cell = Cell::new(name, 0, 1, Path::new("."))?;
    let mut engine = cell.engine()?;
    let train = &cell.task.splits["train"];
    let mut packed = metrics::Packed::default();
    let full = FullBatch::new(&engine, train, &mut packed, cell.batch)?;
    let (flops, positions, parts) = full.count(&mut engine, train, &mut packed)?;
    full.update(&mut engine, train, &mut packed, cell.schedule.rate(0))?;
    engine.cuda.sync()?;
    let (start, stop) = (engine.cuda.timed_event()?, engine.cuda.timed_event()?);
    let wall = Instant::now();
    start.record()?;
    for step in 1..=updates {
        full.update(&mut engine, train, &mut packed, cell.schedule.rate(step))?;
    }
    stop.record()?;
    stop.sync()?;
    let gpu = stop.duration_since(&start)?.as_secs_f64() / updates as f64;
    let wall = wall.elapsed().as_secs_f64() / updates as f64;
    let peaks = peak_report(&engine.cuda)?;
    let peak = peaks["fma_tflops"].as_f64().ok_or("missing peak")? * 1e12;
    let s = &cell.shape;
    println!(
        "{}",
        json!({"case":name,"width":s.width,"heads":s.heads,"layers":s.layers,"vocab":s.vocab,
            "batch":cell.batch,"microbatches":parts,"examples":train.len(),"positions":positions,
            "updates":updates,"ms":1e3 * gpu,"wall_ms":1e3 * wall,
            "product_gflop":flops / 1e9,"product_tflops":flops / gpu / 1e12,
            "product_bound_ms":1e3 * flops / peak,"fraction":flops / peak / gpu,"peaks":peaks})
    );
    Ok(())
}
