//! Independent FP64 two-sided operators and transport between real off-table prefixes.
use super::*;
use kernite_cuda::{Cuda, Grid};
use memory_checks::linear;

fn rotate(x: &[f64], frequency: &[f32], t: f64) -> Vec<f64> {
    let mut z = x.to_vec();
    let pairs = x.len() / 2;
    for (j, &w) in frequency.iter().enumerate() {
        let (s, c) = (t * f64::from(w)).sin_cos();
        z[j] = c * x[j] - s * x[pairs + j];
        z[pairs + j] = s * x[j] + c * x[pairs + j];
    }
    z
}

fn shifted_controls(length: usize) -> Result<()> {
    let (d, rows) = (14, 3 * length);
    let mut rng = random::Random::new(107);
    let mut sample = |n| {
        (0..n)
            .map(|_| rng.gauss(0., 0.33) as f32)
            .collect::<Vec<_>>()
    };
    let ctrl = sample(rows * 3);
    let din = sample(rows);
    let dout = sample(rows);
    let upstream = sample(rows * d);
    let rotated = sample(rows * d);
    let cuda = Cuda::new()?;
    let ops = two_frame::kernels::Kernels::load(&cuda)?;
    let (ci, di, do_, up, rot) = (
        cuda.upload(&ctrl)?,
        cuda.upload(&din)?,
        cuda.upload(&dout)?,
        cuda.upload(&upstream)?,
        cuda.upload(&rotated)?,
    );
    let (pi, po, g, dc, db) = (
        cuda.alloc::<f32>(rows)?,
        cuda.alloc::<f32>(rows)?,
        cuda.alloc::<f32>(rows)?,
        cuda.alloc::<f32>(rows * 3)?,
        cuda.alloc::<f32>(rows * d)?,
    );
    let grid = Grid::new(rows as u32, 1);
    unsafe {
        ops.positions(grid, pi.ptr(), po.ptr(), g.ptr(), ci.ptr(), length as u32)?;
        ops.controls_back(
            grid,
            dc.ptr(),
            db.ptr(),
            di.ptr(),
            do_.ptr(),
            up.ptr(),
            rot.ptr(),
            d as u32,
            length as u32,
        )?;
    }
    let (pi, po, g, dc, db) = (pi.read()?, po.read()?, g.read()?, dc.read()?, db.read()?);
    for r in 0..rows {
        let expected = if r.is_multiple_of(length) {
            [0., 0., 1.]
        } else {
            let c = &ctrl[3 * (r - 1)..3 * r];
            [c[0], c[1], 1. + c[2]]
        };
        if [pi[r], po[r], g[r]] != expected {
            return Err("frame controls crossed a sequence boundary".into());
        }
        let mut dg = [0.; 3];
        let mut bias = vec![0_f64; d];
        if !(r + 1).is_multiple_of(length) {
            dg[0] = f64::from(din[r + 1]);
            dg[1] = f64::from(dout[r + 1]);
            for (j, b) in bias.iter_mut().enumerate() {
                let at = (r + 1) * d + j;
                *b = f64::from(upstream[at]);
                dg[2] += *b * f64::from(rotated[at]);
            }
        }
        phase_checks::error(&dc[3 * r..3 * r + 3], &dg, 2e-6)?;
        phase_checks::error(&db[r * d..r * d + d], &bias, 0.)?;
    }
    println!("two-frame strict-prefix control gradients length={length} rows={rows}");
    Ok(())
}

fn as_f64(x: &[f32]) -> Vec<f64> {
    x.iter().copied().map(f64::from).collect()
}

fn transport(
    e: &mut Engine,
    weights: &[f32],
    control: usize,
    bias: usize,
    matrix: usize,
    frequency: usize,
) -> Result<()> {
    let d = e.config.width;
    let mut rng = random::Random::new(341);
    let mut input = (0..4 * d)
        .map(|_| rng.gauss(0., 0.3) as f32)
        .collect::<Vec<_>>();
    let mut centered = weights.to_vec();
    centered[control + 2 * d..matrix].fill(0.);
    e.parameters.write(&centered)?;
    e.infer_embeddings(&input, 2, 2)?;
    let h = e.diagnostic_snapshot(4)?;
    let prev = &h["readout_input"];
    let left = linear(&weights[control..bias], &as_f64(&prev[..d]));
    let right = linear(&weights[control..bias], &as_f64(&prev[2 * d..3 * d]));
    let (a, b) = (left[0] - right[0], left[1] - right[1]);
    if a.abs() < 1e-3 || b.abs() < 1e-3 {
        return Err("inactive frame transport fixture".into());
    }
    let freq = &weights[frequency..frequency + d / 2];
    let next = rotate(&as_f64(&input[d..2 * d]), freq, a);
    for (x, y) in input[3 * d..].iter_mut().zip(next) {
        *x = y as f32;
    }
    let hidden = e.infer_embeddings(&input, 2, 2)?.hidden;
    let changed = rotate(
        &as_f64(&hidden[3 * d..]),
        &weights[frequency + d / 2..frequency + d],
        b,
    );
    let error = phase_checks::error(&hidden[d..2 * d], &changed, 3e-5)?;
    if hidden[d..2 * d] == hidden[3 * d..] {
        return Err("distinct output frames produced the same fixture".into());
    }
    let mut constant = weights.to_vec();
    let norm: f64 = prev[..d].iter().map(|&v| f64::from(v).powi(2)).sum();
    for (w, &x) in constant[control + 2 * d..bias].iter_mut().zip(&prev[..d]) {
        *w = (-f64::from(x) / norm) as f32;
    }
    e.parameters.write(&constant)?;
    input.copy_within(..d, 2 * d);
    let hidden = e.infer_embeddings(&input, 2, 2)?.hidden;
    let expected = linear(&weights[bias..matrix], &as_f64(&prev[..d]));
    let e1 = phase_checks::error(&hidden[d..2 * d], &expected, 3e-5)?;
    let e2 = phase_checks::error(&hidden[3 * d..], &expected, 3e-5)?;
    println!(
        "two-frame width={d} off_table_transport={error:.3e} constant_row={:.3e}",
        e1.max(e2)
    );
    e.parameters.write(weights)?;
    Ok(())
}

fn network(d: usize) -> Result<()> {
    let mut cfg = config(2)?;
    cfg.head = architecture::HEAD;
    cfg.width = d;
    cfg.heads = d / 32;
    cfg.gain = cfg.heads as f32 / d as f32;
    cfg.vocab = 19;
    cfg.max_length = 5;
    let first = architecture::start(&cfg);
    let control = architecture::memory_start(&cfg);
    let bias = control + 3 * d;
    let matrix = bias + d * d;
    let frequency = matrix + d * d;
    let end = layout::frequency_offset(&cfg);
    if end != frequency + d {
        return Err("two-frame parameter layout mismatch".into());
    }
    let weights = layout::initialize_model(&cfg, 54);
    let mut e = Engine::new(cfg.clone(), &weights)?;
    let tokens = [1, 3, 5, 7, 9, 9, 8, 6, 4, 2];
    let labels = [u32::MAX, 6, 2, 4, 8, 7, u32::MAX, 5, 3, 1];
    e.forward(&tokens, &labels, 2, 5)?;
    let diag = e.diagnostic_snapshot(tokens.len())?;
    let mut expected = Vec::new();
    for (xs, hs) in diag["embedding_normalized"]
        .chunks_exact(5 * d)
        .zip(diag["readout_input"].chunks_exact(5 * d))
    {
        let mut previous = vec![0.; d];
        for (x, h) in xs.chunks_exact(d).zip(hs.chunks_exact(d)) {
            let pos = linear(&weights[control..bias], &previous);
            let x = rotate(&as_f64(x), &weights[frequency..frequency + d / 2], pos[0]);
            let x = linear(&weights[matrix..frequency], &x);
            let x = rotate(&x, &weights[frequency + d / 2..end], pos[1]);
            let y: Vec<_> = linear(&weights[bias..matrix], &previous)
                .into_iter()
                .zip(x)
                .map(|(b, y)| b + (1. + pos[2]) * y)
                .collect();
            expected.extend(linear(&weights[..cfg.vocab * d], &y));
            previous = as_f64(h);
        }
    }
    let error = phase_checks::error(&e.logits.read()?, &expected, 2e-5)?;
    let causal = memory_checks::prefix_error(&mut e, &tokens[..5])?;
    if causal > 2e-5 {
        return Err("a future input changed a frame feature".into());
    }
    println!("two-frame full network width={d} max_logit_error={error:.3e} prefix={causal:.3e}");
    transport(&mut e, &weights, control, bias, matrix, frequency)?;
    let router = layout::router_offset(&cfg, 0);
    memory_checks::check_directions(
        &mut e,
        &weights,
        &tokens,
        &labels,
        5,
        [
            ("tied embedding", 0..cfg.vocab * d),
            ("context router", router..router + 96 * d),
            ("stream read", first..control),
            ("input position", control..control + d),
            ("output position", control + d..control + 2 * d),
            ("gain", control + 2 * d..bias),
            ("bias", bias..matrix),
            ("shared map", matrix..frequency),
            ("input frequencies", frequency..frequency + d / 2),
            ("output frequencies", frequency + d / 2..end),
        ]
        .into_iter()
        .map(|(name, range)| (name, range.collect()))
        .collect(),
    )
}

pub fn run() -> Result<()> {
    if !precision::FP32_AFFINE_REFERENCE {
        return Err("frame checks require FP32 reference affines".into());
    }
    for length in [1, 7, 257] {
        shifted_controls(length)?;
    }
    for d in [64, 128] {
        network(d)?;
    }
    Ok(())
}
