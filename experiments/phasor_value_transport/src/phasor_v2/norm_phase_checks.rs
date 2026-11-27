//! Float64 reference for the single-stream skip in Transformer.PhasorV2.PhaseResidual.
use super::{Engine, Result};
use kernite_cuda::Grid;

fn rotated(input: &[f64], width: usize, theta: f64) -> Vec<f64> {
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(2 * width) {
        let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        let (sine, cosine) = (theta * norm).sin_cos();
        for z in row.chunks_exact(2) {
            output.extend([cosine * z[0] - sine * z[1], sine * z[0] + cosine * z[1]]);
        }
    }
    output
}

/// Include a zero token and a width spanning two warp iterations. Preload the
/// input and coefficient adjoints to test addition to an existing gradient.
pub(super) fn check(engine: &Engine) -> Result<()> {
    let width = 35;
    let rows = 3;
    let size = 2 * width * rows;
    let input: Vec<f32> = (0..size)
        .map(|i| {
            if i < 2 * width {
                0.0
            } else {
                (i as f32 * 0.31).sin()
            }
        })
        .collect();
    let upstream: Vec<f32> = (0..size).map(|i| (i as f32 * 0.27).cos()).collect();
    let theta = 0.23_f32;
    let x = engine.cuda.upload(&input)?;
    let y = engine.cuda.alloc::<f32>(size)?;
    let dy = engine.cuda.upload(&upstream)?;
    let dx = engine.cuda.upload(&vec![0.25_f32; size])?;
    let coefficient = engine.cuda.upload(&[theta])?;
    let dtheta = engine.cuda.upload(&[-0.125_f32])?;
    let grid = Grid::new(rows as u32, 32);
    // SAFETY: each warp owns a full row in all four vectors. The coefficient
    // and its accumulated adjoint each have one real coordinate.
    unsafe {
        engine.kernel.turn_by_norm::<false>(
            grid,
            y.ptr(),
            x.ptr(),
            dy.ptr(),
            dx.ptr(),
            coefficient.ptr(),
            dtheta.ptr(),
            width as u32,
        )?;
        engine.kernel.turn_by_norm::<true>(
            grid,
            y.ptr(),
            x.ptr(),
            dy.ptr(),
            dx.ptr(),
            coefficient.ptr(),
            dtheta.ptr(),
            width as u32,
        )?;
    }
    let mut input64: Vec<_> = input.into_iter().map(f64::from).collect();
    let theta64 = f64::from(theta);
    let objective = |values: &[f64], theta| {
        rotated(values, width, theta)
            .iter()
            .zip(&upstream)
            .map(|(&z, &g)| z * f64::from(g))
            .sum::<f64>()
    };
    let delta = 1e-6;
    let mut expected = rotated(&input64, width, theta64);
    for i in 0..size {
        let original = input64[i];
        input64[i] = original + delta;
        let plus = objective(&input64, theta64);
        input64[i] = original - delta;
        let minus = objective(&input64, theta64);
        input64[i] = original;
        expected.push(0.25 + (plus - minus) / (2.0 * delta));
    }
    expected.push(
        -0.125
            + (objective(&input64, theta64 + delta) - objective(&input64, theta64 - delta))
                / (2.0 * delta),
    );
    let actual = y
        .read()?
        .into_iter()
        .chain(dx.read()?)
        .chain(dtheta.read()?);
    let mut maximum = 0.0_f64;
    for (value, reference) in actual.zip(expected) {
        let error = (f64::from(value) - reference).abs();
        if !error.is_finite() || error > 2e-5 * (1.0 + reference.abs()) {
            return Err(format!("norm phase: GPU {value}, reference {reference}").into());
        }
        maximum = maximum.max(error);
    }
    println!("Norm phase FP64 forward and finite-difference adjoints: max error {maximum:.3e}");
    Ok(())
}
