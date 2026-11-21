//! Scalar forward oracle and an end-to-end directional derivative through the tied embedding.
use super::*;

fn reference(scores: &[f32], e: &[f32], width: usize, offset: usize) -> Vec<f64> {
    let peak = f64::from(scores.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    let masses: Vec<_> = scores
        .iter()
        .map(|&s| (f64::from(s) - peak).exp())
        .collect();
    let total: f64 = masses.iter().sum();
    let mut value = vec![0.; width];
    for (s, mass) in masses.iter().enumerate() {
        let index = (s + offset) % scores.len();
        for (v, &x) in value.iter_mut().zip(&e[index * width..]) {
            *v += (mass / total) * f64::from(x);
        }
    }
    let rms =
        (value.iter().map(|x| x * x).sum::<f64>() / width as f64 + f64::from(f32::EPSILON)).sqrt();
    e.chunks_exact(width)
        .map(|code| {
            code.iter()
                .zip(&value)
                .map(|(&x, &v)| f64::from(x) * v / rms)
                .sum()
        })
        .collect()
}

pub fn run() -> Result<()> {
    if !precision::FP32_AFFINE_REFERENCE {
        return Err("the finite-difference check requires --features fp32-affine-reference".into());
    }
    // Uneven batches, repeated first operand, both sides of modular wrap, zero product.
    let rows = [
        [13, 19, 31],
        [0, 96, 11],
        [13, 96, 78],
        [55, 2, 88],
        [96, 96, 0],
    ];
    let cfg = config(8)?;
    let (p, d) = (cfg.vocab, cfg.width);
    let initial = layout::initialize_model(&cfg, SEED);
    let mut e = Engine::new(cfg, &initial)?;
    let l = ValueLayer::new(&e)?;
    for product in [false, true] {
        e.parameters.write(&initial)?;
        forward(&mut e, &rows)?;
        let scores = e.logits.read()?;
        l.forward(&e, rows.len(), product)?;
        let actual = e.logits.read()?;
        let mut worst = 0_f64;
        for (i, &[a, b, _]) in rows.iter().enumerate() {
            let offset = if product { (a * b) as usize % p } else { 0 };
            let gold = reference(
                &scores[2 * i * p..(2 * i + 1) * p],
                &initial[..p * d],
                d,
                offset,
            );
            for (&v, expected) in actual[(2 * i + 1) * p..].iter().zip(gold) {
                worst = worst.max((f64::from(v) - expected).abs());
            }
        }
        l.backward(&e, rows.len(), product)?;
        let grads = e.gradients.read()?;
        // Perturb real embedding entries only. This includes both inner
        // lookup paths, the backbone and the ordinary tied output gradient.
        for axis in [1_u32, 7, 23] {
            let mut rng = random::Random::new(100 + axis);
            let mut signs: Vec<_> = (0..p * d)
                .map(|i| if i % 2 == 0 { 1. } else { -1. })
                .collect();
            rng.shuffle(&mut signs);
            let adjoint: f64 = grads
                .iter()
                .zip(&signs)
                .map(|(&g, &s)| f64::from(g) * s)
                .sum();
            let delta = 0.0002;
            let mut ce = [0.; 2];
            for (j, sign) in [-1., 1.].into_iter().enumerate() {
                let mut perturbed = initial.clone();
                for (x, &v) in perturbed.iter_mut().zip(&signs) {
                    *x += (sign * delta * v) as f32;
                }
                e.parameters.write(&perturbed)?;
                forward(&mut e, &rows)?;
                l.forward(&e, rows.len(), product)?;
                ce[j] = e
                    .losses(2 * rows.len())?
                    .iter()
                    .map(|&v| f64::from(v))
                    .sum::<f64>()
                    / rows.len() as f64;
            }
            let fd = (ce[1] - ce[0]) / (2. * delta);
            println!(
                "offset={} direction={axis} max_logit_error={worst:.3e} derivative={adjoint:.6} finite_difference={fd:.6}",
                u8::from(product)
            );
            if !worst.is_finite()
                || !adjoint.is_finite()
                || !fd.is_finite()
                || worst > 2e-5
                || (adjoint - fd).abs() > 0.015 * (1. + fd.abs())
            {
                return Err("transported forward or tied adjoint failed independent check".into());
            }
        }
    }
    Ok(())
}
