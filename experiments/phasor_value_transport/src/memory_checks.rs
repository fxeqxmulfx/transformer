//! Scalar FP64 forward, end-to-end masked CE derivatives and prefix-logit checks.
use super::*;
use head::{Head, MEMORY_WIDTH as M};

pub(super) fn linear(weights: &[f32], x: &[f64]) -> Vec<f64> {
    weights
        .chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(&w, &v)| f64::from(w) * v).sum())
        .collect()
}

fn input_product_reference(
    cfg: &Config,
    weights: &[f32],
    normalized: &[f32],
    length: usize,
) -> Vec<f64> {
    let width = cfg.width;
    let start = architecture::start(cfg);
    let (input, output) = weights[start..].split_at(3 * M * width);
    let mut features = Vec::with_capacity(normalized.len());
    for sequence in normalized.chunks_exact(length * width) {
        let mut previous = [0.; M];
        for row in sequence.chunks_exact(width) {
            let row: Vec<_> = row.iter().copied().map(f64::from).collect();
            let projected = linear(input, &row);
            let product: Vec<_> = (0..M)
                .map(|j| projected[j] * (projected[M + j] + previous[j]))
                .collect();
            features.extend(
                row.into_iter()
                    .zip(linear(output, &product))
                    .map(|(x, y)| x + y),
            );
            previous.copy_from_slice(&projected[2 * M..]);
        }
    }
    features
}

fn reference(cfg: &Config, weights: &[f32], hidden: &[f32], length: usize) -> Vec<f64> {
    let d = cfg.width;
    let start = crate::architecture::start(cfg);
    let (input, output) = weights[start..start + cfg.head.parameters(cfg)].split_at(3 * M * d);
    let mut logits = Vec::new();
    for sequence in hidden.chunks_exact(length * d) {
        let features: Vec<_> = sequence
            .chunks_exact(d)
            .map(|h| h.iter().map(|&x| f64::from(x)).collect::<Vec<_>>())
            .collect();
        let projected: Vec<_> = features.iter().map(|h| linear(input, h)).collect();
        let mut state = [0.; M];
        for (t, h) in features.iter().enumerate() {
            let z = &projected[t];
            let mut read = [0.; M];
            if cfg.head == Head::Transition {
                // Only edges with both endpoints strictly before the query.
                let edges: Vec<_> = projected[..t]
                    .windows(2)
                    .map(|edge| {
                        let dot: f64 = edge[0][M..2 * M]
                            .iter()
                            .zip(&z[..M])
                            .map(|(k, q)| k * q)
                            .sum();
                        (dot / (M as f64).sqrt(), &edge[1][2 * M..])
                    })
                    .collect();
                let peak = edges.iter().map(|e| e.0).fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = edges.iter().map(|e| (e.0 - peak).exp()).sum();
                for (score, value) in edges {
                    let p = (score - peak).exp() / sum;
                    for (y, v) in read.iter_mut().zip(value) {
                        *y += p * v;
                    }
                }
            } else {
                for j in 0..M {
                    let proposal = z[M + j].tanh();
                    if cfg.head == Head::Bounded {
                        let retention = 1. / (1. + (-z[j]).exp());
                        state[j] = retention * state[j] + (1. - retention) * proposal;
                    } else {
                        state[j] = z[j].tanh() * state[j] + proposal;
                    }
                    read[j] = state[j] / (1. + (-z[2 * M + j]).exp());
                }
            }
            let merged: Vec<_> = linear(output, &read)
                .iter()
                .zip(h)
                .map(|(a, b)| a + b)
                .collect();
            logits.extend(linear(&weights[..cfg.vocab * d], &merged));
        }
    }
    logits
}

fn logit_difference(left: &[f32], right: &[f32]) -> Result<f32> {
    let mut worst = 0_f32;
    for (a, b) in left.iter().zip(right) {
        let difference = (a - b).abs();
        if !difference.is_finite() {
            return Err("nonfinite prefix logit".into());
        }
        worst = worst.max(difference);
    }
    Ok(worst)
}

/// Replay a real sequence with every nonempty proper suffix changed. Keep
/// the batch shape fixed so the two logits use identical GEMM kernels.
pub(crate) fn trained_prefixes(engine: &mut Engine, seq: &[u32]) -> Result<Value> {
    let length = seq.len();
    let vocab = engine.config.vocab;
    let masked = vec![u32::MAX; length];
    engine.forward(seq, &masked, 1, length)?;
    let baseline = engine.logits.read()?;
    let mut worst = 0_f32;
    for cut in 1..length {
        let mut changed = seq.to_vec();
        for token in &mut changed[cut..] {
            *token = (*token + 1) % vocab as u32;
        }
        engine.forward(&changed, &masked, 1, length)?;
        let alternative = engine.logits.read()?;
        worst = worst.max(logit_difference(
            &baseline[..cut * vocab],
            &alternative[..cut * vocab],
        )?);
    }
    if worst > 2e-5 {
        return Err(format!("trained prefix logits changed by {worst}").into());
    }
    Ok(json!({"sequence_length":length, "cuts":length-1, "max_logit_change":worst}))
}

pub(super) fn prefix_error(engine: &mut Engine, seq: &[u32]) -> Result<f32> {
    let p = engine.config.vocab;
    let length = seq.len();
    let masked = vec![u32::MAX; 2 * length];
    let mut worst = 0_f32;
    for cut in 1..=length {
        let mut changed = seq.to_vec();
        for t in &mut changed[cut..] {
            *t = (*t + 7) % p as u32;
        }
        engine.forward(&[seq, &changed].concat(), &masked, 2, length)?;
        let both = engine.logits.read()?;
        // A different batch size and shorter sequence also exercise the reset
        // between calls, with no right padding or previous example available.
        engine.forward(&seq[..cut], &masked[..cut], 1, cut)?;
        let alone = engine.logits.read()?;
        for offset in [0, length * p] {
            worst = worst.max(logit_difference(
                &both[offset..offset + cut * p],
                &alone[..cut * p],
            )?);
        }
    }
    Ok(worst)
}

fn gradients(
    e: &mut Engine,
    theta: &[f32],
    tokens: &[u32],
    labels: &[u32],
    length: usize,
) -> Result<()> {
    let start = crate::architecture::start(&e.config);
    let width = e.config.width;
    let block = width * M;
    let mut groups = vec![
        ("tied embedding", 0..width * e.config.vocab),
        (
            "backbone router",
            layout::router_offset(&e.config, 0)..layout::router_offset(&e.config, 0) + 96 * width,
        ),
    ];
    for (i, label) in ["projection 1", "projection 2", "projection 3", "output"]
        .into_iter()
        .enumerate()
    {
        groups.push((label, start + i * block..start + (i + 1) * block));
    }
    check_directions(
        e,
        theta,
        tokens,
        labels,
        length,
        groups
            .into_iter()
            .map(|(name, range)| (name, range.collect()))
            .collect(),
    )
}

pub(super) fn check_directions(
    e: &mut Engine,
    theta: &[f32],
    tokens: &[u32],
    labels: &[u32],
    length: usize,
    groups: Vec<(&str, Vec<usize>)>,
) -> Result<()> {
    let batch = tokens.len() / length;
    e.forward(tokens, labels, batch, length)?;
    e.backward(batch, length)?;
    let gradients = e.gradients.read()?;
    let phase = e.config.head == Head::PhasePrefix;
    let count = labels.iter().filter(|&&y| y != u32::MAX).count();
    let mut rng = random::Random::new(142);
    for (name, indices) in groups {
        let direction: Vec<_> = indices
            .iter()
            .map(|_| if rng.random() < 0.5 { -1_f32 } else { 1. })
            .collect();
        if !indices.iter().any(|&i| gradients[i].abs() > 1e-7) {
            return Err(format!("inactive {name} gradient").into());
        }
        let analytic: f64 = indices
            .iter()
            .zip(&direction)
            .map(|(&i, &s)| f64::from(gradients[i]) * f64::from(s))
            .sum();
        let mut difference = |epsilon: f32| -> Result<f64> {
            let mut losses = [0.; 2];
            for (loss, sign) in losses.iter_mut().zip([-1., 1.]) {
                let mut perturbed = theta.to_vec();
                for (&i, s) in indices.iter().zip(&direction) {
                    perturbed[i] += sign * epsilon * s;
                }
                e.parameters.write(&perturbed)?;
                e.forward(tokens, labels, batch, length)?;
                *loss = e
                    .losses(tokens.len())?
                    .iter()
                    .map(|&v| f64::from(v))
                    .sum::<f64>()
                    / count as f64;
            }
            Ok((losses[1] - losses[0]) / (2. * f64::from(epsilon)))
        };
        let coarse = difference(2e-4)?;
        let numeric = if phase {
            let fine = difference(1e-4)?;
            let finer = difference(5e-5)?;
            let first = (4. * fine - coarse) / 3.;
            let second = (4. * finer - fine) / 3.;
            println!(
                "phase difference {name}: h={coarse:.6} h/2={fine:.6} h/4={finer:.6} extrapolated={first:.6}/{second:.6}"
            );
            if (first - second).abs() > 0.004 * (1. + analytic.abs()) {
                return Err(format!("phase finite differences did not converge for {name}").into());
            }
            first
        } else {
            coarse
        };
        println!(
            "{:?} {name}: analytic={analytic:.6} numeric={numeric:.6}",
            e.config.head
        );
        if !numeric.is_finite()
            || !analytic.is_finite()
            || (numeric - analytic).abs() > 0.004 * (1. + analytic.abs())
        {
            return Err(format!("masked CE derivative mismatch for {name}").into());
        }
    }
    e.parameters.write(theta)?;
    Ok(())
}

pub fn run() -> Result<()> {
    if !precision::FP32_AFFINE_REFERENCE {
        return Err("memory-check requires --features fp32-affine-reference".into());
    }
    let tokens = [1, 4, 2, 4, 3, 7, 8, 1, 4, 2, 5, 4];
    let length = tokens.len() / 2;
    let labels = [
        u32::MAX,
        5,
        u32::MAX,
        7,
        4,
        3,
        u32::MAX,
        u32::MAX,
        1,
        4,
        u32::MAX,
        6,
    ];
    for width in [64, 128] {
        let mut cfg = config(2)?;
        cfg.max_length = length;
        cfg.vocab = 17;
        cfg.width = width;
        cfg.gain = cfg.heads as f32 / width as f32;
        let mut base = Engine::new(cfg.clone(), &layout::initialize_model(&cfg, 1))?;
        base.forward(&tokens, &labels, 2, length)?;
        let diagnostic = base.diagnostic_snapshot(tokens.len())?;
        for head in [
            Head::Transition,
            Head::Recurrent,
            Head::Bounded,
            Head::InputProduct,
        ] {
            cfg.head = head;
            let mut weights = layout::initialize_model(&cfg, 1);
            let mut engine = Engine::new(cfg.clone(), &weights)?;
            if matches!(head, Head::Bounded | Head::InputProduct) {
                engine.forward(&tokens, &labels, 2, length)?;
                if engine.logits.read()? != base.logits.read()? {
                    return Err("zero residual changed the backbone logits".into());
                }
                engine.backward(2, length)?;
                let grad = engine.gradients.read()?;
                let start = architecture::start(&cfg);
                let output = start + 3 * M * width;
                let end = output + M * width;
                if grad[start..output].iter().any(|&x| x != 0.)
                    || !grad[output..end].iter().any(|g| g.abs() > 1e-7)
                {
                    return Err("incorrect gradient through a zero output map".into());
                }
                let mut rng = random::Random::new(3407);
                for w in &mut weights[output..end] {
                    *w = rng.gauss(0., 0.12) as f32;
                }
                engine.parameters.write(&weights)?;
            }
            engine.forward(&tokens, &labels, 2, length)?;
            let (quantity, actual, expected) = if head == Head::InputProduct {
                (
                    "input feature",
                    engine.input_product_snapshot(tokens.len())?,
                    input_product_reference(
                        &cfg,
                        &weights,
                        &diagnostic["embedding_normalized"],
                        length,
                    ),
                )
            } else {
                (
                    "logit",
                    engine.logits.read()?,
                    reference(&cfg, &weights, &diagnostic["readout_input"], length),
                )
            };
            if actual.len() != expected.len() {
                return Err("FP64 reference and GPU output shapes differ".into());
            }
            let mut worst = 0_f64;
            for (&a, b) in actual.iter().zip(expected) {
                let delta = (f64::from(a) - b).abs();
                if !delta.is_finite() {
                    return Err("nonfinite forward value".into());
                }
                worst = worst.max(delta);
            }
            let prefix = prefix_error(&mut engine, &tokens[..length])?;
            println!(
                "{head:?} width={width} max FP64 {quantity} error={worst:.3e} max prefix logit change={prefix:.3e}"
            );
            if worst > 2e-5 || prefix > 2e-5 {
                return Err("hidden forward or causality check failed".into());
            }
            gradients(&mut engine, &weights, &tokens, &labels, length)?;
            if head == Head::Transition {
                engine.forward(&tokens[..4], &labels[..4], 2, 2)?;
                engine.backward(2, 2)?;
                let start = crate::architecture::start(&cfg);
                if engine.gradients.read()?[start..start + head.parameters(&cfg)]
                    .iter()
                    .any(|&g| g != 0.)
                {
                    return Err("two-token example wrote a transition gradient".into());
                }
            }
        }
    }
    Ok(())
}
