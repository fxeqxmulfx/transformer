//! Independent checks for public forward, backward and update entry points.
use super::*;

fn mean_loss(e: &Engine, rows: usize, targets: usize) -> Result<f64> {
    Ok(e.losses(rows)?.iter().map(|&x| f64::from(x)).sum::<f64>() / targets as f64)
}

// Independent transcription of PairKey.address, using the reference field
// exponentiation and reading the stored tables directly in double precision.
fn residual(a: u32, b: u32, p: u32, d: usize, tables: &[f32]) -> Vec<f64> {
    let power = |x, i| arithmetic_data::power(x, i, p);
    let inverse = if b == 0 { 0 } else { power(b, p - 2) };
    let mut keys = vec![b, (a + p - b) % p, a.abs_diff(b), a * inverse % p];
    keys.extend((1..=3).flat_map(|i| (1..=3).map(move |j| (power(a, i) + power(b, j)) % p)));
    keys.extend((1..=2).flat_map(|i| (1..=2).map(move |j| power(a, i) * power(b, j) % p)));
    let mut result = vec![0.; d];
    for (table, key) in keys.into_iter().enumerate() {
        let offset = (table * p as usize + key as usize) * d;
        for (x, &v) in result.iter_mut().zip(&tables[offset..offset + d]) {
            *x += f64::from(v);
        }
    }
    result
}

pub fn run() -> Result<()> {
    if !precision::FP32_AFFINE_REFERENCE {
        return Err("directional finite differences require the FP32 affine reference".into());
    }
    // Both positions carry losses. Repeated keys and distinct zero-row targets
    // exercise the direct first-token gradient and the transpose of the table read.
    let tokens = [0, 4, 0, 7, 96, 2, 96, 2, 13, 0];
    let labels = [2, 4, 6, 7, u32::MAX, 3, 5, 7, 8, 9];
    let (batch, targets) = (tokens.len() / 2, labels.len() - 1);
    let mut cfg = config(batch + 1)?;
    let mut base = Engine::new(cfg.clone(), &layout::initialize_model(&cfg, 0))?;
    base.forward(&tokens, &labels, batch, 2)?;
    let scores = base.logits.read()?;
    let old = ValueLayer::new(&base)?;
    old.forward(&base, batch, true)?;
    let transport = base.logits.read()?;
    cfg.pair_values = true;
    let (d, p, start) = (cfg.width, cfg.vocab, pair_memory::start(&cfg));
    let mut theta = layout::initialize_model(&cfg, 0);
    let mut engine = Engine::new(cfg, &theta)?;
    engine.forward(&tokens, &labels, batch, 2)?;
    let actual = engine.logits.read()?;
    for i in 0..2 * batch {
        let range = i * p..(i + 1) * p;
        let expected = if i % 2 == 0 { &scores } else { &transport };
        if actual[range.clone()] != expected[range] {
            return Err("zero memory failed exact first/second position comparisons".into());
        }
    }
    // Realize the Lean witness through the production GPU entry point at width 128.
    let mut witness = theta.clone();
    witness[..p * d].fill(0.);
    for b in 0..p {
        witness[b * d + b] = 1.;
        witness[start + b * d + b] = 2.;
    }
    engine.parameters.write(&witness)?;
    for lo in (0..p).step_by(batch) {
        let words: Vec<_> = (lo..p.min(lo + batch))
            .flat_map(|b| [0, b as u32])
            .collect();
        engine.forward(&words, &vec![u32::MAX; words.len()], words.len() / 2, 2)?;
        let predictions = engine.predictions.read()?;
        if words
            .as_chunks::<2>()
            .0
            .iter()
            .zip(predictions.as_chunks::<2>().0.iter())
            .any(|(t, y)| t[1] != y[1])
        {
            return Err("zero-row witness failed".into());
        }
    }
    println!("zero-initialized readout bit-identical at both positions; zero-row witness {p}/{p}");

    let n = pair_memory::parameters(&engine.config);
    let mut rng = random::Random::new(301);
    for x in &mut theta[start..start + n] {
        *x = rng.gauss(0., 0.005) as f32;
    }
    engine.parameters.write(&theta)?;
    engine.forward(&tokens, &labels, batch, 2)?;
    let fitted = engine.logits.read()?;
    let mut worst = 0_f64;
    for (row, ab) in tokens.as_chunks::<2>().0.iter().enumerate() {
        let at = 2 * row * p;
        let v = residual(ab[0], ab[1], p as u32, d, &theta[start..start + n]);
        let scalar = checks::reference(
            &scores[at..at + p],
            &theta[..p * d],
            &v,
            (ab[0] * ab[1]) as usize % p,
        );
        for (&actual, expected) in fitted[at + p..at + 2 * p].iter().zip(scalar) {
            let error = (f64::from(actual) - expected).abs();
            if !error.is_finite() {
                return Err("nonfinite reference difference".into());
            }
            worst = worst.max(error);
        }
    }
    println!("nonzero pair values versus FP64 forward, maximum logit error {worst:.3e}");
    if worst > 2e-5 {
        return Err("pair memory forward differs from reference".into());
    }
    engine.backward(batch, 2)?;
    let grad = engine.gradients.read()?;
    // Check each memory family. Scale each perturbation relative to the nonzero
    // hidden state; long sums over unused entries then contribute exactly zero.
    let mut directions = vec![("tied embedding".to_owned(), 0..p * d)];
    for k in 0..pair_memory::TABLES {
        directions.push((
            format!("value table {k}"),
            start + k * p * d..start + (k + 1) * p * d,
        ));
    }
    for (label, range) in directions {
        let signs: Vec<f32> = range
            .clone()
            .map(|_| if rng.random() < 0.5 { -1. } else { 1. })
            .collect();
        let derivative: f64 = grad[range.clone()]
            .iter()
            .zip(&signs)
            .map(|(&g, &s)| f64::from(g) * f64::from(s))
            .sum();
        let delta = 0.0002;
        let mut losses = [0.; 2];
        for (i, sign) in [-1., 1.].into_iter().enumerate() {
            let mut v = theta.clone();
            for (x, &s) in v[range.clone()].iter_mut().zip(&signs) {
                *x += sign * delta * s;
            }
            engine.parameters.write(&v)?;
            engine.forward(&tokens, &labels, batch, 2)?;
            losses[i] = mean_loss(&engine, tokens.len(), targets)?;
        }
        let numerical = (losses[1] - losses[0]) / (2. * f64::from(delta));
        println!("{label}: adjoint={derivative:.6} central_difference={numerical:.6}");
        if !derivative.is_finite()
            || !numerical.is_finite()
            || (derivative - numerical).abs() > 0.01 * (1. + numerical.abs())
        {
            return Err(format!("{label} derivative mismatch").into());
        }
    }
    // A one-token call in this same model uses the ordinary backbone. A previous
    // two-token batch must not leave a stale gradient in any memory coordinate.
    engine.parameters.write(&theta)?;
    engine.forward(&[0, 2], &[4, 5], 2, 1)?;
    engine.backward(2, 1)?;
    if engine.gradients.read()?[start..start + n]
        .iter()
        .any(|&v| v != 0.)
    {
        return Err("one-token prefix retained table gradients".into());
    }
    println!("one-token gradient reset {n}/{n}");
    Ok(())
}
