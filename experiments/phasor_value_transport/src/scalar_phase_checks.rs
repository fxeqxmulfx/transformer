//! Scalar bottleneck reference, collision, and rank-one pullback on dense inputs.
use super::*;
use head::MEMORY_WIDTH as M;

fn check_rank_and_collisions(
    e: &mut Engine,
    theta: &[f32],
    input: usize,
    context: usize,
) -> Result<()> {
    let (width, vocab, length) = (e.config.width, e.config.vocab, 5);
    let mut rng = random::Random::new(187);
    let mut sample = || rng.gauss(0., 0.28) as f32;
    let vectors: Vec<_> = (0..length * width).map(|_| sample()).collect();
    let mut gradients = Vec::new();
    for _ in 0..2 {
        let mut upstream = vec![0.; length * vocab];
        for g in &mut upstream[(length - 1) * vocab..] {
            *g = sample();
        }
        gradients.push(e.embedding_pullback(&vectors, 1, length, &upstream)?);
    }
    let dot = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>()
    };
    let (u, v) = (&gradients[0], &gradients[1]);
    let (uu, vv) = (dot(u, u), dot(v, v));
    if uu < 1e-10 || vv < 1e-10 {
        return Err("inactive scalar pullback fixture".into());
    }
    let scale = dot(u, v) / vv;
    let residual = (u
        .iter()
        .zip(v)
        .map(|(&x, &y)| (f64::from(x) - scale * f64::from(y)).powi(2))
        .sum::<f64>()
        / uu)
        .sqrt();
    if !residual.is_finite() || residual > 2e-4 {
        return Err(
            format!("scalar pullbacks span multiple directions: residual {residual}").into(),
        );
    }

    let mut sums = theta.to_vec();
    sums[input + 2 * width..input + 4 * width].fill(0.);
    sums[context..context + width].fill(0.);
    e.parameters.write(&sums)?;
    let last = (length - 1) * width;
    let result = e.infer_embeddings(&vectors, 1, length)?;
    let mut swapped = vectors.clone();
    let (first, rest) = swapped.split_at_mut(width);
    first.swap_with_slice(&mut rest[..width]);
    let collision = e.infer_embeddings(&swapped, 1, length)?;
    if result.hidden[last..] != collision.hidden[last..]
        || result.hidden[..width] == collision.hidden[..width]
    {
        return Err("distinct additive prefixes failed their scalar collision check".into());
    }
    sums[input..input + 2 * width].fill(0.);
    e.parameters.write(&sums)?;
    let zero = e.infer_embeddings(&vectors, 1, length)?;
    let derivative = e.embedding_pullback(&vectors, 1, length, &vec![1.; length * vocab])?;
    if zero
        .hidden
        .iter()
        .chain(&zero.logits)
        .chain(&derivative)
        .any(|&v| v != 0.)
    {
        return Err("a zero scalar left a path from the embedding to the classifier".into());
    }
    e.parameters.write(theta)?;
    println!(
        "scalar bottleneck width={width} pullback_rank_one_residual={residual:.3e} additive_collision=true zero_pullback=true"
    );
    Ok(())
}

fn network(width: usize) -> Result<()> {
    let mut cfg = config(2)?;
    cfg.head = architecture::HEAD;
    cfg.width = width;
    cfg.heads = width / 32;
    cfg.gain = cfg.heads as f32 / width as f32;
    cfg.vocab = 19;
    cfg.max_length = 5;
    let first = architecture::start(&cfg);
    let input = architecture::memory_start(&cfg);
    let output = input + 4 * width;
    let frequency = output + 2 * M * width;
    let context = frequency + M;
    let end = layout::frequency_offset(&cfg);
    if context + width != end {
        return Err("scalar bottleneck layout mismatch".into());
    }
    let weights = layout::initialize_model(&cfg, 54);
    let mut e = Engine::new(cfg.clone(), &weights)?;
    let tokens = [1, 3, 5, 7, 9, 9, 8, 6, 4, 2];
    let labels = [u32::MAX, 6, 2, 4, 8, 7, u32::MAX, 5, 3, 1];
    e.forward(&tokens, &labels, 2, 5)?;
    let diagnostic = e.diagnostic_snapshot(tokens.len())?;
    let context_rows = diagnostic["readout_input"].chunks_exact(width * 5);
    let mut expected = Vec::new();
    for (xs, hs) in diagnostic["embedding_normalized"]
        .chunks_exact(width * 5)
        .zip(context_rows)
    {
        let (mut left, mut state) = (0., 0.);
        for (x, h) in xs.chunks_exact(width).zip(hs.chunks_exact(width)) {
            let x: Vec<_> = x.iter().copied().map(f64::from).collect();
            let delta = memory_checks::linear(&weights[input..output], &x);
            state += delta[0] + left * delta[2];
            left += delta[1];
            let z = state
                + delta[3]
                + weights[context..end]
                    .iter()
                    .zip(h)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum::<f64>();
            let angles: Vec<_> = weights[frequency..context]
                .iter()
                .map(|&v| z * f64::from(v))
                .collect();
            let read: Vec<_> = angles
                .iter()
                .map(|v| v.sin())
                .chain(angles.iter().map(|v| v.cos() - 1.))
                .collect();
            let hidden = memory_checks::linear(&weights[output..frequency], &read);
            expected.extend(memory_checks::linear(
                &weights[..cfg.vocab * width],
                &hidden,
            ));
        }
    }
    let forward = phase_checks::error(&e.logits.read()?, &expected, 2e-5)?;
    let causal = memory_checks::prefix_error(&mut e, &tokens[..5])?;
    if causal > 2e-5 {
        return Err("future embedding changed a scalar prefix".into());
    }
    println!("scalar bottleneck width={width} max_logit_error={forward:.3e} prefix={causal:.3e}");
    check_rank_and_collisions(&mut e, &weights, input, context)?;
    let router = layout::router_offset(&cfg, 0);
    let mut groups = vec![
        ("tied embedding", (0..cfg.vocab * width).collect()),
        ("backbone router", (router..router + 96 * width).collect()),
        ("stream read", (first..input).collect()),
        ("decoded feature", (output..frequency).collect()),
        ("frequencies", (frequency..context).collect()),
        ("context coordinate", (context..end).collect()),
    ];
    for (i, name) in ["additive", "left", "right", "local"]
        .into_iter()
        .enumerate()
    {
        groups.push((name, (input + i * width..input + (i + 1) * width).collect()));
    }
    memory_checks::check_directions(&mut e, &weights, &tokens, &labels, 5, groups)
}

pub fn run() -> Result<()> {
    if !precision::FP32_AFFINE_REFERENCE {
        return Err("scalar checks require FP32 reference affines".into());
    }
    for length in [1, 7, 257] {
        bilinear_phase_checks::scan::<true>(length, 1)?;
    }
    for width in [64, 128] {
        network(width)?;
    }
    Ok(())
}
