//! Exercise the ID adapter and vector-only interface on the same fitted parameters.
use super::*;

pub fn run() -> Result<()> {
    let (batch, length, width, vocab) = (2, 5, 64, 37);
    let rows = batch * length;
    let mut cfg = config(batch)?;
    cfg.width = width;
    cfg.heads = 2;
    cfg.gain = cfg.heads as f32 / width as f32;
    cfg.vocab = vocab;
    cfg.max_length = length;
    cfg.head = architecture::HEAD;
    let mut parameters = layout::initialize_model(&cfg, 17);
    let start = architecture::start(&cfg);
    let count = architecture::parameters(&cfg);
    let mut rng = random::Random::new(2801);
    for x in &mut parameters[start..start + count] {
        *x = rng.gauss(0., 0.17) as f32;
    }
    let mut e = Engine::new(cfg, &parameters)?;
    let tokens = [4, 16, 4, 11, 22, 29, 7, 9, 14, 18];
    let embedded: Vec<_> = tokens
        .iter()
        .flat_map(|&t| {
            let i = t as usize * width;
            parameters[i..i + width].iter().copied()
        })
        .collect();
    e.forward(&tokens, &[5; 10], batch, length)?;
    let expected = e.logits.read()?;
    let previous_loss = e.losses(rows)?;
    e.forward(&tokens, &[11; 10], batch, length)?;
    if e.logits.read()? != expected || e.losses(rows)? == previous_loss {
        return Err(
            "changing only the targets altered inference or failed to change the loss".into(),
        );
    }
    let looked_up = e.infer_embeddings(&embedded, batch, length)?;
    if looked_up.logits != expected || e.backward(batch, length).is_ok() {
        return Err(
            "vector input differs from lookup or reused a stale supervised gradient".into(),
        );
    }
    let vectors: Vec<_> = (0..rows * width)
        .map(|_| rng.gauss(0., 0.27) as f32)
        .collect();
    if vectors.chunks_exact(width).any(|v| {
        parameters[..vocab * width]
            .chunks_exact(width)
            .any(|word| word == v)
    }) {
        return Err("arbitrary-vector fixture accidentally hit an embedding row".into());
    }
    let reference = e.infer_embeddings(&vectors, batch, length)?;
    let cotangent: Vec<_> = (0..rows * vocab)
        .map(|_| rng.gauss(0., 0.12) as f32)
        .collect();
    let gradient = e.embedding_pullback(&vectors, batch, length, &cotangent)?;
    for trial in 0..3 {
        let direction: Vec<_> = (0..vectors.len()).map(|_| rng.gauss(0., 0.25)).collect();
        let analytic: f64 = gradient
            .iter()
            .zip(&direction)
            .map(|(&x, &v)| f64::from(x) * v)
            .sum();
        let epsilon = 0.001;
        let mut value = [0.; 2];
        for (j, sign) in [-1., 1.].into_iter().enumerate() {
            let changed: Vec<_> = vectors
                .iter()
                .zip(&direction)
                .map(|(&x, &v)| x + (sign * epsilon * v) as f32)
                .collect();
            value[j] = e
                .infer_embeddings(&changed, batch, length)?
                .logits
                .iter()
                .zip(&cotangent)
                .map(|(&l, &g)| f64::from(l) * f64::from(g))
                .sum();
        }
        let numeric = (value[1] - value[0]) / (2. * epsilon);
        if !analytic.is_finite()
            || !numeric.is_finite()
            || (analytic - numeric).abs() > 0.003 * (1. + numeric.abs())
        {
            return Err(
                format!("input-vector derivative {analytic} differs from {numeric}").into(),
            );
        }
        println!("embedding input direction={trial} analytic={analytic:.7} numeric={numeric:.7}");
    }
    e.forward(&[23; 10], &[6; 10], batch, length)?;
    let replay = e.infer_embeddings(&vectors, batch, length)?;
    if reference.hidden != replay.hidden
        || reference.logits != replay.logits
        || reference.hidden == looked_up.hidden
    {
        return Err("a prior labeled batch influenced the embedding backbone".into());
    }
    let prefix = 3;
    let mut changed = vectors.clone();
    for x in &mut changed[prefix * width..] {
        *x = rng.gauss(0., 0.51) as f32;
    }
    let extended = e.infer_embeddings(&changed, batch, length)?;
    let isolated = e.infer_embeddings(&vectors[..prefix * width], 1, prefix)?;
    if reference.hidden[..prefix * width] != extended.hidden[..prefix * width]
        || reference.logits[..prefix * vocab] != extended.logits[..prefix * vocab]
        || reference.hidden[..prefix * width] != isolated.hidden
        || reference.logits[..prefix * vocab] != isolated.logits
        || reference.hidden[prefix * width..] == extended.hidden[prefix * width..]
    {
        return Err("future embedding vectors or another batch item influenced a prefix".into());
    }
    parameters.copy_within(4 * width..5 * width, 16 * width);
    e.parameters.write(&parameters)?;
    let aliases: Vec<_> = tokens
        .iter()
        .map(|&v| if v == 16 { 4 } else { v })
        .collect();
    e.forward(&tokens, &[u32::MAX; 10], batch, length)?;
    let aliased = e.logits.read()?;
    e.forward(&aliases, &[u32::MAX; 10], batch, length)?;
    if aliased != e.logits.read()? {
        return Err("equal embedding vectors leaked their token IDs".into());
    }
    if e.infer_embeddings(&vectors[..width - 1], 1, 1).is_ok()
        || e.infer_embeddings(&vec![f32::NAN; width], 1, 1).is_ok()
        || e.infer_embeddings(&[], 1, 0).is_ok()
    {
        return Err("malformed embedding vectors were accepted".into());
    }
    println!(
        "embedding interface positions={rows} width={width} classes={vocab} lookup_logits={} label_invariant_logits={} arbitrary_hidden={} causal_prefix={} alias_logits={}",
        expected.len(),
        expected.len(),
        reference.hidden.len(),
        prefix,
        aliased.len()
    );
    Ok(())
}
