//! Compare the physical compact forward and pullback to a full Phasor
//! with exactly the same active complex coordinates and zero padded units.
use super::*;

pub(super) fn close(name: &str, a: &[f32], b: &[f32], tolerance: f32) -> Result<f32> {
    if a.len() != b.len() || a.is_empty() {
        return Err(format!("shape mismatch for {name}").into());
    }
    let mut worst = 0_f32;
    for (&x, &y) in a.iter().zip(b) {
        let delta = (x - y).abs();
        if !delta.is_finite() || delta > tolerance * (1. + x.abs().max(y.abs())) {
            return Err(format!("{name} mismatch {x} vs {y}").into());
        }
        worst = worst.max(delta);
    }
    Ok(worst)
}

pub fn run() -> Result<()> {
    let (batch, length) = (2, 5);
    let rows = batch * length;
    for width in [64, 128, 256] {
        let mut cfg = config(batch)?;
        cfg.width = width;
        cfg.heads = width / 32;
        cfg.layers = 2;
        cfg.gain = cfg.heads as f32 / width as f32;
        cfg.vocab = 37;
        cfg.max_length = length;
        cfg.head = head::Head::CompactStream;
        let wide = compact::wide_config(&cfg);
        let mut parameters = layout::initialize_model(&cfg, 11);
        let unpruned = layout::initialize_model(&wide, 11);
        let removed = 3 * width * (compact::hidden(&wide) - compact::hidden(&cfg)) * cfg.layers;
        if unpruned.len() != parameters.len() + removed
            || layout::independent_count(&wide) != layout::independent_count(&cfg) + removed / 2
            || parameters != compact::restrict(&cfg, &unpruned)
        {
            return Err("compact shape, tied-coordinate count or seeded projections differ".into());
        }
        let mut rng = random::Random::new(215);
        for x in &mut parameters[architecture::start(&cfg)..] {
            *x = rng.gauss(0., 0.15) as f32;
        }
        let padded = compact::pad(&cfg, &parameters);
        if compact::restrict(&cfg, &padded) != parameters {
            return Err("compact weights did not round-trip through the wide layout".into());
        }
        layout::validate(&cfg, &parameters)?;
        layout::validate(&wide, &padded)?;
        let mut small_engine = Engine::new(cfg.clone(), &parameters)?;
        let mut wide_engine = Engine::new(wide, &padded)?;
        let vectors: Vec<_> = (0..rows * width)
            .map(|_| rng.gauss(0., 0.32) as f32)
            .collect();
        let small = small_engine.infer_embeddings(&vectors, batch, length)?;
        let large = wide_engine.infer_embeddings(&vectors, batch, length)?;
        let features = close("padded hidden states", &small.hidden, &large.hidden, 2e-5)?;
        let logits = close("padded logits", &small.logits, &large.logits, 2e-5)?;
        let g: Vec<_> = (0..rows * cfg.vocab)
            .map(|_| rng.gauss(0., 0.11) as f32)
            .collect();
        let dx = small_engine.embedding_pullback(&vectors, batch, length, &g)?;
        let dy = wide_engine.embedding_pullback(&vectors, batch, length, &g)?;
        let input = close("padded embedding gradient", &dx, &dy, 1e-4)?;
        // This second check includes the tied input lookup and the actual masked CE.
        let tokens = [3, 19, 17, 3, 11, 1, 9, 11, 9, 3];
        let labels = [u32::MAX, 4, 8, u32::MAX, 2, 3, u32::MAX, 1, 19, 3];
        for engine in [&mut small_engine, &mut wide_engine] {
            engine.forward(&tokens, &labels, batch, length)?;
            engine.backward(batch, length)?;
        }
        let sg = small_engine.gradients.read()?;
        let wg = wide_engine.gradients.read()?;
        let weight = close(
            "all padded parameter gradients",
            &compact::pad(&cfg, &sg),
            &wg,
            2e-4,
        )?;
        println!(
            "compact width={width} layers={} removed_stored={removed} removed_independent={} max_abs_hidden={features:.3e} logits={logits:.3e} input_grad={input:.3e} weight_grad={weight:.3e}",
            cfg.layers,
            removed / 2
        );
    }
    Ok(())
}
