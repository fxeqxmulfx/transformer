//! Replay trained weights and evaluate the sufficient margin in TransportedValues.lean.
use super::*;

pub fn save(e: &Engine, path: &Path) -> Result<()> {
    let weights = e.parameters.read()?;
    layout::validate(&e.config, &weights)?;
    let pending = path.with_extension("f32.tmp");
    fs::write(
        &pending,
        weights
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    fs::rename(pending, path)?;
    Ok(())
}

fn load(path: &Path, cfg: Config) -> Result<Engine> {
    let bytes = fs::read(path)?;
    if bytes.len() != 4 * layout::total_count(&cfg) {
        return Err("wrong checkpoint length".into());
    }
    let weights: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_le_bytes)
        .collect();
    Engine::new(cfg, &weights)
}

pub fn predict(path: &Path, field: Option<usize>, a: u32, b: u32) -> Result<()> {
    let mut cfg = config(1)?;
    if let Some(prime) = field {
        cfg.vocab = prime;
        cfg.pair_values = true;
        if !pair_memory::active(&cfg) {
            return Err("pair prediction requires a prime vocabulary".into());
        }
    }
    let mut e = load(path, cfg)?;
    e.forward(&[a, b], &[u32::MAX, u32::MAX], 1, 2)?;
    let address = e.predictions.read()?[0];
    if field.is_none() {
        ValueLayer::new(&e)?.forward(&e, 1, true)?;
    }
    println!(
        "{}",
        json!({"a":a,"b":b,"address":address,"prediction":e.predictions.read()?[1]})
    );
    Ok(())
}

/// For each ordered target/rival pair, compute the good-address gap and the
/// smallest gap across all other value codes. Translation permutes the codes.
fn gram_gaps(e: &[f32], p: usize, d: usize) -> Vec<Vec<(f64, f64)>> {
    let gram: Vec<Vec<f64>> = e
        .chunks_exact(d)
        .take(p)
        .map(|u| {
            e.chunks_exact(d)
                .take(p)
                .map(|v| {
                    u.iter()
                        .zip(v)
                        .map(|(&x, &y)| f64::from(x) * f64::from(y))
                        .sum()
                })
                .collect()
        })
        .collect();
    (0..p)
        .map(|t| {
            (0..p)
                .filter(|&c| c != t)
                .map(|c| {
                    let good = gram[t][t] - gram[c][t];
                    let other = (0..p)
                        .filter(|&k| k != t)
                        .map(|k| gram[t][k] - gram[c][k])
                        .fold(f64::INFINITY, f64::min);
                    (good, other)
                })
                .collect()
        })
        .collect()
}

pub fn inspect(path: &Path) -> Result<()> {
    let data = cubic_data()?;
    let mut e = load(path, config(BATCH)?)?;
    let (p, d) = (e.config.vocab, e.config.width);
    let layer = ValueLayer::new(&e)?;
    let metrics = evaluate(&mut e, Some((&layer, true)), &data.validation)?;
    let gaps = gram_gaps(&e.parameters.read()?, p, d);
    // Position zero only depends on a; at b=0 the stored value index is s.
    let tokens: Vec<_> = (0..p as u32).flat_map(|a| [a, 0]).collect();
    e.forward(&tokens, &vec![u32::MAX; 2 * p], p, 2)?;
    let scores = e.logits.read()?;
    let mut mass = vec![0.; p];
    for (a, m) in mass.iter_mut().enumerate() {
        let s = &scores[2 * a * p..(2 * a + 1) * p];
        let max = f64::from(s.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let z: f64 = s.iter().map(|&x| (f64::from(x) - max).exp()).sum();
        *m = (f64::from(s[a * a * a % p]) - max).exp() / z;
    }
    let mut positive = 0;
    let mut zero_products = 0;
    let mut worst = f64::INFINITY;
    for &[a, b, target] in &data.validation {
        zero_products += usize::from((a as usize * b as usize).is_multiple_of(p));
        let m = mass[a as usize];
        let bound = gaps[target as usize]
            .iter()
            .map(|&(good, other)| m * good + (1. - m) * other)
            .fold(f64::INFINITY, f64::min);
        positive += usize::from(bound > 0.);
        worst = worst.min(bound);
    }
    println!(
        "{}",
        json!({"held_out":metrics,"positive_gram_bound":positive,"minimum_gram_bound":worst,
        "zero_product_pairs":zero_products,
        "minimum_cubic_address_mass":mass.into_iter().fold(1.,f64::min)})
    );
    Ok(())
}
