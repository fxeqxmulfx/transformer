//! Inspect the learned coordinates, routing and phase of a frozen NV11 checkpoint.
use super::*;
use fourier_geometry::Measurements;

fn measure(e: &mut Engine) -> Result<Measurements> {
    let (p, m) = (e.config.vocab, fourier::channels(e.config.width));
    let mut observations = Measurements {
        p,
        m,
        coordinates: Vec::with_capacity(p * m),
        phases: Vec::with_capacity(p * m),
        routes: Vec::with_capacity(p * (m + 1)),
    };
    let words: Vec<_> = (0..p as u32).collect();
    for chunk in words.chunks(e.config.max_batch) {
        let tokens: Vec<_> = chunk.iter().flat_map(|&w| [w, w]).collect();
        e.forward(&tokens, &vec![u32::MAX; tokens.len()], chunk.len(), 2)?;
        let batch = e.fourier_probe()?;
        for i in 0..chunk.len() {
            let row = 2 * i + 1;
            let base = row * 2 * m;
            observations.routes.extend(
                batch.routes[row * (m + 1)..(row + 1) * (m + 1)]
                    .iter()
                    .map(|&w| f64::from(w)),
            );
            for j in 0..m {
                observations.coordinates.push([
                    f64::from(batch.coordinates[base + j]),
                    f64::from(batch.coordinates[base + m + j]),
                ]);
                observations.phases.push([
                    f64::from(batch.phases[base + j]),
                    f64::from(batch.phases[base + m + j]),
                ]);
            }
        }
    }
    Ok(observations)
}

pub fn run(path: &Path) -> Result<()> {
    if !architecture::FOURIER {
        return Err("Fourier audit requires fourier-transport".into());
    }
    let bytes = fs::read(path)?;
    let digest = hash(&bytes);
    let recorded = snapshot::recorded_nv11(path, &digest)?
        .ok_or("expected a named Fourier training checkpoint")?;
    let mut cfg = config(BATCH)?;
    cfg.head = architecture::HEAD;
    let mut e = snapshot::load(&bytes, cfg.clone())?;
    let weights = e.parameters.read()?;
    let data = cubic_data()?;
    let names: Vec<_> = (0..cfg.vocab).collect();
    let score = |e: &mut Engine, rows: &[[u32; 3]], names: &[usize]| {
        dense_audit::score(e, rows, names, |_| Ok(()))
    };
    let splits = [data.training.rows.as_slice(), data.validation.as_slice()];
    let baseline = [
        score(&mut e, splits[0], &names)?,
        score(&mut e, splits[1], &names)?,
    ];
    let full = [baseline[0].correct, baseline[1].correct];
    if recorded["train"]["correct"] != full[0] || recorded["held_out"]["correct"] != full[1] {
        return Err("full model differs from its training observation".into());
    }
    let geometry = fourier_geometry::analyze(&measure(&mut e)?)?;
    let seed = recorded["seed"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or("invalid Fourier initialization seed")?;
    let initial = layout::initialize_model(&cfg, seed);
    let initial_bytes: Vec<_> = initial.iter().flat_map(|w| w.to_le_bytes()).collect();
    if recorded["initial_sha256"] != hash(&initial_bytes) {
        return Err("Fourier initialization differs from the training observation".into());
    }
    e.parameters.write(&initial)?;
    let initial_geometry = fourier_geometry::analyze(&measure(&mut e)?)?;
    let mut permutation = names.clone();
    random::Random::new(73).shuffle(&mut permutation);
    let d = cfg.width;
    let m = fourier::channels(d);
    let mut renamed = weights.clone();
    for (i, &j) in permutation.iter().enumerate() {
        renamed[j * d..(j + 1) * d].copy_from_slice(&weights[i * d..(i + 1) * d]);
    }
    e.parameters.write(&renamed)?;
    for (i, rows) in splits.iter().enumerate() {
        let other = score(&mut e, rows, &permutation)?;
        if other.predictions != baseline[i].predictions
            || !other
                .logits
                .iter()
                .zip(&baseline[i].logits)
                .all(|(x, y)| x.to_bits() == y.to_bits())
        {
            return Err("renaming the externally indexed vocabulary changed the Fourier prediction or a logit".into());
        }
    }
    let mut ablations = serde_json::Map::new();
    let control = architecture::memory_start(&cfg) + 2 * m * d;
    let angle = control + (m + 1) * d;
    let bias = angle + m * d;
    for (name, disable_route) in [("identity_route", true), ("zero_phase", false)] {
        let mut changed = weights.clone();
        if disable_route {
            changed[control..angle].fill(0.);
            // exp(-200) underflows in FP32. The effective route is exactly
            // the zero shift for all contexts, including the constant row.
            changed[bias..bias + m + 1].fill(-200.);
            changed[bias] = 0.;
        } else {
            changed[angle..bias].fill(0.);
            changed[bias + m + 1..bias + 2 * m + 1].fill(0.);
        }
        e.parameters.write(&changed)?;
        let probed = measure(&mut e)?;
        if disable_route {
            if !probed
                .routes
                .chunks_exact(m + 1)
                .all(|r| r[0] == 1. && r[1..].iter().all(|&x| x == 0.))
            {
                return Err("routing ablation failed to give exact identity routes".into());
            }
        } else if probed.phases.iter().any(|&v| v != [1., 0.]) {
            return Err("phase ablation failed to give exact unit phases".into());
        }
        ablations.insert(
            name.into(),
            json!([
                score(&mut e, splits[0], &names)?.correct,
                score(&mut e, splits[1], &names)?.correct
            ]),
        );
    }
    println!(
        "{}",
        json!({"model":architecture::MODEL,"step":recorded["step"],"seed":recorded["seed"],"weights_sha256":digest,
        "channels":m,"parameters":weights.len(),"readout_parameters":cfg.head.parameters(&cfg),"baseline":full,
        "vocabulary_permutation":{"identical_predictions":splits.iter().map(|s|s.len()).sum::<usize>(),"bit_identical_logits":baseline.iter().map(|s|s.logits.len()).sum::<usize>()},
        "ablations":ablations,"geometry":geometry,"initial_geometry":initial_geometry})
    );
    Ok(())
}
