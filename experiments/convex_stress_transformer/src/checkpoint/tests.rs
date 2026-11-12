use super::*;
use crate::{adam, data::Example, layout::Dims, reference};
use std::time::{SystemTime, UNIX_EPOCH};
fn directory() -> Result<std::path::PathBuf> {
    let path = std::env::temp_dir().join(format!(
        "transformer-state-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&path)?;
    Ok(path)
}
fn model(layers: usize) -> Transformer {
    let dims = Dims {
        vocab: 8,
        d: 8,
        heads: 4,
        layers,
        dff: 32,
        ctx: 8,
    };
    Transformer::new(dims, 7, 2, adam::Hyper::source(0.1)).unwrap()
}
fn examples() -> Vec<Example> {
    (0..5)
        .map(|k| {
            let n = 3 + k;
            let mut targets = vec![-100; n];
            targets[0] = (k % 8) as i64;
            targets[n - 1] = ((k + 2) % 8) as i64;
            Example {
                task: "fixture".into(),
                tokens: (0..n).map(|i| ((i + k) % 8) as i64).collect(),
                targets,
                changes: vec![false; n],
                prompt: vec![],
                answer: vec![],
                limit: 0,
                field: None,
                row_index: None,
            }
        })
        .collect()
}
fn batch<'a>(examples: &'a [Example], rng: &mut Mt, mini: bool) -> Vec<&'a Example> {
    if mini {
        (0..3)
            .map(|_| &examples[rng.interval(examples.len() as u32 - 1) as usize])
            .collect()
    } else {
        examples.iter().collect()
    }
}
fn bits_equal(a: &Transformer, b: &Transformer) {
    for (left, right) in [(&a.p, &b.p), (&a.m, &b.m), (&a.v, &b.v)] {
        assert_eq!(
            left.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            right.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    assert_eq!(a.dims, b.dims);
    assert_eq!(a.hyper, b.hyper);
    assert_eq!(a.microbatch, b.microbatch);
}
fn progress(step: u64, examples: u64) -> Progress {
    Progress {
        completed_steps: step,
        examples_presented: examples,
        targets_presented: 2 * examples,
        optimizer_seconds: 0.25,
    }
}

#[test]
fn resumed_full_and_minibatch_updates_match_every_state_bit_at_one_two_and_six_layers() -> Result<()>
{
    let root = directory()?;
    let binding = crate::hash(b"frozen source, data and case fixture");
    let examples = examples();
    for layers in [1, 2, 6] {
        for mini in [false, true] {
            let mut original = model(layers);
            let mut rng = Mt::new(381);
            let mut seen = 0;
            for step in 1..=5 {
                let b = batch(&examples, &mut rng, mini);
                original.step(&b, step)?;
                seen += b.len() as u64;
            }
            let path = root.join(format!("{layers}-{mini}.bin"));
            let p = progress(5, seen);
            let digest = save_new(&path, &original, &rng, p.clone(), &binding)?;
            let mut restored = load(&path, &binding)?;
            assert_eq!(restored.body_sha256, digest);
            assert_eq!(restored.progress, p);
            assert_eq!(restored.rng.words(), rng.words());
            bits_equal(&original, &restored.model);
            for step in 6..=12 {
                let a = batch(&examples, &mut rng, mini);
                let b = batch(&examples, &mut restored.rng, mini);
                assert_eq!(rng.words(), restored.rng.words());
                assert_eq!(original.step(&a, step)?, restored.model.step(&b, step)?);
                bits_equal(&original, &restored.model);
            }
            let e = &examples[0];
            let actual = restored.model.logits(&e.tokens, &[0, e.tokens.len() - 1])?;
            let d = Dims {
                ctx: e.tokens.len(),
                ..restored.model.dims
            };
            let expected = reference::logits(
                d,
                &restored
                    .model
                    .p
                    .iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<_>>(),
                &e.tokens.iter().map(|&x| x as usize).collect::<Vec<_>>(),
            );
            for (slot, pos) in [0, e.tokens.len() - 1].into_iter().enumerate() {
                for (&a, &b) in actual[slot * d.vocab..][..d.vocab]
                    .iter()
                    .zip(&expected[pos * d.vocab..][..d.vocab])
                {
                    assert!((a - b).abs() < 1e-5, "restored logits versus f64 reference");
                }
            }
        }
    }
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn checkpoint_integrity_provenance_and_bounds_reject_corruption_without_overwrite() -> Result<()> {
    let root = directory()?;
    let binding = crate::hash(b"checkpoint fixture");
    let path = root.join("state.bin");
    let model = model(1);
    save_new(&path, &model, &Mt::new(1), progress(0, 0), &binding)?;
    let original = fs::read(&path)?;
    assert!(save_new(&path, &model, &Mt::new(2), progress(0, 0), &binding).is_err());
    assert_eq!(fs::read(&path)?, original);
    assert!(!root.join("state.bin.partial").exists());
    assert!(load(&path, &crate::hash(b"another source or data or case")).is_err());
    let header_len = u64::from_le_bytes(original[8..16].try_into().unwrap()) as usize;
    let mut corrupt = original.clone();
    corrupt[16 + header_len] ^= 1;
    let bad = root.join("bad.bin");
    fs::write(&bad, &corrupt)?;
    assert!(load(&bad, &binding).is_err());
    fs::write(&bad, &original[..original.len() - 1])?;
    assert!(load(&bad, &binding).is_err());
    let mut huge = original.clone();
    huge[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    fs::write(&bad, huge)?;
    assert!(load(&bad, &binding).is_err());
    fs::write(&bad, &original)?;
    fs::OpenOptions::new()
        .write(true)
        .open(&bad)?
        .set_len(1 << 40)?;
    assert!(load(&bad, &binding).is_err());
    fs::remove_file(&bad)?;

    let mut h: Header = serde_json::from_slice(&original[16..16 + header_len])?;
    h.dimensions[5] = usize::MAX;
    let changed = serde_json::to_vec(&h)?;
    let mut body = MAGIC.to_vec();
    body.extend_from_slice(&(changed.len() as u64).to_le_bytes());
    body.extend_from_slice(&changed);
    body.extend_from_slice(&original[16 + header_len..original.len() - 32]);
    let digest = Sha256::digest(&body);
    body.extend_from_slice(&digest);
    fs::write(&bad, body)?;
    assert!(load(&bad, &binding).is_err());
    bits_equal(&model, &load(&path, &binding)?.model);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn invalid_optimizer_state_and_progress_never_create_a_checkpoint() -> Result<()> {
    let root = directory()?;
    let binding = crate::hash(b"checkpoint fixture");
    let path = root.join("state.bin");
    let mut net = model(1);
    let mut p = progress(0, 0);
    p.optimizer_seconds = f64::NAN;
    assert!(save_new(&path, &net, &Mt::new(1), p, &binding).is_err());
    assert!(save_new(&path, &net, &Mt::new(1), progress(1, 0), &binding).is_err());
    net.v[0] = -1.;
    assert!(save_new(&path, &net, &Mt::new(1), progress(0, 0), &binding).is_err());
    net.v[0] = 0.;
    net.p[0] = f32::NAN;
    assert!(save_new(&path, &net, &Mt::new(1), progress(0, 0), &binding).is_err());
    assert!(!path.exists());
    assert!(!root.join("state.bin.partial").exists());
    fs::remove_dir_all(root)?;
    Ok(())
}
