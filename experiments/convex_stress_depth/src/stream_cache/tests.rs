use super::*;
use crate::{
    features::{Execution, Mask, Transport},
    head::{Cache, ForwardChoice, Head},
    metrics::Predictor,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "convex-depth-stream-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn bank(depth: usize, transport: Transport) -> Bank {
    Bank {
        vocab: 3,
        depth,
        history_slots: 16,
        masks: vec![Mask::All, Mask::Pair(0, 2)],
        transport,
        execution: Execution::Cumulative,
    }
}
fn example(tokens: &[i64], targets: &[i64]) -> data::Example {
    data::Example {
        task: "generic-test".into(),
        tokens: tokens.to_vec(),
        targets: targets.to_vec(),
        changes: vec![false; tokens.len()],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    }
}
fn examples() -> Vec<data::Example> {
    vec![
        example(&[0, 1, 2], &[2, -100, 0]),
        example(&[2], &[1]),
        example(&[0, 1, 2], &[2, -100, 0]),
        example(&[1, 0], &[0, 2]),
    ]
}

#[test]
fn cumulative_features_retain_exact_composed_coordinates_and_nonzero_logits() -> Result<()> {
    let mut rng = convex_stress_hull::random::Random::new(91);
    for transport in [Transport::Residual, Transport::Fourier] {
        let deep = bank(4, transport);
        assert_eq!(deep.paths(), 2 * (1 + 2 + 4 + 8 + 16));
        let mut shallow = Head::zero(bank(1, transport), 20.)?;
        for (i, w) in shallow.coefficients.iter_mut().enumerate() {
            *w = (i % 13) as f64 / 1000. - 0.006;
        }
        let mut embedded = Head::zero(deep.clone(), shallow.radius)?;
        embedded.coefficients[..shallow.coefficients.len()].copy_from_slice(&shallow.coefficients);
        assert_eq!(embedded.validate()?, shallow.validate()?);
        for len in [1, 2, 7, 32, 128] {
            let prefix: Vec<_> = (0..len).map(|_| rng.below(3) as i64).collect();
            let all = deep.encode(&prefix)?;
            let (mut cursor, mut offset) = (0, 0);
            for depth in 0..=4 {
                let mut composed = bank(depth, transport);
                composed.execution = Execution::Composed;
                let expected: Vec<_> = composed
                    .encode(&prefix)?
                    .into_iter()
                    .map(|k| offset + k)
                    .collect();
                assert_eq!(&all[cursor..cursor + expected.len()], expected);
                cursor += expected.len();
                offset += composed.dimension();
            }
            assert_eq!(shallow.logits(&prefix), embedded.logits(&prefix));
            for y in 0..3 {
                assert_eq!(
                    head::cross_entropy(&shallow.logits(&prefix), y),
                    head::cross_entropy(&embedded.logits(&prefix), y)
                );
            }
        }
    }
    Ok(())
}

#[test]
fn streamed_rows_ce_gradients_and_updates_equal_dense_with_repeats_and_tails() -> Result<()> {
    let dir = Directory::new()?;
    let bank = bank(2, Transport::Fourier);
    let rows = examples();
    let dense = Cache::build(&bank, &rows)?;
    let path = dir.0.join("features.bin");
    let mut stream = StreamCache::build(&path, &bank, &rows)?;
    assert!(StreamCache::build(&path, &bank, &rows).is_err());
    let mut model = Head::zero(bank, 20.)?;
    for (i, w) in model.coefficients.iter_mut().enumerate() {
        *w = (i % 7) as f64 / 1000. - 0.003;
    }
    let mut left = vec![0.; model.coefficients.len()];
    let mut right = left.clone();
    for order in [dense.full_order(), vec![2, 0, 0, 3, 1], vec![3]] {
        let mut expected = vec![];
        for &e in &order {
            for i in dense.example_offsets[e]..dense.example_offsets[e + 1] {
                expected.push((dense.features(i).to_vec(), dense.labels[i]));
            }
        }
        let mut actual = vec![];
        stream.visit(&order, |f, y| {
            actual.push((f.to_vec(), y));
            Ok(())
        })?;
        assert_eq!(actual, expected);
        let a = head::objective(&model, &dense, &order, Some(&mut left))?;
        let b = objective::objective(&model, &mut stream, &order, Some(&mut right))?;
        assert_eq!(a, b);
        assert_eq!(left, right);
        assert_eq!(
            head::objective(&model, &dense, &order, None)?,
            objective::objective(&model, &mut stream, &order, None)?
        );
        let mut other = model.clone();
        head::gradient_step(&mut model, &left, 1. / stream.width() as f64)?;
        head::gradient_step(&mut other, &right, 1. / stream.width() as f64)?;
        assert_eq!(model.coefficients, other.coefficients);
    }
    assert!(stream.visit(&[usize::MAX], |_, _| Ok(())).is_err());
    let mut changed = model.clone();
    changed.bank.transport = Transport::Residual;
    assert!(objective::objective(&changed, &mut stream, &[0], None).is_err());
    Ok(())
}

#[test]
fn streamed_initialization_matches_dense_and_threshold_prefers_earlier_depth() -> Result<()> {
    let dir = Directory::new()?;
    let bank = bank(2, Transport::Residual);
    let mut rows = vec![example(&[0, 2], &[-100, 0]); 90];
    rows.extend(vec![example(&[1, 2], &[-100, 1]); 10]);
    let dense = Cache::build(&bank, &rows)?;
    let mut stream = StreamCache::build(&dir.0.join("features.bin"), &bank, &rows)?;
    let mut old = Head::zero(bank.clone(), 96.)?;
    let mut new = Head::zero(bank, 96.)?;
    let mut counts = vec![0.; old.coefficients.len()];
    let old_channel = head::forward_initialize(&mut old, &dense, &mut counts)?;
    let new_channel = objective::forward_initialize(
        &mut new,
        &mut stream,
        &mut counts,
        ForwardChoice::MinimumCe,
    )?;
    assert_eq!(old_channel, new_channel);
    assert_eq!(old.coefficients, new.coefficients);
    assert!(new_channel >= 2 * old.bank.masks.len());
    let chosen = objective::forward_initialize(
        &mut new,
        &mut stream,
        &mut counts,
        ForwardChoice::FirstBelow(0.4),
    )?;
    assert_eq!(chosen, 0);
    assert!(objective::objective(&new, &mut stream, &dense.full_order(), None)?.0 > 0.3);
    let chosen = objective::forward_initialize(
        &mut new,
        &mut stream,
        &mut counts,
        ForwardChoice::FirstBelow(1e-8),
    )?;
    assert_eq!(chosen, 2 * old.bank.masks.len());
    assert!(objective::objective(&new, &mut stream, &dense.full_order(), None)?.0 < 1e-8);
    assert!(
        objective::forward_initialize(
            &mut new,
            &mut stream,
            &mut counts,
            ForwardChoice::FirstBelow(f64::NAN)
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn streamed_reader_rejects_corrupt_records_boundaries_and_lengths() -> Result<()> {
    let dir = Directory::new()?;
    let valid = dir.0.join("valid.bin");
    let cache = StreamCache::build(&valid, &bank(1, Transport::Fourier), &examples())?;
    let start = cache.record_start as usize;
    drop(cache);
    let original = fs::read(&valid)?;
    let mut broken = original.clone();
    broken.pop();
    let path = dir.0.join("broken.bin");
    fs::write(&path, &broken)?;
    assert!(StreamCache::open(&path).is_err());
    broken = original.clone();
    broken.push(0);
    fs::write(&path, &broken)?;
    assert!(StreamCache::open(&path).is_err());
    broken = original.clone();
    let header_end = 16 + u64::from_le_bytes(broken[8..16].try_into()?) as usize;
    broken[header_end + 8..header_end + 16].fill(0);
    fs::write(&path, &broken)?;
    assert!(StreamCache::open(&path).is_err());
    for offset in [start, start + 4, start + 8] {
        broken = original.clone();
        broken[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        fs::write(&path, &broken)?;
        assert!(
            StreamCache::open(&path)?
                .visit(&[0], |_, _| Ok(()))
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn streamed_preflight_rejects_large_heads_and_disk_before_creating_a_cache() -> Result<()> {
    let dir = Directory::new()?;
    let mut large = bank(4, Transport::Fourier);
    large.vocab = 1024;
    large.history_slots = 4096;
    large.masks = vec![
        Mask::All,
        Mask::Pair(0, 1),
        Mask::Pair(0, 2),
        Mask::Pair(0, 3),
        Mask::Pair(1, 2),
        Mask::Pair(1, 3),
        Mask::Pair(2, 3),
    ];
    let path = dir.0.join("too-large.bin");
    assert!(!allocation(&large, 1, 1)?.eligible);
    assert!(StreamCache::build(&path, &large, &[example(&[0], &[0])]).is_err());
    assert!(!path.exists());
    let small = bank(0, Transport::Residual);
    assert!(!allocation(&small, 1_000_000_000, 1)?.eligible);
    assert!(allocation(&small, usize::MAX, 1).is_err());
    assert_eq!(
        allocation(&small, 10, 5)?.explicit_buffer_bytes,
        allocation(&small, 1_000_000, 5)?.explicit_buffer_bytes
    );
    Ok(())
}
