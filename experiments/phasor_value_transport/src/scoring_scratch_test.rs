use super::*;

#[test]
#[ignore = "requires Kernite CUDA"]
fn frozen_scoring_reuses_scores_across_depth_and_changing_batches() -> Result<()> {
    for (backbone, head) in [
        ("complex-mhc-lerope", head::Head::Original),
        ("complex-phase-score-lerope", head::Head::Original),
        ("complex-phase-score-lerope", architecture::HEAD),
    ] {
        let mut cfg = config(3)?;
        cfg.architecture = backbone.into();
        cfg.head = head;
        cfg.width = 64;
        cfg.heads = 2;
        cfg.layers = 3;
        cfg.gain = 2. / 64.;
        cfg.vocab = 37;
        cfg.max_length = 11;
        let scores = cfg.max_batch * cfg.heads * cfg.max_length.pow(2);
        let imaginary = if layout::phase_score(&cfg) { scores } else { 1 };
        let depth = cfg.layers;
        let weights = layout::initialize_model(&cfg, 17);
        let mut ordinary = Engine::new(cfg.clone(), &weights)?;
        let mut shared = Engine::for_scoring(cfg, &weights)?;
        assert_eq!(
            ordinary.attention_score_storage(),
            depth * (2 * scores + imaginary)
        );
        assert_eq!(shared.attention_score_storage(), scores + 1 + depth);
        for (batch, len, seed) in [(3, 11, 3), (1, 2, 17), (2, 7, 5), (1, 1, 11), (3, 10, 2)] {
            let dense: Vec<_> = (0..batch * len * 64)
                .map(|i| (((i * 29 + seed) % 197) as f32 - 90.) / 113.)
                .collect();
            let a = ordinary.infer_embeddings(&dense, batch, len)?;
            let b = shared.infer_embeddings(&dense, batch, len)?;
            assert_eq!(a.logits, b.logits);
            assert_eq!(a.hidden, b.hidden);
        }
        assert!(
            shared
                .embedding_pullback(&[0.; 64], 1, 1, &[1.; 37])
                .is_err()
        );
        assert!(shared.backward(1, 1).is_err());
        assert!(shared.update(1).is_err());
        assert!(shared.diagnostic_snapshot(1).is_err());
    }
    Ok(())
}

#[test]
#[ignore = "requires Kernite CUDA"]
fn tiled_phase_rows_match_every_float_across_tiles_and_sequences() -> Result<()> {
    for (width, heads) in [(64, 4), (64, 2), (128, 1)] {
        let mut cfg = config(2)?;
        cfg.architecture = "complex-phase-score-lerope".into();
        cfg.head = head::Head::Original;
        cfg.width = width;
        cfg.heads = heads;
        cfg.layers = 2;
        cfg.gain = heads as f32 / width as f32;
        cfg.vocab = 37;
        let boundary = engine::SCORE_TILE_ROWS / heads;
        cfg.max_length = boundary + 3;
        let weights = layout::initialize_model(&cfg, 37);
        let mut reference = Engine::new(cfg.clone(), &weights)?;
        let mut tiled = Engine::for_scoring(cfg.clone(), &weights)?;
        assert_eq!(
            tiled.attention_score_storage(),
            engine::SCORE_TILE_ROWS * cfg.max_length + 1 + cfg.layers
        );
        for (batch, length) in [
            (2, boundary + 3),
            (1, boundary - 1),
            (1, boundary),
            (2, boundary + 1),
            (1, 2),
        ] {
            let inputs: Vec<_> = (0..batch * length * width)
                .map(|i| (((i * 101 + length) % 241) as f32 - 117.) / 157.)
                .collect();
            let a = reference.infer_embeddings(&inputs, batch, length)?;
            let b = tiled.infer_embeddings(&inputs, batch, length)?;
            for (left, right) in [(a.logits, b.logits), (a.hidden, b.hidden)] {
                assert_eq!(left.len(), right.len());
                assert!(
                    left.iter()
                        .zip(&right)
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                    "width {width} heads {heads} batch {batch} length {length}"
                );
            }
        }
    }
    Ok(())
}
