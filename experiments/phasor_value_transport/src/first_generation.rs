//! Gather the first greedy token only when the earlier forward used exactly
//! the same rectangular prompt batch. FixedEvaluation and FirstGeneration
//! establish the rollout identity, including compaction and termination.
use crate::data::Example;

pub(crate) fn from_identical_prompts(
    examples: &[Example],
    stride: usize,
    predictions: &[u32],
) -> Option<Vec<u32>> {
    let mut tokens = Vec::with_capacity(examples.len());
    for (row, example) in examples.iter().enumerate() {
        let prompt = &example.prompt;
        if prompt.is_empty() || prompt.len() > stride || example.tokens != *prompt {
            return None;
        }
        let position = row.checked_mul(stride)?.checked_add(prompt.len() - 1)?;
        tokens.push(*predictions.get(position)?);
    }
    Some(tokens)
}

/// The pinned greedy rollout requests a single next token for each active
/// prefix. The embedding lookup and the EOS/length rules remain external.
pub(crate) fn requested_rollout(
    engine: &mut crate::Engine,
    embeddings: &[f32],
    examples: &[Example],
) -> crate::Result<Vec<Vec<u32>>> {
    let width = engine.config.width;
    if embeddings.len() != width * engine.config.vocab {
        return Err("external embedding matrix has the wrong shape".into());
    }
    let batch = engine.config.max_batch;
    engine.with_fixed_slot_weights(|engine| {
        let mut result = Vec::with_capacity(examples.len());
        for block in examples.chunks(batch) {
            result.extend(crate::metrics::rollout(block, |contexts| {
                let length = contexts.iter().map(|c| c.len()).max().unwrap_or(0);
                if length == 0
                    || length > engine.config.max_length
                    || contexts.iter().any(|c| c.is_empty())
                {
                    return Err("greedy prefixes must be nonempty and fit in the engine".into());
                }
                let mut input = Vec::with_capacity(contexts.len() * length * width);
                let mut ends = Vec::with_capacity(contexts.len());
                for context in contexts {
                    for i in 0..length {
                        let id = context.get(i).copied().unwrap_or(0) as usize;
                        if id >= engine.config.vocab {
                            return Err(
                                "greedy prefix contains an ID outside the vocabulary".into()
                            );
                        }
                        input.extend_from_slice(&embeddings[id * width..(id + 1) * width]);
                    }
                    ends.push(context.len() - 1);
                }
                engine.requested_predictions(&input, length, &ends)
            })?);
        }
        Ok(result)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Kernite CUDA"]
    fn requested_logits_equal_full_logits_on_unequal_padded_dense_prefixes() -> crate::Result<()> {
        use crate::{Engine, config, layout};
        let mut cfg = config(3)?;
        cfg.width = 64;
        cfg.heads = 2;
        cfg.layers = 2;
        cfg.vocab = 37;
        cfg.max_length = 16;
        let mut weights = layout::initialize_model(&cfg, 9);
        if let Some(path) = cfg.head.peer_read().and_then(|r| r.path)
            && !path.soft_address
        {
            let layout = crate::slot_memory::Layout::new(&cfg);
            let offset = crate::architecture::memory_start(&cfg);
            for (i, value) in weights[offset + layout.path_start.unwrap()..offset + layout.end()]
                .iter_mut()
                .enumerate()
            {
                *value = (i as f32 * 0.17).cos() * 0.21;
            }
        }
        let mut engine = Engine::for_scoring(cfg.clone(), &weights)?;
        let length = 12;
        let input: Vec<_> = (0..3 * length)
            .flat_map(|i| {
                let id = (i * (i + 1)) % 9;
                weights[id * cfg.width..(id + 1) * cfg.width]
                    .iter()
                    .map(move |&x| x * (1. + (i % 3) as f32 / 32.))
            })
            .collect();
        let dense = engine.infer_embeddings(&input, 3, length)?;
        for ends in [[0, 5, 11], [11, 3, 2]] {
            let predictions = engine.requested_predictions(&input, length, &ends)?;
            let logits = engine.logits.read()?;
            for (b, (&t, &prediction)) in ends.iter().zip(&predictions).enumerate() {
                let i = (b * length + t) * cfg.vocab;
                let full = &dense.logits[i..i + cfg.vocab];
                assert!(
                    full.iter()
                        .zip(&logits[i..i + cfg.vocab])
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                assert!(full.iter().all(|&v| v <= full[prediction as usize]));
            }
        }
        assert_eq!(
            dense.logits,
            engine.infer_embeddings(&input, 3, length)?.logits
        );
        Ok(())
    }

    #[test]
    fn gather_real_row_ends_skip_padding_and_depend_only_on_the_inputs() {
        let mut batch = [
            Example::generation("example", vec![1, 8], vec![6]),
            Example::generation("example", vec![1, 7, 3, 2], vec![11]),
        ];
        let predictions = [8, 17, 99, 99, 8, 6, 9, 3];
        assert_eq!(
            from_identical_prompts(&batch, 4, &predictions),
            Some(vec![17, 3])
        );
        batch[0].targets.fill(-100);
        batch[0].answer[0] = 42;
        assert_eq!(
            from_identical_prompts(&batch, 4, &predictions),
            Some(vec![17, 3])
        );
        assert!(from_identical_prompts(&batch, 4, &predictions[..7]).is_none());
        batch[1].tokens[0] += 1;
        assert!(from_identical_prompts(&batch, 4, &predictions).is_none());
        batch[1].tokens = vec![1, 7, 3, 2, 12];
        assert!(from_identical_prompts(&batch, 5, &predictions).is_none());
    }

    #[test]
    #[ignore = "requires Kernite CUDA"]
    fn original_free_rollout_emits_the_same_tokens_with_and_without_a_reusable_prompt()
    -> crate::Result<()> {
        use crate::{Engine, config, layout, metrics};

        let mut cfg = config(3)?;
        cfg.width = 64;
        cfg.heads = 2;
        cfg.layers = 2;
        cfg.vocab = 37;
        cfg.max_length = 12;
        let weights = layout::initialize_model(&cfg, 9);
        let mut reference = Engine::for_scoring(cfg.clone(), &weights)?;
        let mut current = Engine::for_scoring(cfg, &weights)?;
        let mut batch = [
            Example::generation("example", vec![1, 2, 8], vec![6]),
            Example::generation("example", vec![1, 3], vec![7]),
            Example::generation("example", vec![1, 7, 5, 6, 4], vec![8]),
        ];
        // Different limits compact the active batch after the first emission.
        for (e, n) in batch.iter_mut().zip([1, 4, 3]) {
            e.limit = n;
        }
        for appended in [false, true] {
            if appended {
                // A teacher input contains a target continuation which is absent at generation.
                batch[1] = Example::generation("example", vec![1, 3], vec![11, 8, 4]);
            }
            let expected = reference.with_fixed_slot_weights(|e| {
                metrics::rollout(&batch, |contexts| {
                    let stride = contexts.iter().map(|p| p.len()).max().unwrap();
                    let mut packed = vec![0; contexts.len() * stride];
                    for (r, p) in contexts.iter().enumerate() {
                        packed[r * stride..r * stride + p.len()].copy_from_slice(p);
                    }
                    e.forward(
                        &packed,
                        &vec![u32::MAX; packed.len()],
                        contexts.len(),
                        stride,
                    )?;
                    let predictions = e.predictions.read()?;
                    Ok(contexts
                        .iter()
                        .enumerate()
                        .map(|(r, p)| predictions[r * stride + p.len() - 1])
                        .collect())
                })
            })?;
            let mut emitted = Vec::new();
            metrics::evaluate_with_completions(&mut current, &batch, true, |i, tokens| {
                assert_eq!(i, emitted.len());
                emitted.push(tokens.to_vec());
                Ok(())
            })?;
            assert_eq!(emitted, expected);
        }
        Ok(())
    }
}
