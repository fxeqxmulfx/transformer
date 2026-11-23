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

#[cfg(test)]
mod tests {
    use super::*;

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
