//! Paired comparison on one nested half-table split, chosen without labels.
use super::*;
use crate::mt::Mt;

mod inspection;
pub(crate) use inspection::inspect_checkpoints;

pub(super) fn words() -> Vec<u32> {
    let (mut observed, mut remaining): (Vec<_>, Vec<_>) = (0..256).partition(|&w| staircase(w));
    Mt::new(0).shuffle(&mut remaining);
    observed.extend_from_slice(&remaining[..128 - observed.len()]);
    observed.sort_unstable();
    observed
}

pub(super) fn split(observed: bool) -> Split {
    let train = words();
    Split::from_words(
        (0..256).filter(|w| train.binary_search(w).is_ok() == observed),
        parity,
    )
}

/// Score the original 31-word control checkpoints on exactly the 128 words
/// held out by both experiments. Verify each checkpoint digest before scoring.
pub(crate) fn score_control(dir: &Path) -> Result<Value> {
    let (updates, plan) = manifest(dir, None, true, Method::AdamW)?;
    let hidden = split(false);
    let mut records = Vec::new();
    for (draw, seeds) in plan["seeds"]
        .as_array()
        .ok_or("missing seeds")?
        .iter()
        .enumerate()
    {
        let cell = dir.join(format!("draw-{draw}"));
        let rec: Value = serde_json::from_slice(&fs::read(cell.join("result.json"))?)?;
        if rec["protocol"] != Method::AdamW.protocol_id()
            || rec["seeds"] != *seeds
            || rec["updates"] != updates
            || rec["optimizer"]["step"] != updates
        {
            return Err(format!("invalid 31-word control in draw {draw}").into());
        }
        let mut model = engine_with_batch(read_seed(seeds, 0)?, 128)?;
        load_weights(&cell, "weights.f32", &rec["weights_sha256"], &mut model)?;
        records.push(
            json!({"draw":draw,"seeds":seeds,"step":updates,"weights_sha256":rec["weights_sha256"],
            "common_hidden":hidden.score(&mut model)?}),
        );
    }
    Ok(json!({"control_protocol":plan,"half_train_ids":words(),"records":records}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_split_preserves_the_staircase_and_partitions_all_words() {
        let train = words();
        assert_eq!(train.len(), 128);
        let ids: Vec<_> = train.iter().map(|&w| w as u8).collect();
        assert_eq!(
            crate::hash(&ids),
            "aa85d606655f8d04d6a251e09719c809fbd26cf97b2e11db54fbcc456b04a412"
        );
        assert!(train.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(
            (0..256)
                .filter(|&w| staircase(w))
                .all(|w| train.contains(&w))
        );
        for observed in [true, false] {
            let split = split(observed);
            assert_eq!(split.tokens.len(), 128 * LENGTH);
            for (bits, labels) in split
                .tokens
                .as_chunks::<LENGTH>()
                .0
                .iter()
                .zip(split.labels.as_chunks::<LENGTH>().0)
            {
                let word = bits.iter().fold(0, |w, bit| (w << 1) | bit);
                assert_eq!(train.contains(&word), observed);
                assert!(labels[..LENGTH - 1].iter().all(|&label| label == u32::MAX));
                assert_eq!(labels[LENGTH - 1], parity(word));
            }
        }
    }
}
