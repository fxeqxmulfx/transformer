//! DiffGPT adapters.temporal_records at 5a37c515 uses the public codec for
//! both the full trajectory and the observed prefix. The actual prefix grid
//! also determines which generated token decodes to the reference future.
use super::*;
use crate::decoded_future::Decoder;
use source_codec::{Difference, Domain, Matrix, Tokenizer, decode, encode};

pub(crate) fn build(case: &Value) -> basis::Result<(Task, Value, Vec<Decoder>)> {
    let data = &case["data"];
    if case["data_seed"] != 1
        || data["task"] != "fibonacci"
        || data["order"] != 1
        || data["context"] != 2
    {
        return Err("expected the source first-difference Fibonacci configuration".into());
    }
    let tokenizer = match data["tokenizer"].as_str() {
        Some("mash") => Tokenizer::Mash,
        Some("mash-half") => Tokenizer::MashHalf,
        _ => return Err("expected a MASH temporal tokenizer".into()),
    };
    let parts = Recurrence::Fibonacci.ordered_splits(1)?;
    let cut = parts[0].len();
    let windows: Vec<_> = parts
        .into_iter()
        .flatten()
        .map(|state| Recurrence::Fibonacci.window(state, 4))
        .collect();
    let count = windows.len();
    let mut raw_digest = format!("torch.int64:({count}, 4):").into_bytes();
    for w in &windows {
        raw_digest.extend(w.iter().flat_map(|v| v.to_le_bytes()));
    }
    let columns: Vec<_> = (0..4)
        .flat_map(|t| windows.iter().map(move |w| w[t] as f64))
        .collect();
    let raw = Matrix::new(count, columns)?;
    let past = Matrix::new(count, raw.values()[..3 * count].to_vec())?;
    let vocab = 256i64;
    let center = vocab / 2;
    let domains = vec![Domain::symmetric(127.); count];
    let operators = vec![Difference::order(1)?; count];
    let full = encode(&raw, vocab, &domains, &operators, tokenizer)?;
    let history = encode(&past, vocab, &domains, &operators, tokenizer)?;
    if full.clipped_tokens != 0
        || history.clipped_tokens != 0
        || history.tokens.values() != &full.tokens.values()[..2 * count]
        || full.prefix != history.prefix
        || full.grid.scale != history.grid.scale
        || full.grid.zero != history.grid.zero
    {
        return Err("appending the future changed the temporal prefix or clipped a target".into());
    }
    let mut continued = history.tokens.values().to_vec();
    continued.extend(std::iter::repeat_n(center, count));
    let centered = decode(
        &history.prefix,
        &history.grid,
        &Matrix::new(count, continued)?,
        vocab,
        &operators,
        tokenizer,
    )?;
    let mut rows = Vec::with_capacity(count);
    let mut decoders = Vec::with_capacity(count);
    for (i, window) in windows.iter().enumerate() {
        let target = full.tokens.values()[2 * count + i];
        let input = vec![
            history.tokens.values()[i],
            history.tokens.values()[count + i],
        ];
        let base = centered.values()[3 * count + i];
        let step = history.grid.scale[i].recip();
        let decoder = Decoder {
            center,
            base,
            step,
            future: window[3],
        };
        if !decoder.accepts(target) {
            return Err(format!("source target fails to reconstruct the future at row {i}").into());
        }
        decoders.push(decoder);
        let mut row = wire::example("recurrence", input, vec![-100, target]);
        row.changes.fill(false);
        row.row_index = Some(i as i64);
        rows.push(row);
    }
    let held = rows.split_off(cut);
    let hashes = json!({"raw": arithmetic::hash(&raw_digest), "train":wire::tensor_hash(&rows),
        "validation":wire::tensor_hash(&held)});
    let splits = BTreeMap::from([("train".into(), rows), ("validation".into(), held)]);
    Ok((
        Task {
            splits,
            vocab: vocab as usize,
            selection: "validation".into(),
        },
        hashes,
        decoders,
    ))
}
