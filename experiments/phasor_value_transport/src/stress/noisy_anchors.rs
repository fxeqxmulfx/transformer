//! Pinned DiffGPT `series_data.anchor` and `SeriesTask`: one supervised
//! answer, original clean training support, and source field scoring.
use super::{differenced, temporal};
use basis::{data::Task, wire};
use minibatch::pcg::Pcg;
use serde_json::{Value, json};
use source_codec::{Difference, Domain, Matrix, Tokenizer, encode};
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
mod tests;

/// NumPy 2.5.4 Generator.choice without replacement uses Floyd's algorithm
/// for these populations. Its final shuffle consumes draws before integers().
pub(super) fn replacements(
    population: usize,
    fraction: f64,
    seed: u32,
) -> basis::Result<Vec<(usize, i64)>> {
    if !(0.0..=1.0).contains(&fraction) || population > 10_000 {
        return Err("expected a source noise fraction and a Floyd-sized population".into());
    }
    let size = (population as f64 * fraction).round_ties_even() as usize;
    let mut rng = Pcg::new(&[seed ^ 0xc077]);
    let mut used = vec![false; population];
    let mut indices = Vec::with_capacity(size);
    for j in population - size..population {
        let candidate = rng.below(j as u32 + 1) as usize;
        let index = if used[candidate] { j } else { candidate };
        used[index] = true;
        indices.push(index);
    }
    for i in (1..size).rev() {
        indices.swap(i, rng.below(i as u32 + 1) as usize);
    }
    indices.sort_unstable();
    Ok(indices
        .into_iter()
        .map(|i| (i, i64::from(rng.below(97))))
        .collect())
}

fn division(seed: u32) -> basis::Result<(Task, Vec<i64>)> {
    let levels = Matrix::new(1, (0..97).map(f64::from).collect())?;
    let encoded = encode(
        &levels,
        256,
        &[Domain::symmetric(96.)],
        &[Difference::order(0)?],
        Tokenizer::Lgt,
    )?;
    let table = encoded.tokens.values().to_vec();
    if table.iter().collect::<BTreeSet<_>>().len() != 97 || encoded.clipped_tokens != 0 {
        return Err("source LGT codec did not preserve all division residues".into());
    }
    let raw = differenced::division_rows(seed.into());
    let (training, held_out) = raw.split_at(raw.len() / 2);
    let mut splits = BTreeMap::new();
    for (name, rows) in [("train", training), ("validation", held_out)] {
        let examples = rows
            .iter()
            .map(|row| {
                let [a, b, q] = row.map(|v| table[v as usize]);
                let mut example = wire::example("div97", vec![a, b], vec![-100, q]);
                example.changes.fill(false);
                example
            })
            .collect();
        splits.insert(name.into(), examples);
    }
    Ok((
        Task {
            splits,
            vocab: 256,
            selection: "validation".into(),
        },
        table,
    ))
}

pub(super) fn build(case: &Value) -> basis::Result<(Task, Value)> {
    let data = &case["data"];
    let seed = u32::try_from(case["data_seed"].as_u64().ok_or("missing data seed")?)?;
    let noise = data["noise"].as_f64().ok_or("missing answer noise")?;
    let (mut task, division_table) = match (data["dataset"].as_str(), data["order"].as_u64()) {
        (Some("div97"), Some(0)) => {
            let (task, table) = division(seed)?;
            (task, Some(table))
        }
        (Some("fibonacci"), Some(1)) => {
            let configuration = json!({"data_seed":seed,"data":{
                "task":"fibonacci","order":1,"context":2,"tokenizer":"lgt"}});
            (temporal::build(&configuration)?.0, None)
        }
        _ => return Err("expected an original division or Fibonacci noise anchor".into()),
    };
    let train = task
        .splits
        .get_mut("train")
        .ok_or("missing anchor training split")?;
    let supported: BTreeSet<_> = train
        .iter()
        .flat_map(|r| r.tokens.iter().chain(r.targets.last()).copied())
        .collect();
    let selected = replacements(train.len(), noise, seed)?;
    let mut id_bytes = format!("<i8:({},):", selected.len()).into_bytes();
    let mut changed = 0;
    for &(index, residue) in &selected {
        id_bytes.extend_from_slice(&(index as i64).to_le_bytes());
        let row = &mut train[index];
        let target = row.targets.last_mut().ok_or("missing answer position")?;
        let token = match (&mut row.field, &division_table) {
            (None, Some(table)) => table[residue as usize],
            (Some((base, future)), None) => {
                *future = residue;
                128 + residue - *base
            }
            _ => return Err("missing or ambiguous anchor readout".into()),
        };
        changed += usize::from(*target != token);
        *target = token;
    }
    let validation = task
        .splits
        .get_mut("validation")
        .ok_or("missing anchor validation split")?;
    let full_test = validation.len();
    validation.retain(|row| {
        row.tokens
            .iter()
            .chain(row.targets.last())
            .all(|t| supported.contains(t))
    });
    let excluded = full_test - validation.len();
    let mut hashes = json!({"selected_ids_sha256":arithmetic::hash(&id_bytes),
        "selected_training_answers":selected.len(),"changed_training_answers":changed,
        "excluded_validation_rows":excluded});
    let mut index = 0;
    for name in ["train", "validation"] {
        let rows = task.splits.get_mut(name).ok_or("missing anchor split")?;
        for row in rows.iter_mut() {
            row.row_index = Some(index);
            index += 1;
        }
        hashes[name] = json!(wire::tensor_hash(rows));
    }
    Ok((task, hashes))
}
