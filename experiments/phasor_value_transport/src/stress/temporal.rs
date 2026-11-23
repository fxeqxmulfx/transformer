//! Orbit splitting and the unit LGT windows in DiffGPT's adapters.temporal_records
//! at 5a37c515. TemporalRecurrenceScore proves the difference and decoder laws.
use arithmetic::reference::{SplitMix64, power};
use basis::{data::Task, wire};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Recurrence {
    Fibonacci,
    Division,
}

type State = (u32, u32);

impl Recurrence {
    fn next(self, (a, b): State) -> State {
        let y = match self {
            Self::Fibonacci => (a + b) % 97,
            Self::Division => a * power(b, 95, 97) % 97,
        };
        (b, y)
    }

    /// Lexicographically first unseen state fixes each orbit's origin.
    /// Preserving the original ordering before shuffling fixes both splits.
    fn ordered_splits(self, seed: u64) -> basis::Result<[Vec<State>; 2]> {
        let mut unseen: BTreeSet<_> = (0..97)
            .flat_map(|a| (0..97).map(move |b| (a, b)))
            .filter(|&(a, b)| match self {
                Self::Division => a != 0 && b != 0,
                Self::Fibonacci => (a, b) != (0, 0),
            })
            .collect();
        let mut orbits = Vec::new();
        while let Some(&first) = unseen.first() {
            let mut orbit = Vec::new();
            let mut state = first;
            while unseen.remove(&state) {
                orbit.push(state);
                state = self.next(state);
            }
            if state != first {
                return Err("recurrence left its starting orbit".into());
            }
            orbits.push(orbit);
        }
        SplitMix64(seed ^ 0x6f72_6269_7473_0001).shuffle(&mut orbits);
        let mut splits = [Vec::new(), Vec::new()];
        for orbit in orbits {
            let which = usize::from(splits[0].len() > splits[1].len());
            splits[which].extend(orbit);
        }
        for (states, salt) in splits
            .iter_mut()
            .zip([0x7472_6169_6e00_0002, 0x7465_7374_0000_0003])
        {
            SplitMix64(seed ^ salt).shuffle(states);
        }
        Ok(splits)
    }
}

pub(super) fn build(case: &Value) -> basis::Result<(Task, Value)> {
    let data = &case["data"];
    let recurrence = match data["task"].as_str() {
        Some("division") => Recurrence::Division,
        Some("fibonacci") => Recurrence::Fibonacci,
        _ => return Err("unknown temporal recurrence".into()),
    };
    let order = data["order"].as_u64().ok_or("missing difference order")? as usize;
    let context = data["context"]
        .as_u64()
        .ok_or("missing temporal context length")? as usize;
    let supported = match recurrence {
        Recurrence::Division => matches!((order, context), (1, 4) | (2, 4) | (1, 48)),
        Recurrence::Fibonacci => matches!((order, context), (1 | 2, 2)),
    };
    if !supported || data["tokenizer"] != "lgt" || case["data_seed"] != 1 {
        return Err("expected an original unit-grid LGT recurrence".into());
    }
    let vocab = 256 << (order - 1);
    let center = (vocab / 2) as i64;
    let raw_length = context + order + 1;
    let states = recurrence.ordered_splits(1)?;
    let count = states[0].len() + states[1].len();
    let mut raw = format!("torch.int64:({count}, {raw_length}):").into_bytes();
    let mut splits = BTreeMap::new();
    let mut fingerprints = json!({});
    let mut index = 0;
    for (name, states) in ["train", "validation"].into_iter().zip(states) {
        let mut rows = Vec::with_capacity(states.len());
        for (a, b) in states {
            let mut state = (a, b);
            let mut values = vec![i64::from(a), i64::from(b)];
            while values.len() < raw_length {
                state = recurrence.next(state);
                values.push(i64::from(state.1));
            }
            raw.extend(values.iter().flat_map(|v| v.to_le_bytes()));
            let future = values[raw_length - 1];
            let last = values[raw_length - 2];
            let base = if order == 1 {
                last
            } else {
                2 * last - values[raw_length - 3]
            };
            for _ in 0..order {
                values = values.windows(2).map(|x| x[1] - x[0]).collect();
            }
            let mut tokens: Vec<_> = values.into_iter().map(|v| v + center).collect();
            let target = tokens.pop().ok_or("empty temporal window")?;
            if target - center + base != future {
                return Err("unit-grid decoder failed to recover the held-back value".into());
            }
            let mut targets = vec![-100; context];
            targets[context - 1] = target;
            let mut example = wire::example("recurrence", tokens, targets);
            example.changes.fill(false);
            example.row_index = Some(index);
            if recurrence == Recurrence::Fibonacci {
                example.field = Some((base, future));
            }
            rows.push(example);
            index += 1;
        }
        fingerprints[name] = wire::tensor_hash(&rows).into();
        splits.insert(name.to_owned(), rows);
    }
    fingerprints["raw"] = arithmetic::hash(&raw).into();
    Ok((
        Task {
            splits,
            vocab,
            selection: "validation".into(),
        },
        fingerprints,
    ))
}
