//! Fixed equation frames from DiffGPT 5a37c515 mqar_div97.queries and Task.tensors.
//! The two splits and the random-label control retain their original draw order.
use arithmetic::reference::{SplitMix64, power};
use basis::{
    data::{Example, Task},
    wire,
};
use minibatch::pcg::Pcg;
use serde_json::{Value, json};
use source_codec::{Difference, Domain, Matrix, Tokenizer, encode};
use std::{cmp::Reverse, collections::BTreeMap};

#[path = "register_programs.rs"]
mod programs;

const HALF: usize = 4_656;
const PRIME: u32 = 97;

/// Every residue pair occurs once, including zero dividends. The source's
/// optional random relabeling takes place after the browser shuffle.
fn equations(random_labels: bool) -> Vec<[u32; 3]> {
    let mut rows: Vec<_> = (0..PRIME)
        .flat_map(|a| (1..PRIME).map(move |b| [a, b, a * power(b, PRIME - 2, PRIME) % PRIME]))
        .collect();
    SplitMix64(1 ^ 0x7370_6c69_7473_0002).shuffle(&mut rows);
    if random_labels {
        let entropy = 1 ^ 0x7261_6e64_6f6du64;
        let mut rng = Pcg::new(&[entropy as u32, (entropy >> 32) as u32]);
        for row in &mut rows {
            row[2] = rng.below(PRIME);
        }
    }
    rows
}

struct Frame {
    keys: Vec<usize>,
    order: Vec<usize>,
    operands: [usize; 2],
    following: Vec<u32>,
    latest: Vec<usize>,
    values: Vec<u32>,
}

fn contexts(
    loads: usize,
    extra: usize,
    equations: &[[u32; 3]],
    stream: u32,
    code: &[i64],
) -> Vec<Example> {
    let mut rng = Pcg::new(&[1, stream]);
    let writes = loads + extra;
    // NumPy makes one array-wide call per phase. Drawing an entire frame
    // before the next one would assign different contexts to the equations.
    let mut frames: Vec<_> = equations
        .iter()
        .map(|_| {
            let mut keys: Vec<_> = (1..=32).collect();
            rng.shuffle(&mut keys);
            keys.truncate(loads);
            Frame {
                keys,
                order: (0..loads).collect(),
                operands: [0; 2],
                following: Vec::new(),
                latest: vec![0; loads],
                values: vec![0; writes],
            }
        })
        .collect();
    for frame in &mut frames {
        frame
            .order
            .extend((0..extra).map(|_| rng.below(loads as u32) as usize));
    }
    for frame in &mut frames {
        rng.shuffle(&mut frame.order);
        for (time, &slot) in frame.order.iter().enumerate() {
            frame.latest[slot] = time;
        }
    }
    for frame in &mut frames {
        // Gap zero makes every loaded slot eligible. NumPy ranks the top
        // 53 bits in stable descending order to choose two different keys.
        let mut ranked: Vec<_> = (0..loads)
            .map(|slot| (Reverse(rng.next64() >> 11), slot))
            .collect();
        ranked.sort_unstable();
        frame.operands = [ranked[0].1, ranked[1].1];
    }
    for (frame, eq) in frames.iter_mut().zip(equations) {
        frame.following = (0..loads).map(|_| rng.below(PRIME)).collect();
        for (&slot, &value) in frame.operands.iter().zip(eq) {
            frame.following[slot] = value;
        }
    }
    for time in (0..writes).rev() {
        for frame in &mut frames {
            let slot = frame.order[time];
            let draw = rng.below(PRIME - 1);
            let after = &mut frame.following[slot];
            if frame.latest[slot] != time {
                *after = draw + u32::from(draw >= *after);
            }
            frame.values[time] = *after;
        }
    }
    frames
        .into_iter()
        .zip(equations)
        .map(|(frame, eq)| {
            let mut tokens = Vec::with_capacity(3 * writes + 2);
            for (&slot, &value) in frame.order.iter().zip(&frame.values) {
                tokens.extend([
                    code[96 - frame.keys[slot]],
                    code[0],
                    code[96 + value as usize],
                ]);
            }
            tokens.extend(frame.operands.map(|slot| code[96 - frame.keys[slot]]));
            let mut mask = vec![-100; tokens.len()];
            *mask.last_mut().unwrap() = code[96 + eq[2] as usize];
            let mut example = wire::example("associative", tokens, mask);
            example.changes.fill(false);
            example
        })
        .collect()
}

pub(super) fn build(case: &Value) -> basis::Result<(Task, Value)> {
    let data = &case["data"];
    let compute = data["compute"]
        .as_u64()
        .ok_or("missing number of compute rows")?;
    if case["kind"] != "associative" || case["data_seed"] != 1 {
        return Err("expected the original fixed-context equation protocol".into());
    }
    // Independent NumPy C distribution routines and the native source LGT
    // encoder produce these full input, supervision and padding fingerprints.
    let (loads, extra, random, expected) = match (
        compute,
        data["loads"].as_u64(),
        data["overwrites"].as_u64(),
        data["labels"].as_str(),
    ) {
        (1, Some(2), Some(0), Some("division")) => (
            2,
            0,
            false,
            [
                "21401639625292a957d6f8f367aa473dddbc6dd3e01529c65b640624a308b676",
                "999c21cc00c3db91bce8f3d791ef6392b8ca1db39018c96f4196f3649ba344af",
            ],
        ),
        (1, Some(2), Some(0), Some("random")) => (
            2,
            0,
            true,
            [
                "a306b6da79fbb2f54dd259f23f3eaddf4f51e6384942d8e7fd81eea074b8aaba",
                "0f9ede69f35a28284d3e66ce79deef49ede078c80a03c052a46d598d63544c0b",
            ],
        ),
        (1, Some(8), Some(0), Some("division")) => (
            8,
            0,
            false,
            [
                "92ca421c01560152c368915eadb406a805203651280a0f52f664c3017c534188",
                "234336e7c0c0ba1cd9afe5a4c9c81a51aa847665571e081439b5cb42dd4737f0",
            ],
        ),
        (1, Some(8), Some(4), Some("division")) => (
            8,
            4,
            false,
            [
                "16a60432906927760fba58a2be9c4dacf6cd3188a105105d8ed13b340dcb8418",
                "54f8deb7d54a1a7970c675681aa86d5b829a3330461435d9275ce9ec2874ffb0",
            ],
        ),
        (2, Some(8), Some(0), Some("division")) => (
            8,
            0,
            false,
            [
                "7e4769756de0e69a08b80254e0ef28c9838dcb732d7530911d93bf6cc46258d4",
                "6f03f8b19194d02d9301d6e9268d67d1142dd01f590532e5f87de30be522fbbc",
            ],
        ),
        (4, Some(8), Some(0), Some("division")) => (
            8,
            0,
            false,
            [
                "7a000d4b6d6bcc1e7d8f99ab26a060accc0161860178cda9c9a24c21400bf9ae",
                "7346c5bda1519edddcd2a262207596bd076de3adb0a88a47904a1a39c9ab4318",
            ],
        ),
        _ => return Err("unrecognized register equation configuration".into()),
    };
    let levels = Matrix::new(1, (-96..=96).map(f64::from).collect())?;
    let code = encode(
        &levels,
        256,
        &[Domain::symmetric(96.)],
        &[Difference::order(0)?],
        Tokenizer::Lgt,
    )?;
    let rows = equations(random);
    let mut splits = BTreeMap::new();
    let mut fingerprints = json!({});
    for ((name, stream), (equations, hash)) in [("train", 0), ("validation", 2)]
        .into_iter()
        .zip(rows.as_chunks::<HALF>().0.iter().zip(expected))
    {
        let examples = if compute == 1 {
            contexts(loads, extra, equations, stream, code.tokens.values())
        } else {
            programs::examples(compute as usize, equations, stream, code.tokens.values())
        };
        let actual = wire::tensor_hash(&examples);
        if actual != hash {
            return Err(format!("equation-frame fingerprint differs on {name}: {actual}").into());
        }
        fingerprints[name] = actual.into();
        splits.insert(name.into(), examples);
    }
    Ok((
        Task {
            splits,
            vocab: 256,
            selection: "validation".into(),
        },
        fingerprints,
    ))
}
