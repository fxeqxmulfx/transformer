//! DiffGPT 5a37c515, current_grokking/probes.py::fashion. ImagePatchScore
//! formalizes the four-pixel ties-to-even grid and the final supervised class.
use basis::{data::Task, wire};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek},
    path::Path,
    process::Command,
};

const REFERENCE: &str = include_str!(
    "../../../../.lake/reproduction/gpt/research/current_grokking/results/fashion_seed1.json"
);
const SIDE: usize = 28;
const PIXELS: usize = SIDE * SIDE;
const PATCHES: usize = 14 * 14;

fn archive(
    dir: &Path,
    name: &str,
    expected: &Value,
    images: bool,
    count: usize,
) -> basis::Result<Vec<u8>> {
    let path = dir.join(name);
    let mut file = File::open(&path).map_err(|err| format!("{}: {err}", path.display()))?;
    let mut packed = Vec::new();
    file.read_to_end(&mut packed)?;
    if expected[name] != basis::hash(&packed) {
        return Err(format!("wrong Fashion-MNIST archive digest for {name}").into());
    }
    drop(packed);
    file.rewind()?;
    let raw = Command::new("gzip").arg("-dc").stdin(file).output()?;
    if !raw.status.success() {
        return Err(format!("could not decode IDX archive {name}").into());
    }
    let shape = if images {
        vec![2051, count, 28, 28]
    } else {
        vec![2049, count]
    };
    let header = shape
        .iter()
        .map(|&v| (v as u32).to_be_bytes())
        .collect::<Vec<_>>()
        .concat();
    let stride = if images { PIXELS } else { 1 };
    let data = raw
        .stdout
        .strip_prefix(header.as_slice())
        .ok_or("wrong IDX header or image dimensions")?;
    if data.len() != count * stride || (!images && data.iter().any(|&v| v >= 10)) {
        return Err("inconsistent IDX payload size or class".into());
    }
    Ok(data.to_vec())
}

fn rounded(sum: u16) -> i64 {
    let q = sum / 4;
    i64::from(q + u16::from(sum % 4 > 2 || (sum % 4 == 2 && q % 2 == 1)))
}
fn pixels(image: &[u8]) -> Vec<i64> {
    let mut input = Vec::with_capacity(1 + PATCHES);
    input.push(266);
    for r in (0..SIDE).step_by(2) {
        for c in (0..SIDE).step_by(2) {
            let k = r * SIDE + c;
            let sum = [k, k + 1, k + SIDE, k + SIDE + 1]
                .into_iter()
                .map(|i| u16::from(image[i]))
                .sum();
            input.push(rounded(sum));
        }
    }
    input
}

pub(super) fn build(case: &Value, dir: &Path) -> basis::Result<(Task, Value)> {
    let source: Value = serde_json::from_str(REFERENCE)?;
    if source["protocol"] != *case {
        return Err("Fashion-MNIST source protocol changed".into());
    }
    let meta = &source["data_metadata"];
    let expected = &meta["source_sha256"];
    let images = archive(dir, "train-images-idx3-ubyte.gz", expected, true, 60000)?;
    let labels = archive(dir, "train-labels-idx1-ubyte.gz", expected, false, 60000)?;
    let held_images = archive(dir, "t10k-images-idx3-ubyte.gz", expected, true, 10000)?;
    let held_labels = archive(dir, "t10k-labels-idx1-ubyte.gz", expected, false, 10000)?;
    // The small-n CPU torch.randperm path uses one MT output per forward swap.
    let mut order: Vec<_> = (0..labels.len()).collect();
    let mut rng = basis::Random::torch_seed(1);
    for i in 0..order.len() - 1 {
        let j = i + rng.next_u32() as usize % (order.len() - i);
        order.swap(i, j);
    }
    let test: Vec<_> = (0..held_labels.len()).collect();
    let mut splits = BTreeMap::new();
    let mut hashes = json!({});
    for (name, order, im, ys) in [
        ("train", &order[..50000], &images, &labels),
        ("validation", &order[50000..], &images, &labels),
        ("test", &test[..], &held_images, &held_labels),
    ] {
        let rows: Vec<_> = order
            .iter()
            .map(|&i| {
                let x = pixels(&im[i * PIXELS..][..PIXELS]);
                let mut y = vec![-100; PATCHES + 1];
                y[PATCHES] = 256 + i64::from(ys[i]);
                let mut example = wire::example("fashion", x, y);
                example.changes.fill(false);
                example
            })
            .collect();
        let digest = wire::tensor_hash(&rows);
        if digest != source["data_fingerprints"][name] {
            return Err(format!("Fashion-MNIST {name} tensor digest differs: {digest}").into());
        }
        hashes[name] = json!(digest);
        splits.insert(name.to_owned(), rows);
    }
    Ok((
        Task {
            splits,
            vocab: 267,
            selection: "validation".into(),
        },
        hashes,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn patch_grid_preserves_nearest_even_means_and_row_major_order() {
        for s in 0..=1020 {
            assert_eq!(rounded(s), (f64::from(s) / 4.).round_ties_even() as i64);
        }
        let image: Vec<_> = (0..PIXELS).map(|i| ((i * 13 + 1) % 256) as u8).collect();
        let actual = pixels(&image);
        assert_eq!(actual[0], 266);
        assert_eq!(actual.len(), 197);
        for k in 0..196 {
            let mut exact = 0.;
            for (i, &p) in image.iter().enumerate() {
                if (i / 28) / 2 * 14 + (i % 28) / 2 == k {
                    exact += f64::from(p) / 4.;
                }
            }
            assert_eq!(actual[k + 1], exact.round_ties_even() as i64);
        }
    }
}
