use convex_recall_compare::{convex, data, engine, mt::Mt};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

enum Learner {
    Convex(convex::Convex),
    Transformer(Box<engine::Transformer>),
}

impl Learner {
    fn step(&mut self, batch: &[&data::Sample], step: u64) {
        match self {
            Self::Convex(c) => c.step(batch),
            Self::Transformer(t) => t.step(batch, step),
        }
    }

    fn arrays(&self) -> Vec<Vec<f64>> {
        match self {
            Self::Convex(c) => vec![c.p.clone()],
            Self::Transformer(t) => {
                [&t.p, &t.m, &t.v].iter().map(|p| p.iter().map(|&v| v as f64).collect()).collect()
            }
        }
    }

    fn restore(&mut self, arrays: Vec<Vec<f64>>) {
        match self {
            Self::Convex(c) => {
                assert_eq!(arrays.len(), 1);
                assert_eq!(c.p.len(), arrays[0].len());
                c.p.clone_from(&arrays[0]);
            }
            Self::Transformer(t) => {
                assert_eq!(arrays.len(), 3);
                for (dst, src) in [&mut t.p, &mut t.m, &mut t.v].into_iter().zip(arrays) {
                    assert_eq!(dst.len(), src.len());
                    for (d, s) in dst.iter_mut().zip(src) {
                        *d = s as f32;
                    }
                }
            }
        }
    }

    fn evaluate(&mut self, samples: &[data::Sample], vocab: usize) -> Value {
        let (mut total, mut correct, mut all_correct) = (0usize, 0usize, 0usize);
        let mut ce = 0.0;
        for chunk in samples.chunks(32) {
            let logits = match self {
                Self::Convex(c) => chunk.iter().flat_map(|s| c.logits(s)).collect::<Vec<_>>(),
                Self::Transformer(t) => t.logits(chunk),
            };
            let mut row = 0;
            for sample in chunk {
                let mut all = true;
                for &target in &sample.targets {
                    let z = &logits[row * vocab..][..vocab];
                    assert!(z.iter().all(|x| x.is_finite()), "nonfinite logits");
                    let (loss, _) = convex::ce_gradient(z, target);
                    let mut pred = 0;
                    for c in 1..vocab {
                        if z[c] > z[pred] {
                            pred = c;
                        }
                    }
                    ce += loss;
                    correct += usize::from(pred == target);
                    all &= pred == target;
                    total += 1;
                    row += 1;
                }
                all_correct += usize::from(all);
            }
        }
        json!({"ce":ce / total as f64, "accuracy":correct as f64 / total as f64,
            "all_queries_correct":all_correct as f64 / samples.len() as f64, "queries":total})
    }

    fn parameters(&self) -> usize {
        match self {
            Self::Convex(c) => c.p.len(),
            Self::Transformer(t) => t.p.len(),
        }
    }
}

fn source_hash() -> String {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut hash = Sha256::new();
    for name in [
        "Cargo.toml",
        "Cargo.lock",
        "src/main.rs",
        "src/lib.rs",
        "src/data.rs",
        "src/convex.rs",
        "src/engine.rs",
        "src/transformer_model.rs",
        "../grok_division/src/gemm.rs",
        "../grok_division/src/layout.rs",
        "../grok_division/src/adam.rs",
        "../grok_division/src/mt.rs",
    ] {
        hash.update(name.as_bytes());
        hash.update(fs::read(base.join(name)).expect("source fingerprint input missing"));
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn checkpoint(path: &Path, learner: &Learner, rng: &Mt, step: u64, elapsed: f64) {
    let arrays = learner.arrays();
    let header = json!({"step":step, "elapsed":elapsed, "rng":rng.words(),
        "lengths":arrays.iter().map(Vec::len).collect::<Vec<_>>()});
    let bytes = serde_json::to_vec(&header).unwrap();
    let temp = path.with_extension("tmp");
    let mut file = BufWriter::new(fs::File::create(&temp).unwrap());
    file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&bytes).unwrap();
    for array in arrays {
        for x in array {
            file.write_all(&x.to_le_bytes()).unwrap();
        }
    }
    file.flush().unwrap();
    fs::rename(temp, path).unwrap();
}

fn resume(path: &Path, learner: &mut Learner, rng: &mut Mt) -> (u64, f64) {
    if !path.exists() {
        return (0, 0.0);
    }
    let bytes = fs::read(path).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let h: Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let words: Vec<_> = h["rng"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    *rng = Mt::from_words(&words);
    let mut offset = 8 + n;
    let mut arrays = Vec::new();
    for len in h["lengths"].as_array().unwrap() {
        let len = len.as_u64().unwrap() as usize;
        let values = bytes[offset..offset + 8 * len]
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        offset += 8 * len;
        arrays.push(values);
    }
    assert_eq!(offset, bytes.len(), "checkpoint size mismatch");
    learner.restore(arrays);
    (h["step"].as_u64().unwrap(), h["elapsed"].as_f64().unwrap())
}

fn record(
    learner: &mut Learner,
    dataset: &data::Dataset,
    log: &mut fs::File,
    task: &str,
    method: &str,
    seed: u32,
    step: u64,
    elapsed: f64,
) {
    let result = json!({"task":task,"method":method,"seed":seed,"step":step,
        "elapsed_seconds":elapsed,"parameters":learner.parameters(),
        "train":learner.evaluate(&dataset.train,dataset.vocab),
        "test":learner.evaluate(&dataset.test,dataset.vocab),
        "identity_error":match learner { Learner::Convex(c)=>c.identity_error(), _=>None }});
    println!("{result}");
    writeln!(log, "{result}").unwrap();
    log.flush().unwrap();
}

fn main() {
    let revision = std::process::Command::new("git")
        .args(["-C", "/home/misha/rust_projects/kgen", "rev-parse", "HEAD"])
        .output()
        .expect("read Kernite revision");
    assert!(revision.status.success());
    assert_eq!(
        String::from_utf8(revision.stdout).unwrap().trim(),
        "a1b6bc2c956e1a5f35960e26b209e1b6c213096c",
        "Kernite revision differs from the frozen protocol"
    );
    let clean = std::process::Command::new("git")
        .args(["-C", "/home/misha/rust_projects/kgen", "status", "--porcelain"])
        .output()
        .expect("read Kernite status");
    assert!(clean.status.success() && clean.stdout.is_empty(), "Kernite has uncommitted changes");
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert!(
        args.len() >= 6,
        "usage: convex-recall-compare run TASK METHOD SEED STEPS OUT [WIDTH=32 THREADS=4]"
    );
    assert_eq!(args[0], "run");
    let (task, method) = (args[1].as_str(), args[2].as_str());
    let seed: u32 = args[3].parse().unwrap();
    let steps: u64 = args[4].parse().unwrap();
    let dir = Path::new(&args[5]);
    let width = args.get(6).map_or(32, |s| s.parse().unwrap());
    let threads = args.get(7).map_or(4, |s| s.parse().unwrap());
    assert!(width > 0 && width % 4 == 0 && threads > 0);
    let dataset = match task {
        "mqar" => data::mqar(seed, 8, 16, 512, 512),
        "div17" => data::division(seed, 17),
        "div97" => data::division(seed, 97),
        _ => panic!("task must be mqar, div17, or div97"),
    };
    let batch = if task == "mqar" { 32 } else { 64 };
    let mut learner = match method {
        "matching" if dataset.keys > 0 => {
            Learner::Convex(convex::Convex::new(convex::Kind::Matching, &dataset))
        }
        "copy" if dataset.p > 0 => Learner::Convex(convex::Convex::new(convex::Kind::Copy, &dataset)),
        "table" if dataset.p > 0 => Learner::Convex(convex::Convex::new(convex::Kind::Table, &dataset)),
        "transformer" => {
            let decay = if dataset.keys > 0 { 0.1 } else { 1.0 };
            let mut t = engine::Transformer::new(&dataset, seed, width, threads, decay);
            if dataset.keys > 0 {
                t.hyper.max_lr = 3e-3;
            }
            Learner::Transformer(Box::new(t))
        }
        _ => panic!("invalid method for task"),
    };
    let spec = json!({"version":1,"task":task,"method":method,"seed":seed,
        "width":width,"threads":threads,"batch":batch,"source_sha256":source_hash(),
        "train_sha256":data::fingerprint(&dataset.train),"test_sha256":data::fingerprint(&dataset.test),
        "train_examples":dataset.train.len(),"test_examples":dataset.test.len(),
        "vocab":dataset.vocab,"ctx":dataset.ctx,"sup":dataset.sup,
        "separates":(dataset.keys>0).then(||data::separates(&dataset)),
        "copy_support_train":(dataset.p>0).then(||data::copy_support_accuracy(&dataset.train)),
        "copy_support_test":(dataset.p>0).then(||data::copy_support_accuracy(&dataset.test))});
    fs::create_dir_all(dir).unwrap();
    let metadata = dir.join("run.json");
    if metadata.exists() {
        let old: Value = serde_json::from_slice(&fs::read(&metadata).unwrap()).unwrap();
        assert_eq!(old, spec, "run configuration or source changed; use a new output directory");
    } else {
        fs::write(metadata, serde_json::to_string_pretty(&spec).unwrap()).unwrap();
    }
    let mut rng = Mt::new(2_000_003u32.wrapping_mul(seed.wrapping_add(1)));
    let path = dir.join("checkpoint.bin");
    let (completed, base_elapsed) = resume(&path, &mut learner, &mut rng);
    if completed >= steps && path.exists() {
        println!("already completed {completed} updates: {}", dir.display());
        return;
    }
    let mut log = OpenOptions::new().create(true).append(true).open(dir.join("metrics.jsonl")).unwrap();
    if !path.exists() {
        record(&mut learner, &dataset, &mut log, task, method, seed, 0, 0.0);
    }
    let start = Instant::now();
    for step in completed + 1..=steps {
        let samples: Vec<_> = (0..batch)
            .map(|_| &dataset.train[rng.interval(dataset.train.len() as u32 - 1) as usize])
            .collect();
        learner.step(&samples, step);
        if [1, 2, 5, 10, 20, 50, 100, 200, 500].contains(&step) || step % 1000 == 0 || step == steps {
            let elapsed = base_elapsed + start.elapsed().as_secs_f64();
            record(&mut learner, &dataset, &mut log, task, method, seed, step, elapsed);
            checkpoint(&path, &learner, &rng, step, base_elapsed + start.elapsed().as_secs_f64());
        }
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;

    #[test]
    fn resumed_updates_equal_uninterrupted_updates() {
        let data = data::division(7, 7);
        for method in ["copy", "transformer"] {
            let make = || {
                if method == "copy" {
                    Learner::Convex(convex::Convex::new(convex::Kind::Copy, &data))
                } else {
                    Learner::Transformer(Box::new(engine::Transformer::new(&data, 1, 8, 2, 1.0)))
                }
            };
            let mut original = make();
            let mut rng = Mt::new(123);
            let update = |learner: &mut Learner, rng: &mut Mt, step| {
                let samples: Vec<_> =
                    (0..4).map(|_| &data.train[rng.interval(data.train.len() as u32 - 1) as usize]).collect();
                learner.step(&samples, step);
            };
            for step in 1..=3 {
                update(&mut original, &mut rng, step);
            }
            let path =
                std::env::temp_dir().join(format!("convex-recall-{}-{method}.bin", std::process::id()));
            checkpoint(&path, &original, &rng, 3, 1.25);
            let mut restored = make();
            let mut rrng = Mt::new(0);
            assert_eq!(resume(&path, &mut restored, &mut rrng), (3, 1.25));
            for step in 4..=6 {
                update(&mut original, &mut rng, step);
                update(&mut restored, &mut rrng, step);
            }
            assert_eq!(rng.words(), rrng.words());
            assert_eq!(original.arrays(), restored.arrays());
            fs::remove_file(path).unwrap();
        }
    }
}
