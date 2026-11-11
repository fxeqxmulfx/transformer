use convex_channels::{
    channel::Channel,
    provenance::source_hash,
    task::{self, Task},
};
use convex_recall_compare::{
    convex::{self, Convex, Kind},
    data::{self, Sample},
    engine::Transformer,
    mt::Mt,
};
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path, time::Instant};

enum Learner {
    Channel(Channel),
    Copy(Convex),
    Transformer(Box<Transformer>),
}

impl Learner {
    fn logits(&mut self, samples: &[Sample]) -> Vec<f64> {
        match self {
            Self::Channel(ch) => samples.iter().flat_map(|s| ch.logits(s)).collect(),
            Self::Copy(copy) => samples.iter().flat_map(|s| copy.logits(s)).collect(),
            Self::Transformer(t) => t.logits(samples),
        }
    }

    fn arrays(&self) -> Vec<Vec<f64>> {
        match self {
            Self::Channel(ch) => vec![ch.weights.clone()],
            Self::Copy(copy) => vec![copy.p.clone()],
            Self::Transformer(t) => {
                [&t.p, &t.m, &t.v].iter().map(|xs| xs.iter().map(|&x| x as f64).collect()).collect()
            }
        }
    }

    fn evaluate(&mut self, samples: &[Sample]) -> Value {
        let (mut correct, mut total, mut all_correct) = (0, 0, 0);
        let mut loss = 0.0;
        for chunk in samples.chunks(32) {
            let z = self.logits(chunk);
            let mut row = 0;
            for s in chunk {
                let mut all = true;
                for &target in &s.targets {
                    let scores = &z[row * task::CLASSES..][..task::CLASSES];
                    assert!(scores.iter().all(|x| x.is_finite()));
                    loss += convex::ce_gradient(scores, target).0;
                    let mut pred = 0;
                    for c in 1..task::CLASSES {
                        if scores[c] > scores[pred] {
                            pred = c;
                        }
                    }
                    correct += usize::from(pred == target);
                    total += 1;
                    all &= pred == target;
                    row += 1;
                }
                all_correct += usize::from(all);
            }
        }
        json!({"accuracy":correct as f64 / total as f64,"ce":loss / total as f64,
            "all_queries_correct":all_correct as f64 / samples.len() as f64,"queries":total})
    }

    fn record(&mut self, task: &Task, step: u64, elapsed: f64) -> Value {
        let extra = if let Self::Channel(ch) = self {
            let error = ch.feasibility_error();
            assert!(error < 1e-10, "channel left the feasible domain");
            let gap = ch.gap(&task.data.train.iter().collect::<Vec<_>>());
            assert!(gap.is_finite() && gap > -1e-10);
            json!({"feasibility_error":error,"fw_gap":gap,"rule_error":task::rule_error(ch,task)})
        } else {
            Value::Null
        };
        json!({"step":step,"elapsed_seconds":elapsed,"parameters":self.arrays()[0].len(),
            "train":self.evaluate(&task.data.train),"test":self.evaluate(&task.data.test),
            "channel":extra})
    }
}

fn pinned_kernite() {
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .args(["-C", "/home/misha/rust_projects/kgen"])
            .args(args)
            .output()
            .expect("read Kernite checkout");
        assert!(out.status.success());
        String::from_utf8(out.stdout).unwrap()
    };
    assert_eq!(git(&["rev-parse", "HEAD"]).trim(), "a1b6bc2c956e1a5f35960e26b209e1b6c213096c");
    assert!(git(&["status", "--porcelain"]).is_empty(), "Kernite has uncommitted changes");
}

fn write_new(path: &Path, value: &Value) {
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(path).unwrap();
    writeln!(file, "{}", serde_json::to_string_pretty(value).unwrap()).unwrap();
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        5,
        "usage: convex-channels run mqar|recoded channel|copy|transformer SEED OUT"
    );
    assert_eq!(args[0], "run");
    assert!(args[1] == "mqar" || args[1] == "recoded");
    let method = args[2].as_str();
    let seed: u32 = args[3].parse().unwrap();
    assert!(seed < 3, "the frozen protocol uses seeds 0,1,2");
    let dir = Path::new(&args[4]);
    pinned_kernite();
    let task = task::make(seed, args[1] == "recoded");
    assert!(task::separates_atoms(&task), "training does not satisfy the Lean premise");
    let (mut learner, steps) = match method {
        "channel" => (
            Learner::Channel(Channel::new(task::KEYS, task::KEYS, task::VALUES, task::CLASSES)),
            1000,
        ),
        "copy" => (Learner::Copy(Convex::new(Kind::Matching, &task.data)), 1000),
        "transformer" => {
            let mut t = Transformer::new(&task.data, seed, 32, 4, 0.1);
            t.hyper.max_lr = 0.003;
            (Learner::Transformer(Box::new(t)), 6000)
        }
        _ => panic!("unknown model"),
    };
    fs::create_dir_all(dir).unwrap();
    write_new(
        &dir.join("run.json"),
        &json!({"version":1,"task":args[1],"method":method,
        "seed":seed,"steps":steps,"keys":task::KEYS,"values":task::VALUES,"classes":task::CLASSES,
        "train_examples":task::TRAIN,"test_examples":task::TEST,"source_sha256":source_hash(),
        "train_sha256":data::fingerprint(&task.data.train),"test_sha256":data::fingerprint(&task.data.test),
        "atom_separation":true,"batch":if method=="transformer" {32} else {task::TRAIN},
        "optimizer":match method {"channel"=>"full-batch Frank-Wolfe eta=2/(step+1)",
            "copy"=>"full-batch projected GD lr=1",_=>"AdamW lr=.003 wd=.1 width32 layers2 heads4"},
        "output_rule":task.rule,"flat_target":task::spectrum(&task,None)}),
    );
    let mut log = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dir.join("metrics.jsonl"))
        .unwrap();
    let start = Instant::now();
    let mut record = |learner: &mut Learner, step| {
        let line = learner.record(&task, step, start.elapsed().as_secs_f64());
        writeln!(log, "{line}").unwrap();
        log.flush().unwrap();
        println!("{line}");
        line
    };
    record(&mut learner, 0);
    let full: Vec<_> = task.data.train.iter().collect();
    let mut rng = Mt::new(2_000_003u32.wrapping_mul(seed + 1));
    let mut final_record = Value::Null;
    for step in 1..=steps {
        match &mut learner {
            Learner::Channel(ch) => ch.step(&full, step),
            Learner::Copy(copy) => copy.step(&full),
            Learner::Transformer(t) => {
                let batch: Vec<_> =
                    (0..32).map(|_| full[rng.interval(full.len() as u32 - 1) as usize]).collect();
                t.step(&batch, step);
            }
        }
        if [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 6000].contains(&step) {
            final_record = record(&mut learner, step);
        }
    }
    if let Learner::Channel(ch) = &learner {
        final_record["spectrum"] = task::spectrum(&task, Some(ch));
    }
    write_new(&dir.join("final.json"), &final_record);
    write_new(&dir.join("parameters.json"), &json!({"step":steps,"arrays":learner.arrays()}));
}
