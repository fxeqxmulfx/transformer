mod bench;
mod checks;
mod provenance;

use convex_channel_forward::model::Model;
use convex_channels::task;
use convex_recall_compare::data::fingerprint;
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path};

fn write_new(path: &Path, value: &Value) {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    writeln!(file, "{}", serde_json::to_string_pretty(value).unwrap()).unwrap();
}

fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn run(out: &Path) {
    assert!(!out.exists(), "use a fresh output directory");
    let environment = provenance::environment();
    let source = provenance::source_hash();
    fs::create_dir_all(out).unwrap();
    write_new(
        &out.join("run.json"),
        &json!({"protocol":bench::protocol(),"source_sha256":source,"environment":environment}),
    );
    for recoded in [false, true] {
        let name = if recoded { "recoded" } else { "mqar" };
        for seed in 0..3 {
            let task = task::make(seed, recoded);
            let (reference, visits) = Model::fit(bench::shape(), &task.data.train, 0).unwrap();
            let timing = bench::measure(&task.data.train);
            let mut batches = Vec::new();
            for (i, size) in bench::SIZES.into_iter().enumerate() {
                let (model, _, _) = bench::fit(&task.data.train, size);
                let diagnostics = checks::inspect(&task, size, &model, &reference);
                batches.push(json!({"batch_size":size,"model":checks::model_json(&model),
                    "diagnostics":diagnostics,"fit":timing[i+1]}));
            }
            let result = json!({"task":name,"seed":seed,"source_sha256":source,
                "train_sha256":fingerprint(&task.data.train),"test_sha256":fingerprint(&task.data.test),
                "reference_model":checks::model_json(&reference),"reference_visits":visits,
                "reference_fit":timing[0],"batches":batches});
            write_new(&out.join(format!("{name}-{seed}.json")), &result);
            println!("{name} seed {seed}: seven batch sizes and all prefixes agree");
        }
    }
}

fn report(out: &Path) -> Value {
    let metadata = read(&out.join("run.json"));
    assert_eq!(metadata["protocol"], bench::protocol());
    assert_eq!(metadata["source_sha256"], provenance::source_hash());
    let mut rows = Vec::new();
    for recoded in [false, true] {
        let name = if recoded { "recoded" } else { "mqar" };
        for seed in 0..3 {
            let saved = read(&out.join(format!("{name}-{seed}.json")));
            let task = task::make(seed, recoded);
            assert_eq!(saved["task"], name);
            assert_eq!(saved["seed"], seed);
            assert_eq!(saved["source_sha256"], metadata["source_sha256"]);
            assert_eq!(saved["train_sha256"], fingerprint(&task.data.train));
            assert_eq!(saved["test_sha256"], fingerprint(&task.data.test));
            let (reference, visits) = Model::fit(bench::shape(), &task.data.train, 0).unwrap();
            assert_eq!(checks::read_model(&saved["reference_model"]), reference);
            assert_eq!(saved["reference_visits"], visits);
            bench::validate_timing(&saved["reference_fit"]);
            let batches = saved["batches"].as_array().unwrap();
            assert_eq!(batches.len(), bench::SIZES.len());
            let brief = |dist: &Value| json!({"median_us":dist["median_us"],"p10_us":dist["p10_us"],"p90_us":dist["p90_us"]});
            for (&size, result) in bench::SIZES.iter().zip(batches) {
                assert_eq!(result["batch_size"], size);
                let model = checks::read_model(&result["model"]);
                checks::equivalent(
                    &result["diagnostics"],
                    &checks::inspect(&task, size, &model, &reference),
                );
                bench::validate_timing(&result["fit"]);
                rows.push(json!({"task":name,"seed":seed,"batch_size":size,
                    "fit":brief(&result["fit"]),"reference_fit":brief(&saved["reference_fit"]),
                    "reference_over_streaming":saved["reference_fit"]["median_us"].as_f64().unwrap()
                        /result["fit"]["median_us"].as_f64().unwrap(),
                    "diagnostics":result["diagnostics"],"model_payload_bytes":model.payload_bytes()}));
            }
        }
    }
    json!({"validated_cases":rows.len(),"source_sha256":metadata["source_sha256"],
        "protocol":metadata["protocol"],"environment":metadata["environment"],"cases":rows})
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        2,
        "usage: convex-channel-minibatch run|report OUT"
    );
    let out = Path::new(&args[1]);
    match args[0].as_str() {
        "run" => run(out),
        "report" => println!("{}", serde_json::to_string_pretty(&report(out)).unwrap()),
        _ => panic!("unknown command"),
    }
}
