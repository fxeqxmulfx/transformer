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

fn difference(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn run(out: &Path) {
    assert!(!out.exists(), "use a fresh output directory");
    let environment = provenance::environment();
    fs::create_dir_all(out).unwrap();
    let source = provenance::source_hash();
    write_new(
        &out.join("run.json"),
        &json!({"protocol":bench::protocol(),
        "source_sha256":source,"environment":environment}),
    );
    for recoded in [false, true] {
        let name = if recoded { "recoded" } else { "mqar" };
        for seed in 0..3 {
            let task = task::make(seed, recoded);
            let (direct, visits) = Model::fit(bench::shape(), &task.data.train, 0)
                .expect("a query has no consistent key");
            let direct_dense = direct.to_channel();
            let fw = bench::fit_fw(&task.data.train, 1);
            let endpoint = bench::fit_fw(&task.data.train, bench::ENDPOINT);
            let compact =
                Model::compact(&fw).expect("one-step baseline is not a deterministic vertex");
            checks::check_compact(&direct, &direct_dense, &task);
            checks::check_compact(&compact, &fw, &task);
            let bound = task::KEYS
                * task
                    .data
                    .train
                    .iter()
                    .map(|s| s.targets.len())
                    .sum::<usize>();
            assert!(visits <= bound);
            let timing = bench::measure(&task.data.train, &task.data.test, &direct, &fw);
            let value = json!({"task":name,"seed":seed,"source_sha256":source,
                "train_sha256":fingerprint(&task.data.train),"test_sha256":fingerprint(&task.data.test),
                "identifying_data":checks::identifying_data(&task),"atom_separation":task::separates_atoms(&task),
                "candidate_visits":visits,"candidate_visit_bound":bound,
                "direct_model":checks::model_json(&direct),"fw_compact_model":checks::model_json(&compact),
                "fw_one_weights":fw.weights,"fw_endpoint_weights":endpoint.weights,
                "direct":checks::inspect(&direct_dense,&task),"fw_one":checks::inspect(&fw,&task),
                "fw_endpoint":checks::inspect(&endpoint,&task),
                "direct_fw_difference":difference(&direct_dense.weights,&fw.weights),
                "fw_endpoint_difference":difference(&fw.weights,&endpoint.weights),"timing":timing});
            write_new(&out.join(format!("{name}-{seed}.json")), &value);
            println!(
                "{name} seed {seed}: direct/test={} fw/test={} train median us: direct={} fw={}",
                value["direct"]["test"]["accuracy"],
                value["fw_one"]["test"]["accuracy"],
                value["timing"]["fit"]["direct"]["median_us"],
                value["timing"]["fit"]["fw_one"]["median_us"]
            );
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
            assert_eq!(saved["identifying_data"], checks::identifying_data(&task));
            assert_eq!(saved["atom_separation"], task::separates_atoms(&task));
            let direct = checks::read_model(&saved["direct_model"]);
            let (fresh, visits) = Model::fit(bench::shape(), &task.data.train, 0).unwrap();
            assert_eq!(direct, fresh);
            assert_eq!(saved["candidate_visits"].as_u64().unwrap() as usize, visits);
            let bound = task::KEYS
                * task
                    .data
                    .train
                    .iter()
                    .map(|s| s.targets.len())
                    .sum::<usize>();
            assert_eq!(
                saved["candidate_visit_bound"].as_u64().unwrap() as usize,
                bound
            );
            assert!(visits <= bound);
            let direct_dense = direct.to_channel();
            let fw = checks::read_channel(&saved["fw_one_weights"]);
            let endpoint = checks::read_channel(&saved["fw_endpoint_weights"]);
            assert_eq!(fw.weights, bench::fit_fw(&task.data.train, 1).weights);
            let compact = checks::read_model(&saved["fw_compact_model"]);
            assert_eq!(Some(compact.clone()), Model::compact(&fw));
            checks::check_compact(&direct, &direct_dense, &task);
            checks::check_compact(&compact, &fw, &task);
            for (method, model) in [
                ("direct", &direct_dense),
                ("fw_one", &fw),
                ("fw_endpoint", &endpoint),
            ] {
                checks::equivalent(&saved[method], &checks::inspect(model, &task));
            }
            checks::equivalent(
                &saved["direct_fw_difference"],
                &json!(difference(&direct_dense.weights, &fw.weights)),
            );
            checks::equivalent(
                &saved["fw_endpoint_difference"],
                &json!(difference(&fw.weights, &endpoint.weights)),
            );
            bench::validate_timing(&saved["timing"]);
            let fit = &saved["timing"]["fit"];
            let predict = &saved["timing"]["prediction_per_dictionary"];
            let brief = |distribution: &Value| {
                json!({"median_us":distribution["median_us"],
                "p10_us":distribution["p10_us"],"p90_us":distribution["p90_us"]})
            };
            rows.push(json!({"task":name,"seed":seed,"direct":saved["direct"],
                "fw_one":saved["fw_one"],"fw_endpoint":saved["fw_endpoint"],
                "candidate_visits":visits,"candidate_visit_bound":bound,
                "direct_fit":brief(&fit["direct"]),"fw_one_fit":brief(&fit["fw_one"]),
                "training_speed_ratio":fit["fw_one"]["median_us"].as_f64().unwrap()/fit["direct"]["median_us"].as_f64().unwrap(),
                "direct_predict":brief(&predict["direct"]),"fw_dense_predict":brief(&predict["fw_dense"]),
                "fw_compact_predict":brief(&predict["fw_compact"]),
                "direct_payload_bytes":direct.payload_bytes(),"fw_compact_payload_bytes":compact.payload_bytes(),
                "direct_fw_difference":saved["direct_fw_difference"],"fw_endpoint_difference":saved["fw_endpoint_difference"]}));
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
        "usage: convex-channel-forward run|report OUT"
    );
    let out = Path::new(&args[1]);
    match args[0].as_str() {
        "run" => run(out),
        "report" => println!("{}", serde_json::to_string_pretty(&report(out)).unwrap()),
        _ => panic!("unknown command"),
    }
}
