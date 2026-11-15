//! Label-free diagnosis of a frozen saved model on explicit input prefixes.
//! This command does not build benchmark data or score target labels.
use convex_stress_certified_head::{model::ModelEvaluator, previous, *};
use serde_json::json;
use std::{fs, path::Path};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 || args[1] != "diagnose" {
        return Err(
            "usage: convex-stress-certified-head diagnose MODEL_JSON PREFIXES_JSON OUTPUT_JSON"
                .into(),
        );
    }
    let (model_path, prefixes_path, output) = (
        Path::new(&args[2]),
        Path::new(&args[3]),
        Path::new(&args[4]),
    );
    if output.exists()
        || fs::metadata(model_path)?.len() > 64 * 1024 * 1024
        || fs::metadata(prefixes_path)?.len() > 64 * 1024 * 1024
    {
        return Err("fresh output and bounded model/prefix inputs required".into());
    }
    let model: previous::model::Model = previous::artifacts::read(model_path)?;
    let prefixes: Vec<Vec<i64>> = previous::artifacts::read(prefixes_path)?;
    if prefixes.is_empty() || prefixes.len() > 1024 {
        return Err("label-free diagnostic prefix count refused".into());
    }
    let evaluator = ModelEvaluator::new(&model)?;
    let mut results = vec![];
    for prefix in &prefixes {
        let checked = evaluator.inspect(prefix)?;
        let tokens: Vec<_> = prefix
            .iter()
            .map(|&x| usize::try_from(x))
            .collect::<std::result::Result<_, _>>()?;
        let old = model.query(&tokens)?;
        let other = model.independent_logits(&tokens)?;
        let residual = old
            .logits
            .iter()
            .zip(&other)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        let real_winner = joint::argmax(&old.logits);
        let complex_winner = joint::argmax(&other);
        results.push(json!({"tokens":prefix,"original_real_logits":old.logits,
            "original_complex_logits":other,"original_real_winner":real_winner,
            "original_complex_winner":complex_winner,"original_residual":residual,
            "original_consistency_refusal":residual>1e-7 || real_winner!=complex_winner,
            "certified":checked}));
    }
    previous::artifacts::write(
        output,
        &json!({"input_model_sha256":previous::artifacts::digest(model_path)?,
        "input_prefixes_sha256":previous::artifacts::digest(prefixes_path)?,
        "binary_sha256":previous::artifacts::digest(&std::env::current_exe()?)?,
        "forecast":forecast(model.programs.len(),model.head.coordinates.len())?,
        "target_labels_consumed":0,"new_accuracy_measurements":0,"queries":results}),
    )
}
