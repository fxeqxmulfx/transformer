//! Training-only probe. The input file contains Training alone; this binary
//! never reads task metadata, generators, validation rows or target formulas.
use convex_stress_pair_latents::{Result, Training, Work, selection};
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Write},
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [command, input, output] = args.as_slice() else {
        return Err(
            "usage: convex-stress-pair-latents select|encode TRAINING_JSON NEW_OUTPUT".into(),
        );
    };
    if !["select", "encode"].contains(&command.as_str()) {
        return Err("unknown training-only probe".into());
    }
    let file = File::open(input)?;
    if file.metadata()?.len() > 8 * 1024 * 1024 {
        return Err("training JSON exceeds 8MiB".into());
    }
    let training: Training = serde_json::from_reader(BufReader::new(file))?;
    let mut work = Work::new(joint::MAX_WORK)?;
    let selected = selection::select(&training, &mut work)?;
    let encoded = if command == "encode" {
        match selection::encode(&training, &selected, &mut work) {
            Ok((domain, rows)) => serde_json::json!({"shape":domain.shape,"records":rows.len(),
                "known_decoder_rows":domain.labels.iter().filter(|x|x.is_some()).count(),
                "complete_buffer_forecast":domain.shape.admit(rows.len())?,
                "both_interpreters_agree":true,"refusal":null}),
            Err(e) => serde_json::json!({"refusal":e.to_string()}),
        }
    } else {
        serde_json::Value::Null
    };
    let report = serde_json::json!({"selection":selected,"encoded":encoded,"work":work,
        "heldout_pools_read":0,"accuracy_measurements":0,"training_updates":0});
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?,
    );
    serde_json::to_writer(&mut writer, &report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}
