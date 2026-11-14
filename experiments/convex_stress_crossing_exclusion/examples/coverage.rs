//! Construct from saved CM49 source envelopes without reading evaluation data.
use convex_stress_crossing_exclusion::{Model, Result, digest, parent};
use serde_json::{Value, json};
use std::{fs::File, io::BufReader};

fn main() -> Result<()> {
    for path in std::env::args().skip(1) {
        let mut envelope: Value = serde_json::from_reader(BufReader::new(File::open(&path)?))?;
        for role in ["prior", "supplement"] {
            let value = envelope
                .as_object_mut()
                .ok_or("missing source envelope")?
                .remove(role)
                .ok_or("missing source role")?;
            if value.is_null() {
                continue;
            }
            let binding = serde_json::from_value(value["binding"].clone())?;
            let parent = parent::Model::load(value, &binding)?;
            let pin = digest(&parent)?;
            let model = Model::new(parent, &pin)?;
            println!(
                "{}",
                json!({"source":path,"role":role,"mode":model.endpoint().mode,
                "binding":model.binding(),"statistics":model.statistics(),"work":model.work(),
                "bounds":model.bounds(),"heldout_metrics_evaluated":false,
                "candidate_pool_closure_proved":false,"target_convexity_proved":false,
                "unconditional_raw_label_identification":false})
            );
        }
    }
    Ok(())
}
