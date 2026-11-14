//! Inspect all dimensional costs before training or reading evaluation labels.
use convex_stress_margin_budget::{Result, estimate, forecast, parent};
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
            let model = parent::Model::load(value, &binding)?;
            let bounds = estimate(&model)?;
            println!(
                "{}",
                json!({"source":path,"role":role,"mode":model.endpoint().mode,
                "bounds":bounds,"refusal":forecast(&model).err().map(|e|e.to_string())})
            );
        }
    }
    Ok(())
}
