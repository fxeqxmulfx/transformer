//! Numerical outcomes and per-case summaries survive; tensors, raw records,
//! long scalar curves and large observation arrays become explicit omissions.
#[cfg(test)]
use serde_json::{Value, json};
pub(super) fn payload(key: &str) -> bool {
    matches!(
        key,
        "parameters"
            | "weights"
            | "coefficients"
            | "real"
            | "imaginary"
            | "gamma"
            | "theta"
            | "logits"
            | "predictions"
            | "raw"
            | "raw_outputs"
            | "observations"
            | "samples"
            | "train_samples"
            | "validation_samples"
            | "train_rows"
            | "validation_rows"
            | "rows"
            | "features"
            | "train_inputs"
            | "val_inputs"
            | "latent_states"
            | "source_coordinates"
            | "embeddings"
            | "optimizer_state"
            | "state_dict"
            | "input_ids"
            | "targets"
            | "model"
            | "state"
            | "table"
            | "bank"
            | "decoder"
            | "coordinates"
            | "states"
    )
}
#[cfg(test)]
pub fn value(input: Value, key: &str) -> Value {
    if payload(key) {
        match &input {
            Value::Array(a) => return json!({"omitted_payload_items":a.len()}),
            Value::Object(m) => return json!({"omitted_payload_fields":m.len()}),
            _ => {}
        }
    }
    match input {
        Value::Object(map) => {
            if map.len() > 2048 {
                return json!({"omitted_object_fields":map.len()});
            }
            Value::Object(
                map.into_iter()
                    .map(|(k, v)| {
                        let compact = value(v, &k);
                        (k, compact)
                    })
                    .collect(),
            )
        }
        Value::Array(a)
            if a.len() > 512
                || (a.len() > 64 && a.iter().all(|v| !v.is_array() && !v.is_object())) =>
        {
            let count = a.len();
            let first = a.first().cloned().map(|v| value(v, ""));
            let last = a.last().cloned().map(|v| value(v, ""));
            json!({"omitted_array_items":count,"first":first,"last":last})
        }
        Value::Array(a) => Value::Array(a.into_iter().map(|v| value(v, "")).collect()),
        x => x,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn keeps_exact_counts_failures_modes_and_case_metrics() {
        let source = json!({"complete":true,"cases":[{"case":"example","mode":"Mini","metrics":{
            "correct":1963,"targets":1984,"loss":0.0031902142191690647},"refusal":null,"raw99":false,
            "model":{"weights":[1.,2.],"gamma":[3.,4.]}}],"parameters":[8.,9.]});
        let kept = value(source.clone(), "");
        assert_eq!(kept["cases"][0]["metrics"], source["cases"][0]["metrics"]);
        assert_eq!(kept["cases"][0]["raw99"], false);
        assert_eq!(kept["cases"][0]["mode"], "Mini");
        assert_eq!(kept["parameters"], json!({"omitted_payload_items":2}));
        assert!(kept["cases"][0]["model"]["weights"].is_null());
    }
    #[test]
    fn large_raw_arrays_are_explicitly_omitted_not_silently_sampled() {
        let kept = value(
            json!({"records":(0..1000).collect::<Vec<_>>(),"targets":42}),
            "",
        );
        assert_eq!(kept["records"]["omitted_array_items"], 1000);
        assert_eq!(kept["records"]["first"], 0);
        assert_eq!(kept["records"]["last"], 999);
        assert_eq!(kept["targets"], 42);
    }
}
