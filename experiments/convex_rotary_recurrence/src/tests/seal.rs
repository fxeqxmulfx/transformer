use crate::{atom::Family, experiment};
use bilinear::experiment as original;
use serde_json::{Value, json};
#[test]
fn scoring_requires_both_families_and_unchanged_producer_and_endpoint() {
    let root = std::env::temp_dir().join(format!("rotary-seal-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    assert!(experiment::score::sealed(&root).is_err());
    let protocol: Value = serde_json::from_str(experiment::PROTOCOL).unwrap();
    let binary = original::hash(&std::fs::read(std::env::current_exe().unwrap()).unwrap());
    for family in [Family::Plain, Family::Rotary] {
        assert!(experiment::score::sealed(&root).is_err());
        for case in protocol["cases"].as_array().unwrap() {
            for mode in ["mini", "forward"] {
                let cell =
                    experiment::score::cell(&root, case["name"].as_str().unwrap(), mode, 1, family);
                std::fs::create_dir_all(&cell).unwrap();
                original::write_json(&cell.join("metadata.json"),&json!({"source_fingerprint":experiment::fingerprint(),
                "protocol":protocol,"case":case,"binary_sha256":binary,"mode":mode,"seed":1,"family":family})).unwrap();
                original::write_json(&cell.join("training_receipt.json"),&json!({"status":"sealed","heldout_scored":false,
                "source_fingerprint":experiment::fingerprint(),"updates":case["maximum_updates"],"case":case["name"],
                "mode":mode,"seed":1,"family":family,"endpoint_sha256":original::hash(b"{}\n")})).unwrap();
                original::write_json(&cell.join("endpoint.json"), &json!({})).unwrap();
            }
        }
    }
    experiment::score::sealed(&root).unwrap();
    let cell = experiment::score::cell(
        &root,
        "gpt-probe-key-value-binding",
        "forward",
        1,
        Family::Rotary,
    );
    std::fs::write(cell.join("endpoint.json"), b"{\"changed\":true}").unwrap();
    assert!(experiment::score::sealed(&root).is_err());
    std::fs::write(cell.join("endpoint.json"), b"{}\n").unwrap();
    experiment::score::sealed(&root).unwrap();
    std::fs::write(cell.join("metadata.json"), b"{}").unwrap();
    assert!(experiment::score::sealed(&root).is_err());
    std::fs::remove_dir_all(root).unwrap();
}
