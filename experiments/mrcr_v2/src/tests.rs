use super::*;
use std::io::Cursor;

const CSV: &[u8] = b"queries,answer,num_relevant,context_len\n\"first\nquestion\",ABCDEFGHIJKLsecret,8,4096\nsecond,ABCDEFGHIJKLother,8,4097\n";

fn score(lines: &str) -> Result<serde_json::Value> {
    let mut out = Vec::new();
    let values = predictions(Cursor::new(lines))?;
    evaluate(
        CSV,
        "s",
        Action::Score {
            predictions: values,
            model: "fixture".into(),
            tools: false,
        },
        &mut out,
    )?;
    Ok(serde_json::from_slice(&out)?)
}

#[test]
fn every_id_exactly_once_and_empty_completions_count() {
    let inverse = "{\"id\":\"s/1\",\"response\":\"\"}\n{\"id\":\"s/0\",\"response\":\"ABCDEFGHIJKLsecret\"}\n";
    let report = score(inverse).unwrap();
    assert_eq!(report["mean_match_ratio"], 0.5);
    assert_eq!(report["perfect"], 1);
    assert!(score("").is_err());
    assert!(score(&inverse.replace("s/1", "other/1")).is_err());
    assert!(score(&format!("{inverse}{inverse}")).is_err());
    assert!(score(&format!("{inverse}{{\"id\":\"s/2\",\"response\":\"\"}}")).is_err());
}

#[test]
fn prompt_export_preserves_multiline_text_and_excludes_the_reference() {
    let mut out = Vec::new();
    evaluate(CSV, "s", Action::Prompts, &mut out).unwrap();
    let lines: Vec<serde_json::Value> = Cursor::new(out)
        .lines()
        .map(|s| serde_json::from_str(&s.unwrap()).unwrap())
        .collect();
    assert_eq!(
        lines,
        [
            json!({"id":"s/0","prompt":"first\nquestion"}),
            json!({"id":"s/1","prompt":"second"})
        ]
    );
    let wrong = String::from_utf8_lossy(CSV).replace(",8,", ",4,");
    assert!(evaluate(wrong.as_bytes(), "s", Action::Verify, Vec::new()).is_err());
}
