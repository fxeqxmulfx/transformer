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

/// Replay both score endpoints over the complete, digest-checked public files.
#[test]
#[ignore = "set MRCR_DATA_DIR to the two downloaded release CSVs"]
fn published_files_score_endpoints() -> Result<()> {
    let dir = std::path::PathBuf::from(std::env::var("MRCR_DATA_DIR")?);
    for (split, rows) in [("upto-128k", 484), ("at-1m", 310)] {
        let path = dir.join(format!("{split}.csv"));
        let mut gold = BTreeMap::new();
        for (index, record) in csv::Reader::from_reader(verified(split, &path)?)
            .deserialize::<Example>()
            .enumerate()
        {
            gold.insert(format!("{split}/{index}"), record?.answer);
        }
        assert_eq!(gold.len(), rows);
        let empty = gold.keys().map(|id| (id.clone(), String::new())).collect();
        for (predictions, expected) in [(gold, 1.0), (empty, 0.0)] {
            let action = Action::Score {
                predictions,
                model: "grader control".into(),
                tools: false,
            };
            let mut out = Vec::new();
            evaluate(verified(split, &path)?, split, action, &mut out)?;
            let report: serde_json::Value = serde_json::from_slice(&out)?;
            assert_eq!(report["rows"], rows);
            assert_eq!(report["mean_match_ratio"], expected);
            assert_eq!(report["perfect"], if expected == 1.0 { rows } else { 0 });
        }
    }
    Ok(())
}
