//! Free-generation fit on the same training and validation prompts at a saved update.
use super::*;

fn letter(token: i64, words: &[String]) -> Option<usize> {
    let word = words.get(usize::try_from(token).ok()?)?;
    data::LETTERS.iter().position(|s| *s == word)
}

fn split(engine: &mut Engine, rows: &[Example], words: &[String]) -> Result<Value> {
    let mut matrix = [[0usize; 6]; 5];
    let mut seen = 0usize;
    let metric = metrics::evaluate_with_completions(engine, rows, true, |index, tokens| {
        if index != seen {
            return Err("fit scorer changed generation order".into());
        }
        seen += 1;
        let [target] = rows[index].answer.as_slice() else {
            return Err("expected a single reference letter".into());
        };
        let expected = letter(*target, words).ok_or("invalid reference letter")?;
        let predicted = match tokens {
            [token] => letter(i64::from(*token), words),
            _ => None,
        };
        matrix[expected][predicted.unwrap_or(5)] += 1;
        Ok(())
    })?;
    let correct: usize = (0..5).map(|i| matrix[i][i]).sum();
    if seen != rows.len() || metric["exact"] != correct {
        return Err("letter confusion counts disagree with generated exact answers".into());
    }
    let ce = metric["loss"]
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or("missing fit loss")?;
    Ok(json!({"correct":correct,"total":seen,"cross_entropy":ce,"confusion":matrix}))
}

pub fn run(seed: u32, out: &Path) -> Result<()> {
    let (row, task) = data()?;
    let cfg = task_config(&row, &task, true)?;
    let (weights, point) = checkpoint(seed, out, &row, &cfg)?;
    let mut engine = Engine::for_scoring(cfg, &weights)?;
    let words = vocabulary();
    let train = split(&mut engine, &task.splits["train"], &words)?;
    let validation = split(&mut engine, &task.splits[&task.selection], &words)?;
    if validation["correct"] != point["correct"] {
        return Err("fit validation differs from the checkpoint observation".into());
    }
    println!(
        "{}",
        json!({"study":NAME,"model":MODEL,"seed":seed,"step":point["step"],
        "weights_sha256":weight_digest(&weights),"protocol_sha256":row["catalog_sha256"],"kernite":KERNITE_VERSION,
        "references":data::LETTERS,"predictions":["A","B","C","D","E","invalid"],
        "train":train,"validation":validation})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_printed_answer_letters_have_confusion_columns() {
        let words = vocabulary();
        for (i, s) in words.iter().enumerate() {
            assert_eq!(
                letter(i as i64, &words),
                data::LETTERS.iter().position(|x| *x == s)
            );
        }
        assert_eq!(letter(-1, &words), None);
        assert_eq!(letter(words.len() as i64, &words), None);
    }
}
