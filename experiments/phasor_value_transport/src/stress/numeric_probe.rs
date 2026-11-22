//! External numeric teachers for the four frozen GPT mechanism probes.
use super::*;
use probes::scoring::NumericTotals;

pub(super) struct Teachers {
    pub step: f64,
    pub offset: i64,
    pub splits: BTreeMap<String, Vec<Vec<u64>>>,
}

impl Teachers {
    pub fn score(&self, engine: &mut Engine, split: &str, rows: &[Example]) -> Result<Value> {
        let teachers = self
            .splits
            .get(split)
            .ok_or("missing unquantized test teachers")?;
        if teachers.len() != rows.len() {
            return Err("teacher and generated-row counts differ".into());
        }
        let vocabulary = engine.config.vocab;
        let mut totals = NumericTotals::default();
        let mut visited = 0;
        let measured = metrics::evaluate_with_completions(engine, rows, true, |index, tokens| {
            if index != visited {
                return Err("completion rows are out of source order".into());
            }
            totals
                .add(
                    teachers
                        .get(index)
                        .ok_or("completion beyond the source split")?,
                    tokens.iter().map(|&t| t as usize),
                    self.step,
                    self.offset,
                    vocabulary,
                )
                .map_err(|e| e.to_string())?;
            visited += 1;
            Ok(())
        })?;
        let mse = totals
            .mse()
            .filter(|mse| mse.is_finite())
            .ok_or("invalid source MSE")?;
        if visited != rows.len() {
            return Err("incomplete numeric free generation".into());
        }
        Ok(
            json!({"correct":measured["exact"],"total":visited,"mse":mse,
            "squared_error":totals.squared_error,"numeric_components":totals.components,
            "valid_numeric_components":totals.valid}),
        )
    }
}

/// Independently generate with the source scorer, which requests one prefix at
/// a time. This also checks masking and EOS against the batched GPU evaluator.
pub fn audit(name: &str, file: &Path, limit: Option<usize>) -> Result<()> {
    let (row, task, teachers) = catalog::load_declared(name)?;
    let mut teachers = teachers.ok_or("numeric-audit requires a numeric source probe")?;
    let (mut source, _) = catalog::probe_data(&row["case"]).map_err(|e| e.to_string())?;
    let split = &task.selection;
    let original = source
        .splits
        .get_mut(split)
        .ok_or("missing original split")?;
    let count = limit.unwrap_or(original.records.len());
    if count == 0 || count > original.records.len() {
        return Err("invalid audit row count".into());
    }
    original.records.truncate(count);
    original
        .teacher_words
        .as_mut()
        .ok_or("missing source teachers")?
        .truncate(count);
    teachers
        .splits
        .get_mut(split)
        .ok_or("missing numeric teachers")?
        .truncate(count);
    let cfg = task_config(&row, &task, false)?;
    let bytes = fs::read(file)?;
    if bytes.len() != layout::total_count(&cfg) * 4 {
        return Err("numeric checkpoint shape mismatch".into());
    }
    let weights: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    layout::validate(&cfg, &weights)?;
    let batched = {
        let mut engine = Engine::new(cfg.clone(), &weights)?;
        teachers.score(&mut engine, split, &task.splits[split][..count])?
    };
    let mut scalar_cfg = cfg;
    scalar_cfg.max_batch = 1;
    let vocab = scalar_cfg.vocab;
    let mut engine = Engine::new(scalar_cfg, &weights)?;
    let mut predict = |prefix: &[i64]| -> probes::Result<Vec<f64>> {
        let tokens: Vec<_> = prefix.iter().map(|&t| t as u32).collect();
        engine
            .forward(&tokens, &vec![u32::MAX; tokens.len()], 1, tokens.len())
            .map_err(|e| e.to_string())?;
        let logits = engine.logits.read().map_err(|e| e.to_string())?;
        Ok(logits[(tokens.len() - 1) * vocab..tokens.len() * vocab]
            .iter()
            .map(|&x| f64::from(x))
            .collect())
    };
    let official =
        probes::scoring::score(&source, split, &mut predict).map_err(|e| e.to_string())?;
    let expected = json!({"correct":official.raw.exact,"total":official.raw.example_count,
        "mse":official.mse,"squared_error":official.squared_error,
        "numeric_components":official.numeric_components,"valid_numeric_components":official.valid_numeric_components});
    if batched != expected {
        return Err("source rollout and batched numeric score differ".into());
    }
    println!(
        "{}",
        json!({"case":name,"split":split,"weights_sha256":hash(&bytes),
        "model":MODEL,"kernite_version":KERNITE_VERSION,"verified":expected})
    );
    Ok(())
}
