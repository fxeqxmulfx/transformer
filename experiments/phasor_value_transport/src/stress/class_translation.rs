//! Compare complete class distributions in the CM121 evaluator.
use serde_json::{Value, json};

pub struct ScoredPair {
    pub input: [usize; 2],
    pub target: usize,
    pub predicted: usize,
    pub probabilities: Vec<f64>,
}

/// Compute the ordinary softmax of the checkpoint's unaltered FP32 scores.
pub fn probabilities(scores: &[f32]) -> Result<Vec<f64>, &'static str> {
    if scores.is_empty() || !scores.iter().all(|x| x.is_finite()) {
        return Err("class comparison requires finite nonempty scores");
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut mass: Vec<_> = scores
        .iter()
        .map(|&x| (f64::from(x) - f64::from(max)).exp())
        .collect();
    let sum: f64 = mass.iter().sum();
    for x in &mut mass {
        *x /= sum;
    }
    Ok(mass)
}

/// Compare each candidate class c+s with reference class c. The shift
/// comes solely from the two source targets in this evaluation row.
pub fn split(
    base: &[ScoredPair],
    other: &[ScoredPair],
    vocab: usize,
    shifts: &mut [Option<usize>],
) -> Result<Value, &'static str> {
    if base.is_empty()
        || base.len() != other.len()
        || vocab.checked_mul(vocab) != Some(shifts.len())
    {
        return Err("class comparison has inconsistent split or table dimensions");
    }
    let (mut reference_correct, mut candidate_correct, mut moved) = (0, 0, 0);
    let (mut deviation, mut total_variation) = (0.0_f64, 0.0_f64);
    for (a, b) in base.iter().zip(other) {
        let [i, j] = a.input;
        if a.input != b.input
            || i >= vocab
            || j >= vocab
            || a.target >= vocab
            || b.target >= vocab
            || a.predicted >= vocab
            || b.predicted >= vocab
            || a.probabilities.len() != vocab
            || b.probabilities.len() != vocab
        {
            return Err("class comparison requires aligned input pairs and class vectors");
        }
        let shift = (vocab + b.target - a.target) % vocab;
        if shifts[i * vocab + j].replace(shift).is_some() {
            return Err("duplicate input pair across class-comparison splits");
        }
        reference_correct += usize::from(a.predicted == a.target);
        candidate_correct += usize::from(b.predicted == b.target);
        moved += usize::from((a.predicted + shift) % vocab != b.predicted);
        for (c, &p) in a.probabilities.iter().enumerate() {
            let error = (p - b.probabilities[(c + shift) % vocab]).abs();
            deviation = deviation.max(error);
            total_variation += 0.5 * error;
        }
    }
    Ok(
        json!({"pairs":base.len(),"compared_probabilities":base.len()*vocab,
        "reference_correct":reference_correct,"candidate_correct":candidate_correct,
        "shifted_argmax_disagreements":moved,"max_probability_difference":deviation,
        "mean_total_variation":total_variation/base.len() as f64}),
    )
}

/// The same anchor-row/column identity as all_prediction_rectangles_iff_shift.
pub fn separable(cells: &[Option<usize>], p: usize) -> Result<bool, &'static str> {
    if p == 0 || p.checked_mul(p) != Some(cells.len()) || cells.iter().any(Option::is_none) {
        return Err("class comparison requires a complete square input table");
    }
    let value = |i: usize, j: usize| cells[i * p + j].unwrap();
    Ok((0..p).all(|i| {
        (0..p).all(|j| (value(i, j) + value(0, 0)) % p == (value(i, 0) + value(0, j)) % p)
    }))
}
