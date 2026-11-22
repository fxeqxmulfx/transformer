#[path = "../src/stress/class_translation.rs"]
mod comparison;
use comparison::{ScoredPair, probabilities, separable, split};

fn row(i: usize, j: usize, shift: usize) -> ScoredPair {
    let target = (2 * i + j) % 3;
    let mut mass = probabilities(&[1000., 999., -1000.]).unwrap();
    mass.rotate_right((target + shift) % 3);
    ScoredPair {
        input: [i, j],
        target: (target + shift) % 3,
        predicted: (target + shift) % 3,
        probabilities: mass,
    }
}

#[test]
fn exact_translations_and_a_changed_held_out_prediction() {
    let base: Vec<_> = (0..9).map(|n| row(n / 3, n % 3, 0)).collect();
    let mut other: Vec<_> = (0..9)
        .map(|n| row(n / 3, n % 3, n / 3 + 2 * (n % 3)))
        .collect();
    let mut shifts = vec![None; 9];
    let train = split(&base[..4], &other[..4], 3, &mut shifts).unwrap();
    assert_eq!(train["compared_probabilities"], 12);
    assert_eq!(train["max_probability_difference"], 0.);
    assert_eq!(train["mean_total_variation"], 0.);
    assert_eq!(train["shifted_argmax_disagreements"], 0);
    assert!(separable(&shifts, 3).is_err());
    other[8].probabilities.rotate_left(1);
    other[8].predicted = (other[8].predicted + 2) % 3;
    let held = split(&base[4..], &other[4..], 3, &mut shifts).unwrap();
    assert_eq!(held["pairs"], 5);
    assert_eq!(held["reference_correct"], 5);
    assert_eq!(held["candidate_correct"], 4);
    assert_eq!(held["shifted_argmax_disagreements"], 1);
    let max = probabilities(&[1., 0., -2000.]).unwrap()[0];
    assert!((held["max_probability_difference"].as_f64().unwrap() - max).abs() < 1e-14);
    assert!((held["mean_total_variation"].as_f64().unwrap() - max / 5.).abs() < 1e-14);
    assert!(separable(&shifts, 3).unwrap());
    shifts[8] = Some((shifts[8].unwrap() + 1) % 3);
    assert!(!separable(&shifts, 3).unwrap());
    assert!(split(&base[..1], &other[..1], 3, &mut shifts).is_err());
}

#[test]
fn invalid_logits_and_reordered_rows_are_rejected() {
    assert!(probabilities(&[]).is_err());
    assert!(probabilities(&[0., f32::NAN]).is_err());
    let mut shift = vec![None; 9];
    assert!(split(&[row(0, 0, 0)], &[row(0, 1, 0)], 3, &mut shift).is_err());
    assert_eq!(shift, vec![None; 9]);
}
