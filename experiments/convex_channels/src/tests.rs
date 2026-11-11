use crate::{channel::Channel, task};
use convex_recall_compare::{convex::ce_gradient, data::Sample, mt::Mt};

fn sample(values: Vec<usize>, queries: Vec<usize>, targets: Vec<usize>) -> Sample {
    Sample { values, queries, targets, tokens: Vec::new(), id: 0 }
}

#[test]
fn ordinary_ce_gradient_matches_independent_finite_differences() {
    let mut ch = Channel::new(2, 2, 2, 3);
    for (i, w) in ch.weights.iter_mut().enumerate() {
        *w += i as f64 * 0.003;
    }
    let data =
        [sample(vec![0, 1], vec![0, 1], vec![2, 0]), sample(vec![1, 0], vec![1, 0], vec![1, 2])];
    let samples: Vec<_> = data.iter().collect();
    let (_, gradient) = ch.loss_gradient(&samples);
    let loss = |ch: &Channel| -> f64 {
        data.iter()
            .map(|s| {
                let z = ch.logits(s);
                s.targets
                    .iter()
                    .enumerate()
                    .map(|(i, &t)| {
                        let row = &z[i * ch.classes..][..ch.classes];
                        row.iter().map(|x| x.exp()).sum::<f64>().ln() - row[t]
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
            / 4.0
    };
    for (i, &g) in gradient.iter().enumerate() {
        let old = ch.weights[i];
        ch.weights[i] = old + 1e-6;
        let plus = loss(&ch);
        ch.weights[i] = old - 1e-6;
        let minus = loss(&ch);
        ch.weights[i] = old;
        assert!(((plus - minus) / 2e-6 - g).abs() < 3e-9);
    }
}

#[test]
fn exact_oracle_matches_exhaustive_vertices() {
    let ch = Channel::new(1, 2, 2, 2);
    let mut rng = Mt::new(917);
    for _ in 0..100 {
        let g: Vec<_> = ch.weights.iter().map(|_| rng.normal()).collect();
        let vertex = ch.oracle(&g);
        let actual: f64 = g.iter().zip(&vertex).map(|(g, s)| g * s).sum();
        let mut optimum = f64::INFINITY;
        for k in 0..2 {
            for c0 in 0..2 {
                for c1 in 0..2 {
                    optimum = optimum.min(g[ch.index(0, k, 0, c0)] + g[ch.index(0, k, 1, c1)]);
                }
            }
        }
        assert!((actual - optimum).abs() < 1e-12);
        let candidate = Channel { weights: vertex, ..Channel::new(1, 2, 2, 2) };
        assert!(candidate.feasibility_error() < 1e-12);
    }
}

#[test]
fn answer_training_learns_generation_and_keeps_the_domain() {
    let mut ch = Channel::new(2, 2, 2, 4);
    let data: Vec<_> = (0..2)
        .flat_map(|a| (0..2).map(move |b| sample(vec![a, b], vec![0, 1], vec![3 - a, 3 - b])))
        .collect();
    let samples: Vec<_> = data.iter().collect();
    ch.step(&samples, 1);
    for s in &data {
        let z = ch.logits(s);
        for (i, &target) in s.targets.iter().enumerate() {
            assert_eq!(z[i * 4 + target], 1.0);
            assert!(!s.values.contains(&target));
        }
    }
    assert!(ch.feasibility_error() < 1e-12);
    assert!(ch.gap(&samples).abs() < 1e-12);
    let (loss, _) = ch.loss_gradient(&samples);
    assert!((loss - ((std::f64::consts::E + 3.0).ln() - 1.0)).abs() < 1e-12);
    for step in 2..20 {
        ch.step(&samples, step);
        assert!(ch.feasibility_error() < 1e-12);
    }
    let mut changed_labels = data[0].clone();
    changed_labels.targets.fill(0);
    assert_eq!(ch.logits(&changed_labels), ch.logits(&data[0]));
}

#[test]
fn visible_tokens_cover_inputs_but_do_not_contain_recoded_answers() {
    let task = task::make(731, true);
    assert!(task::separates_atoms(&task));
    let train: std::collections::HashSet<_> = task.data.train.iter().map(|s| &s.values).collect();
    for s in &task.data.test {
        assert!(!train.contains(&s.values));
    }
    for s in task.data.train.iter().chain(&task.data.test) {
        let mut parsed = vec![usize::MAX; task::KEYS];
        for pair in s.tokens[..2 * task::KEYS].chunks_exact(2) {
            assert_eq!(parsed[pair[0]], usize::MAX);
            parsed[pair[0]] = pair[1] - task::KEYS;
        }
        assert_eq!(parsed, s.values);
        assert_eq!(&s.tokens[2 * task::KEYS..], &s.queries);
        assert!(s.targets.iter().all(|t| !s.tokens.contains(t)));
    }
    let mut insufficient = task::make(731, true);
    insufficient.data.train.truncate(1);
    assert!(!task::separates_atoms(&insufficient));
}

#[test]
fn flat_bound_is_sharp_and_missing_rows_remain_ambiguous() {
    let target = task::spectrum(&task::make(731, true), None);
    assert_eq!(target["rank"], 4);
    assert_eq!(
        target["rank_d_truncation_errors"],
        serde_json::json!([16384., 12288., 8192., 4096., 0.])
    );
    // Independent enumeration of the two Boolean matrices in the Lean example.
    let a = [[0usize, 0], [0, 1]];
    let b = [[0usize, 0], [1, 0]];
    for table in [a, b] {
        for q in 0..2 {
            for r in 0..2 {
                let dot: i32 = (0..2)
                    .map(|v| (1 - 2 * table[q][v] as i32) * (1 - 2 * table[r][v] as i32))
                    .sum();
                assert_eq!(dot, if q == r { 2 } else { 0 });
            }
        }
        for v in 0..2 {
            let mut z = [0.0; 2];
            z[table[0][v]] = 1.0;
            assert!((ce_gradient(&z, 0).0 - (std::f64::consts::E + 1.0).ln() + 1.0).abs() < 1e-12);
        }
    }
    assert_eq!(a[0], b[0]);
    assert_ne!(a[1], b[1]);
}
