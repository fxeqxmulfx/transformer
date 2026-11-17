use super::{
    Result,
    cohort::protocol,
    gradient::{Batch, Sample},
    model::{DIM, Model, dot},
    train,
};
use baseline::data::Example;

fn fixture() -> Result<Model> {
    let mut m = Model::new(4, 6, 3., 0.55, 2., 7)?;
    for k in m.relative_start()..m.scale_index() {
        m.weights[k] = 0.14 * (k % 5) as f64;
    }
    Ok(m)
}

#[test]
fn pooled_complex_state_matches_complete_literal_path_sum() -> Result<()> {
    let m = fixture()?;
    let prefix = [0, 1, 0, 2, 3, 0];
    let n = prefix.len() - 1;
    let link = m.links(n);
    let pass = m.pass(&prefix, &link);
    let key_sum: f64 = pass.attention.iter().sum();
    assert!((key_sum - 1.).abs() < 1e-14);
    assert!((pass.mass.iter().sum::<f64>() - 1.).abs() < 1e-14);
    for row in link.chunks_exact(n) {
        assert!((row.iter().sum::<f64>() - 1.).abs() < 1e-14);
    }
    for c in 0..m.vocabulary {
        let real_inner = |a: &[f64], b: &[f64]| {
            a.as_chunks::<2>()
                .0
                .iter()
                .zip(b.as_chunks::<2>().0)
                .map(|(u, v)| u[0] * v[0] + u[1] * v[1])
                .sum::<f64>()
        };
        let a: Vec<_> = prefix[..n]
            .iter()
            .map(|&o| (m.scale() * real_inner(m.key(0), m.key(o))).exp())
            .collect();
        let mut literal = 0.;
        for j in 0..n {
            let row: Vec<_> = (0..n)
                .map(|k| m.weights[m.relative_index(j, k)].exp())
                .collect();
            for k in 0..n {
                literal += a[j] / a.iter().sum::<f64>() * row[k] / row.iter().sum::<f64>()
                    * real_inner(m.value(c), m.value(prefix[k]));
            }
        }
        assert!((pass.logits[c] - m.capacity * literal).abs() < 1e-13);
        assert!(pass.logits[c].abs() <= m.capacity);
    }
    assert_eq!(m.logits(&[])?, vec![0.; 4]);
    assert_eq!(m.logits(&[2])?, vec![0.; 4]);
    assert!(m.logits(&[-1]).is_err() && m.logits(&[4]).is_err() && m.logits(&[0; 7]).is_err());
    Ok(())
}
#[test]
fn full_gradient_matches_feasible_loss_queries_on_all_coordinates() -> Result<()> {
    let m = fixture()?;
    let a = Sample {
        prefix: vec![0, 1, 0, 2, 0],
        target: 2,
    };
    let b = Sample {
        prefix: vec![1, 1, 3, 2],
        target: 0,
    };
    let batch = Batch::new(vec![&a, &a, &b])?;
    let gradient = batch.analytic(&m);
    let all: Vec<_> = (0..m.weights.len()).collect();
    let fd = batch.finite(&m, &all, 1e-5);
    for i in all {
        assert!((gradient[i] - fd[i]).abs() < 2e-8, "coordinate {i}");
    }
    for range in [
        0..m.table_len(),
        m.table_len()..m.relative_start(),
        m.relative_start()..m.scale_index(),
        m.scale_index()..m.weights.len(),
    ] {
        assert!(gradient[range].iter().any(|g| g.abs() > 1e-6));
    }
    Ok(())
}
#[test]
fn opaque_object_renaming_and_projected_boundary_queries() -> Result<()> {
    let m = fixture()?;
    let permutation = [2, 0, 3, 1];
    let mut other = m.clone();
    for (old, &new) in permutation.iter().enumerate() {
        for base in [0, m.table_len()] {
            other.weights[base + new * DIM..base + (new + 1) * DIM]
                .copy_from_slice(&m.weights[base + old * DIM..base + (old + 1) * DIM]);
        }
    }
    let x = [3, 1, 0, 1, 3];
    let recoded = x.map(|o| permutation[o as usize] as i64);
    for (c, &new) in permutation.iter().enumerate() {
        assert_eq!(m.logits(&x)?[c], other.logits(&recoded)?[new]);
    }
    let mut boundary = m.clone();
    boundary.weights[..DIM].fill(0.);
    boundary.weights[0] = 1.;
    let proposal = boundary.perturb(1, 0.125);
    assert!(proposal.validate().is_ok() && proposal.weights[0] < 1. && proposal.weights[1] > 0.);
    assert_eq!(boundary.weights[0], 1.);
    assert!((dot(proposal.key(0), proposal.key(0)) - 1.).abs() < 2e-15);
    Ok(())
}
fn example(label: i64) -> Example {
    Example {
        task: "fixture".into(),
        tokens: vec![0, 2, 1, 0],
        targets: vec![-100, label, -100, label],
        changes: vec![false, true, false, false],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    }
}
#[test]
fn both_modes_keep_duplicate_occurrences_and_the_short_batch() -> Result<()> {
    let data = vec![example(2), example(2), example(1)];
    let mut p = protocol()?;
    p.updates = 2;
    p.batch = 2;
    p.coordinates_per_table = 4 * DIM;
    for mode in ["mini", "forward"] {
        let mut m = fixture()?;
        let report = train::fit(&mut m, &data, mode, 0, &p, || Ok(()))?;
        assert_eq!(report["examples_seen"], 3);
        assert_eq!(report["targets_seen"], 6);
        assert_eq!(report["minimum_coordinate_visits"], 2);
        for field in [
            "key_change_l2",
            "value_readout_change_l2",
            "relative_change_l2",
        ] {
            assert!(report[field].as_f64().unwrap() > 0.);
        }
        let queries = if mode == "forward" {
            4 * m.weights.len()
        } else {
            0
        };
        assert_eq!(report["minibatch_loss_queries"], queries);
    }
    Ok(())
}
