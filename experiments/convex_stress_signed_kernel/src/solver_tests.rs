use crate::*;

fn gaussian_elimination(mut a: Vec<f64>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))
            .unwrap();
        for j in 0..n {
            a.swap(k * n + j, pivot * n + j);
        }
        b.swap(k, pivot);
        for i in k + 1..n {
            let ratio = a[i * n + k] / a[k * n + k];
            for j in k..n {
                a[i * n + j] -= ratio * a[k * n + j];
            }
            b[i] -= ratio * b[k];
        }
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            b[i] -= a[i * n + j] * b[j];
        }
        b[i] /= a[i * n + i];
    }
    b
}

#[test]
fn independent_solve_and_leave_source_out_system_agree() -> Result<()> {
    let points = vec![[0., 0.], [0.3, 1.], [0.8, 0.2], [1., 1.]];
    for spec in kernels::bank().into_iter().skip(1) {
        let ridge = match spec {
            Spec::Polynomial { ridge, .. } | Spec::Gaussian { ridge, .. } => ridge,
            _ => unreachable!(),
        };
        let (fit, _, residual) =
            kernels::Fit::new(points.clone(), spec.clone(), &mut Work::default())?;
        assert!(residual < solver::RESIDUAL_LIMIT);
        for excluded in 0..points.len() {
            let remaining: Vec<_> = (0..points.len()).filter(|&i| i != excluded).collect();
            let mut matrix = vec![];
            for &i in &remaining {
                for &j in &remaining {
                    matrix.push(spec.value(points[i], points[j]) + if i == j { ridge } else { 0. });
                }
            }
            let rhs = remaining
                .iter()
                .map(|&i| spec.value(points[i], points[excluded]))
                .collect();
            let solved = gaussian_elimination(matrix, rhs);
            let mut expected = vec![0.; points.len()];
            for (&state, weight) in remaining.iter().zip(solved) {
                expected[state] = weight;
            }
            solver::normalize(&mut expected)?;
            let actual = fit.excluded(excluded, &mut Work::default())?;
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 1e-8);
            }
            assert_eq!(actual[excluded], 0.);
        }
    }
    Ok(())
}

#[test]
fn signed_normalization_zero_rows_and_solver_refusals() -> Result<()> {
    let mut row = vec![-1., 1., 1., 0.];
    solver::normalize(&mut row)?;
    assert!(row[0] < 0. && row[1] > 0. && row[2] > 0.);
    assert!(solver::upper_mass(&row)? <= 1.);
    for (a, b) in row.iter().zip([-1. / 3., 1. / 3., 1. / 3., 0.]) {
        assert!((a - b).abs() < 1e-14);
    }
    let mut zero = vec![0.; 4];
    assert_eq!(solver::normalize(&mut zero)?, 1.);
    assert_eq!(zero, vec![0.; 4]);
    assert!(solver::normalize(&mut [f64::NAN]).is_err());
    assert!(solver::normalize(&mut [f64::MAX, f64::MAX]).is_err());
    for matrix in [
        vec![1., 2., 2., 1.],
        vec![1., 2., 0., 1.],
        vec![1., 1., 1., 1.],
        vec![f64::NAN, 0., 0., 1.],
    ] {
        assert!(solver::inverse(&matrix, 2).is_err());
    }
    assert!(solver::inverse(&[], 0).is_err());
    assert!(solver::inverse(&[], solver::MAX_OBSERVATIONS + 1).is_err());
    Ok(())
}

#[test]
fn extrapolation_has_negative_weights_and_bad_fits_keep_parent() -> Result<()> {
    let spec = Spec::Polynomial {
        degree: 1,
        axis: [1, 1],
        ridge: 0.001,
    };
    let (fit, _, _) = kernels::Fit::new(
        vec![[0., 0.], [0.5, 0.], [1., 0.]],
        spec,
        &mut Work::default(),
    )?;
    let w = fit.query([1.2, 0.], &mut Work::default())?;
    assert!(w.iter().any(|&a| a < 0.));
    assert!(solver::upper_mass(&w)? <= 1.);
    let model = tests::make()?;
    let bank = vec![
        Spec::Baseline,
        Spec::Polynomial {
            degree: 0,
            axis: [1, 1],
            ridge: -1.,
        },
    ];
    let mut work = Work::default();
    let active = model::active_blocks(model.parent());
    let (fits, reports) = kernels::fit_bank(model.parent(), &bank, &active, &mut work)?;
    assert!(!reports.is_empty());
    assert!(reports.iter().all(|r| r.refusal.is_some()));
    let mut selected = model.selection.clone();
    selected.weights = vec![0., 1.];
    let (rows, stats) = model::rows(model.parent(), &fits, &selected, &mut work)?;
    assert!(rows.iter().all(Option::is_none));
    assert_eq!(stats.retained_baseline_rows, stats.eligible_cells);
    Ok(())
}
