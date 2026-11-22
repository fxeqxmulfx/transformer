#[path = "../src/shift_diagnostics.rs"]
mod shift_diagnostics;

use serde_json::{Value, json};
use shift_diagnostics::Residuals;

// Enumerate each distinct pair or four-corner rectangle directly. This
// oracle uses signed arithmetic rather than column-difference buckets.
fn enumerate<const P: usize>(errors: &[[Option<i32>; P]; P]) -> Value {
    let mut counts = [[0_u64; 2]; 3];
    let mut observe = |kind: usize, residual: i32| {
        counts[kind][0] += u64::from(residual.rem_euclid(P as i32) == 0);
        counts[kind][1] += 1;
    };
    for i in 0..P {
        for j in 0..P {
            let Some(x) = errors[i][j] else { continue };
            for &y in errors[i][..j].iter().flatten() {
                observe(0, x - y);
            }
            for k in 0..i {
                if let Some(y) = errors[k][j] {
                    observe(1, x - y);
                    for l in 0..j {
                        if let (Some(z), Some(w)) = (errors[i][l], errors[k][l]) {
                            observe(2, x - y - z + w);
                        }
                    }
                }
            }
        }
    }
    let [rows, cols, rects] =
        counts.map(|[consistent, total]| json!({ "consistent": consistent, "total": total }));
    json!({"row_pairs":rows, "column_pairs":cols, "rectangles":rects})
}

#[test]
fn masked_tables_match_direct_enumeration() {
    const P: usize = 7;
    for mask in 0..5 {
        let mut cells = Residuals::<P>::new();
        let mut errors = [[None; P]; P];
        for (a, row) in errors.iter_mut().enumerate() {
            for (b, e) in row.iter_mut().enumerate() {
                if (a + 3 * b) % 5 < mask || (mask == 2 && a == 2) {
                    continue;
                }
                let y = (a * a * a + a * b) % P;
                let fault = usize::from(mask > 0 && a == 3 && b % 2 == 1);
                let guess = (y + 2 * a + 3 * b + fault) % P;
                cells
                    .record(a as u32, b as u32, guess as u32, y as u32)
                    .unwrap();
                *e = Some(guess as i32 - y as i32);
            }
        }
        let report = cells.report();
        assert_eq!(report, enumerate(&errors));
        if mask == 0 {
            assert_eq!(report["rectangles"], json!({"consistent":441,"total":441}));
            assert_eq!(report["row_pairs"], json!({"consistent":0,"total":147}));
        } else if mask == 1 {
            assert!(
                report["rectangles"]["consistent"].as_u64()
                    < report["rectangles"]["total"].as_u64()
            );
        }
    }
}

#[test]
fn invalid_cells_do_not_change_the_split() {
    let mut cells = Residuals::<3>::new();
    cells.record(1, 1, 0, 2).unwrap();
    cells.record(1, 2, 1, 0).unwrap();
    let before = cells.report();
    assert!(cells.record(1, 1, 2, 2).is_err());
    for args in [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]] {
        let [a, b, pred, y] = args;
        assert!(cells.record(a, b, pred, y).is_err());
    }
    assert_eq!(before, cells.report());
    assert_eq!(before["row_pairs"], json!({"consistent":1,"total":1}));
}
