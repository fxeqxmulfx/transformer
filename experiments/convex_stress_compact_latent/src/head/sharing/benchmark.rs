//! Bounded synthetic timing only. It neither reads original data nor credits
//! task accuracy. Compare the same complete tensor with and without its plan.
use super::*;
use std::{hint::black_box, path::PathBuf, process::Command, time::Instant};

fn resources(guard: &legacy::resources::Guard) {
    guard.check().unwrap();
    let output = Command::new("ps")
        .args(["-e", "-o", "rss="])
        .output()
        .unwrap();
    assert!(output.status.success());
    let rss: u64 = std::str::from_utf8(&output.stdout)
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse::<u64>().unwrap() * 1024)
        .sum();
    assert!(
        rss + 1024u64.pow(3) < 110_000_000_000,
        "aggregate RAM cannot admit the bounded benchmark plus 1 GiB reserve"
    );
}

#[test]
#[ignore = "explicit bounded benchmark requires both filesystem paths and process admission"]
fn benchmark_complete_column_sharing() {
    let workspace = PathBuf::from(std::env::var_os("SHARED_COLUMNS_WORKSPACE").unwrap());
    let storage = PathBuf::from(std::env::var_os("SHARED_COLUMNS_STORAGE").unwrap());
    let guard = legacy::resources::Guard::new(&workspace, &storage, 1024 * 1024).unwrap();
    resources(&guard);
    for programs in [128, 512, 2048] {
        for classes in [97, 193] {
            resources(&guard);
            let sh = Shape {
                programs,
                states: 8,
                classes,
            };
            let mut labels = vec![None; sh.rows()];
            for p in 0..programs {
                labels[p * 8] = Some(0);
            }
            let domain = Domain::from_labels(sh, labels, 1., 8., &mut work()).unwrap();
            let q = Parameters::initial(&domain, 8., LIMIT, &mut work()).unwrap();
            let mut head = Head::build(&domain, &q, LIMIT, &mut work()).unwrap();
            let queries = (0..16)
                .map(|i| (0..programs).map(|p| (p + i) % 4).collect())
                .collect::<Vec<Vec<usize>>>();
            // All words and class CEs, not merely the winning label, match.
            same_queries(&mut head, &queries);
            let physical = head.evaluated_columns(&queries[0]).unwrap();
            let logical = head.query(&queries[0]).unwrap();
            assert_eq!(physical, programs * 2);
            assert_eq!(logical, programs * classes);
            let (mut full_times, mut shared_times) = (Vec::new(), Vec::new());
            for round in 0..7 {
                // Alternate order so warm caches are not assigned to one path.
                for shared in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    resources(&guard);
                    let plan = if shared { None } else { head.sharing.take() };
                    let mut w = work();
                    let start = Instant::now();
                    for states in &queries {
                        black_box(head.logits(black_box(states), &mut w).unwrap());
                    }
                    let elapsed = start.elapsed().as_nanos() as u64;
                    if shared {
                        shared_times.push(elapsed);
                    } else {
                        full_times.push(elapsed);
                    }
                    if let Some(plan) = plan {
                        head.sharing = Some(plan);
                    }
                }
            }
            full_times.sort_unstable();
            shared_times.sort_unstable();
            println!(
                concat!(
                    "{{\"programs\":{},\"states\":8,\"classes\":{},\"queries\":16,",
                    "\"rounds\":7,\"logical_columns_per_query\":{},\"evaluated_columns_per_query\":{},",
                    "\"full_median_ns\":{},\"shared_median_ns\":{},\"speedup\":{},",
                    "\"head_bytes\":{},\"all_bits_ce_and_work_equal\":true}}"
                ),
                programs,
                classes,
                logical,
                physical,
                full_times[3],
                shared_times[3],
                full_times[3] as f64 / shared_times[3] as f64,
                head.allocation_bytes()
            );
        }
    }
    resources(&guard);
}
