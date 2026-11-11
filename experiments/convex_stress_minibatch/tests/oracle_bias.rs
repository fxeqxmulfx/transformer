//! Diagnostic counterexample: exact batch oracles need not approach the full-data
//! optimum. This tests the frozen algorithm, without changing its training code.
use convex_stress_minibatch::{
    channel::argmax, features::Adapter, input::Observation, stochastic::Stochastic,
};

#[test]
fn singleton_oracles_can_remain_suboptimal_on_representable_data() {
    let adapter = Adapter {
        vocab: 3,
        context: 2,
    };
    let prefixes = [vec![0, 2], vec![1, 2]];
    let samples: Vec<_> = prefixes
        .iter()
        .enumerate()
        .map(|(label, prefix)| {
            let (query, items) = adapter.encode(prefix);
            Observation {
                query,
                items,
                label,
            }
        })
        .collect();

    // The value at lag one identifies both labels perfectly. With both examples
    // in a batch, its key is the first separating one and a single update fits.
    let mut complete = Stochastic::new(adapter);
    complete.update_batch(&samples.iter().collect::<Vec<_>>());
    for (label, prefix) in prefixes.iter().enumerate() {
        assert_eq!(argmax(&complete.predict(prefix)), label);
    }
    let (optimal_loss, optimal_gap) = complete.objective(&samples.iter().collect::<Vec<_>>());
    assert!(optimal_gap.abs() < 1e-14);

    // On a single example every key can assign the same best class at its one
    // observed value, so all key costs tie. The smallest key is lag zero, which
    // is constant (token 2) on this dataset. Every selected vertex uses it.
    let mut singleton = Stochastic::new(adapter);
    for update in 0..1000 {
        singleton.update_batch(&[&samples[update % 2]]);
        assert_eq!(singleton.snapshot().queries[&2].routes, [(0, 1.)].into());
    }
    assert_eq!(
        singleton.predict(&prefixes[0]),
        singleton.predict(&prefixes[1])
    );
    let correct = prefixes
        .iter()
        .enumerate()
        .filter(|(label, prefix)| argmax(&singleton.predict(prefix)) == *label)
        .count();
    assert_eq!(correct, 1);
    let (loss, gap) = singleton.objective(&samples.iter().collect::<Vec<_>>());
    assert!(loss > optimal_loss + 0.4);
    assert!(gap > 0.49);
    assert!(singleton.snapshot().feasibility_error().unwrap() < 1e-14);
}
