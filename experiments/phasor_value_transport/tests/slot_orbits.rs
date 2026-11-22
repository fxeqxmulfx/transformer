#[path = "../src/slot_orbits.rs"]
mod slot_orbits;
use slot_orbits::{ABSENT, Reads, TripleGraph, fill_from_peers};

fn complete(slots: usize, bank: &[u32], priority: &[usize]) -> Reads {
    TripleGraph::default().complete(slots, bank, priority)
}

#[test]
fn transported_slots_follow_arbitrary_renamings() {
    let k = 7;
    let mut bank = Vec::new();
    for (a, b) in [(1, 1), (3, 0)] {
        bank.extend((0..k).map(|x| ((a * x + b) % k) as u32));
    }
    bank.extend([3, 5, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT]);
    let priority: Vec<_> = (0..k).collect();
    let read = complete(k, &bank, &priority);
    for x in 2..k {
        assert_eq!(read.transport[2 * k + x], ((2 * x + 3) % k) as u32);
    }
    let rename = [3, 0, 6, 4, 1, 5, 2];
    let mut permuted = vec![ABSENT; bank.len()];
    for (i, &y) in bank.iter().enumerate() {
        if y != ABSENT {
            permuted[i / k * k + rename[i % k]] = rename[y as usize] as u32;
        }
    }
    let other = complete(k, &permuted, &rename);
    for (i, &y) in read.transport.iter().enumerate() {
        let expected = if y == ABSENT {
            ABSENT
        } else {
            rename[y as usize] as u32
        };
        assert_eq!(other.transport[i / k * k + rename[i % k]], expected);
    }
}

#[test]
fn ambiguous_or_unobserved_relations_abstain() {
    let k = 5;
    let priority: Vec<_> = (0..k).collect();
    let mut bank = vec![1, 2, 3, 4, 0, 1, 0, 2, 3, 4, 0, 1, ABSENT, ABSENT, ABSENT];
    // A cycle and a transposition generate every permutation. No choice of a
    // third distinct point is distinguished by the two fixed anchors.
    let read = complete(k, &bank, &priority);
    assert!(read.transport[2 * k + 2..].iter().all(|&v| v == ABSENT));
    bank.fill(ABSENT);
    let empty = complete(k, &bank, &priority);
    assert!(
        empty
            .constant
            .iter()
            .chain(&empty.transport)
            .all(|&v| v == ABSENT)
    );
}

#[test]
fn constant_consensus_excludes_the_current_query() {
    let bank = [ABSENT, 2, 2, ABSENT, ABSENT, 2, 2, 2, 1, 1, 2, ABSENT];
    let result = complete(4, &bank, &[2, 0, 3, 1]);
    assert_eq!(
        result.constant,
        [
            2, ABSENT, ABSENT, 2, 2, 2, 2, 2, ABSENT, ABSENT, ABSENT, ABSENT
        ]
    );
    assert!(result.transport.iter().all(|&v| v == ABSENT));
}

#[test]
fn growing_and_replaced_banks_match_fresh_components() {
    for separate_images in [false, true] {
        compare_updates(separate_images);
    }
}

fn compare_updates(separate_images: bool) {
    let mut cache = TripleGraph::new(separate_images);
    for k in [7_usize, 5, 7] {
        let mut bank = vec![ABSENT; k * k];
        let mut priority: Vec<_> = (0..k).collect();
        let compare = |cache: &mut TripleGraph, bank: &[u32], priority: &[usize]| {
            assert_eq!(
                cache.complete(k, bank, priority),
                TripleGraph::new(separate_images).complete(k, bank, priority)
            );
        };
        compare(&mut cache, &bank, &priority);
        for x in 0..k {
            // The first two rows grow into injective maps, the third supplies only anchors.
            bank[x] = ((x + 1) % k) as u32;
            bank[k + x] = ((3 * x + 1) % k) as u32;
            if x < 2 {
                bank[2 * k + x] = ((2 * x + 3) % k) as u32;
            }
            compare(&mut cache, &bank, &priority);
            priority.rotate_left(1);
            compare(&mut cache, &bank, &priority);
        }
        let mut rewritten = bank.clone();
        rewritten[k] = rewritten[k + 1]; // A previously eligible row becomes noninjective.
        compare(&mut cache, &rewritten, &priority);
        compare(&mut cache, &bank, &priority);
        rewritten[..k].fill(0); // Switch a graph row to a constant read.
        compare(&mut cache, &rewritten, &priority);
        rewritten.fill(ABSENT);
        compare(&mut cache, &rewritten, &priority);
        for row in bank.chunks_exact_mut(k) {
            row.reverse();
        }
        compare(&mut cache, &bank, &priority);
    }
}

#[test]
fn completions_allow_independent_renaming_of_images() {
    let k = 7;
    let permutation = [2, 5, 0, 6, 3, 1, 4];
    let priority: Vec<_> = (0..k).collect();
    let mut bank = Vec::new();
    for shift in [0, 1, 4] {
        bank.extend((0..k).map(|x| ((x + shift) % k) as u32));
    }
    bank.extend([2, 3, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT]);
    let original = TripleGraph::new(true).complete(k, &bank, &priority);
    for (x, &y) in original.transport[3 * k..].iter().enumerate().skip(2) {
        assert_eq!(y, ((x + 2) % k) as u32);
    }
    for y in &mut bank {
        if *y != ABSENT {
            *y = permutation[*y as usize];
        }
    }
    let renamed = TripleGraph::new(true).complete(k, &bank, &priority);
    for (&a, &b) in original.transport.iter().zip(&renamed.transport) {
        assert_eq!(
            b,
            if a == ABSENT {
                ABSENT
            } else {
                permutation[a as usize]
            }
        );
    }
}
#[test]
fn repeated_keys_and_renamed_outputs_recover_held_out_cells() {
    let k = 17;
    let rename = |x: usize| ((7 * x + 3) % k) as u32;
    let target = |i: usize, j: usize| rename((3 * i + j / 2) % k);
    let mut bank = vec![ABSENT; k * k];
    let mut read = bank.clone();
    for i in 0..k {
        for j in 0..k {
            if (7 * i + 5 * j + i * j) % 13 < 9 {
                bank[i * k + j] = target(i, j);
            }
        }
    }
    fill_from_peers(k, &bank, &mut read);
    let mut recovered = 0;
    for i in 0..k {
        for j in 0..k {
            if bank[i * k + j] == ABSENT && read[i * k + j] != ABSENT {
                assert_eq!(read[i * k + j], target(i, j));
                recovered += 1;
            }
        }
    }
    assert!(recovered >= 65, "{recovered} held-out cells");
    let (mut permuted, mut expected) = (vec![ABSENT; k * k], vec![ABSENT; k * k]);
    let slot = |x: u32| {
        if x == ABSENT {
            ABSENT
        } else {
            (11 * x + 6) % k as u32
        }
    };
    for i in 0..k {
        for j in 0..k {
            let dest = ((5 * i + 3) % k) * k + (9 * j + 2) % k;
            permuted[dest] = slot(bank[i * k + j]);
            expected[dest] = slot(read[i * k + j]);
        }
    }
    let mut changed = vec![ABSENT; k * k];
    fill_from_peers(k, &permuted, &mut changed);
    assert_eq!(changed, expected);
}

#[test]
fn insufficient_peer_evidence_preserves_existing_reads() {
    let mut bank = vec![ABSENT; 25];
    bank[..4].copy_from_slice(&[0, 1, 2, 3]);
    bank[5..8].copy_from_slice(&[1, 2, 3]);
    let mut read = vec![ABSENT; 25];
    read[8] = 4;
    let old = read.clone();
    fill_from_peers(5, &bank, &mut read);
    assert_eq!(read, old);
}
