//! Matched removals of the triple transport read, scored on the frozen task splits.
use super::*;

pub fn run(name: &str, seed: u32, out: &Path, probes: ProbeSet) -> Result<()> {
    if !architecture::TRIPLE_TRANSPORT {
        return Err("read removal requires triple transport".into());
    }
    let (row, task, fingerprints) = prepare_evaluation(name, probes)?;
    let cfg = task_config(&row, &task, false)?;
    let dir = out.join(name).join(format!("seed{seed}"));
    let (saved, original) = embedding::verified_weights(&row, seed, &cfg, &dir)?;
    let mut engine = Engine::new(cfg.clone(), &original)?;
    let score = |engine: &mut Engine, weights: &[f32]| -> Result<Value> {
        engine.parameters.write(weights)?;
        let mut splits = BTreeMap::new();
        for (split, examples) in &task.splits {
            if split == &task.selection || split == "test" || split.starts_with("test/") {
                let counts = metrics::evaluate(engine, examples, true)?;
                splits.insert(
                    split,
                    json!({"correct":counts["exact"],"total":examples.len()}),
                );
            }
        }
        Ok(json!(splits))
    };
    let baseline = score(&mut engine, &original)?;
    if baseline[&task.selection]["correct"] != saved["observation"]["correct"]
        || baseline[&task.selection]["total"] != saved["observation"]["total"]
    {
        return Err("triple-transport baseline differs from the saved held-out observation".into());
    }
    let mut results = BTreeMap::new();
    let mut weights = original.clone();
    for (name, removed) in [
        ("without_context", &[0][..]),
        ("without_direct", &[1][..]),
        ("without_constant", &[2][..]),
        ("without_transport", &[3][..]),
        ("context_only", &[1, 2, 3][..]),
        ("structural_only", &[0, 1][..]),
    ] {
        weights.copy_from_slice(&original);
        engine.disable_slot_reads(&mut weights, removed)?;
        if cfg.head.has_triple_phase() && matches!(name, "context_only" | "structural_only") {
            weights[crate::slot_memory::phase_outputs(&cfg)].fill(0.);
        }
        results.insert(name, score(&mut engine, &weights)?);
    }
    if cfg.head == head::Head::BipartiteTriple {
        let mut shared_cfg = cfg.clone();
        shared_cfg.head = head::Head::TripleTransport;
        let mut shared = Engine::new(shared_cfg, &original)?;
        results.insert("shared_roles", score(&mut shared, &original)?);
    }
    if let Some(read) = cfg.head.peer_read() {
        if read.pairs.is_some() {
            for (name, pairs) in [
                ("without_pair_classes", None),
                (
                    "unique_pair_candidate",
                    Some(crate::slot_orbits::PairRead::Unique),
                ),
            ] {
                let mut control = cfg.clone();
                control.head = head::Head::PeerTriplePhase(head::PeerRead { pairs, ..read });
                results.insert(
                    name,
                    score(&mut Engine::new(control, &original)?, &original)?,
                );
            }
        }
        let mut control = cfg.clone();
        control.head = head::Head::PeerTriplePhase(head::PeerRead {
            search: None,
            ..read
        });
        let mut plain = Engine::new(control, &original)?;
        results.insert("without_peers", score(&mut plain, &original)?);
    }
    if let Some(
        read @ head::PeerRead {
            search: Some(head::PeerSearch::Iterated(None)),
            ..
        },
    ) = cfg.head.peer_read()
    {
        let mut single_cfg = cfg.clone();
        single_cfg.head = head::Head::PeerTriplePhase(head::PeerRead {
            search: Some(head::PeerSearch::Witnesses),
            ..read
        });
        let mut single = Engine::new(single_cfg, &original)?;
        results.insert("one_round", score(&mut single, &original)?);
    }
    if let Some(
        read @ head::PeerRead {
            search: Some(head::PeerSearch::Iterated(Some(roles))),
            ..
        },
    ) = cfg.head.peer_read()
    {
        use crate::head::PeerSearch;
        use crate::slot_orbits::RoleEvidence;
        let mut controls = vec![
            (
                "without_role_proposals",
                Some(RoleEvidence {
                    role_proposals: false,
                    ..roles
                }),
            ),
            (
                "without_relative_overlap",
                Some(RoleEvidence {
                    relative_overlap: false,
                    ..roles
                }),
            ),
            ("without_role_evidence", None),
        ];
        if roles.graph_layers {
            controls.push((
                "without_graph_layers",
                Some(RoleEvidence {
                    graph_layers: false,
                    ..roles
                }),
            ));
        }
        for (name, terms) in controls {
            let mut control = cfg.clone();
            control.head = head::Head::PeerTriplePhase(head::PeerRead {
                search: Some(PeerSearch::Iterated(terms)),
                ..read
            });
            results.insert(
                name,
                score(&mut Engine::new(control, &original)?, &original)?,
            );
        }
    }
    let mut report = json!({"case":name,"seed":seed,"model":MODEL,"weights_sha256":saved["weights_sha256"],
        "kernite_version":trained_with(&saved)?,"scored_with_kernite":KERNITE_VERSION,
        "step":saved["observation"]["step"],"selection":task.selection,"baseline":baseline,"ablations":results});
    if !fingerprints.is_empty() {
        report["probe_fingerprints"] = json!(fingerprints);
    }
    println!("{report}");
    Ok(())
}
