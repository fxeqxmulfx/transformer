use super::*;
use std::{collections::BTreeMap, time::Instant};

fn fixture(rejected: bool) -> Result<Inputs> {
    let labels = [
        3, 0, 4, 1, 2, 0, 4, 1, 2, 3, 4, 1, 2, 3, 0, 1, 2, 3, 0, 4, 2, 3, 0, 4, 1,
    ];
    let example = |cell: usize| data::Example {
        task: "categorical-fixture".into(),
        tokens: vec![(cell / 5) as i64, (cell % 5) as i64],
        targets: vec![-100, if rejected { 0 } else { labels[cell] }],
        changes: vec![false, true],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    };
    let train: Vec<_> = (0..25).filter(|cell| *cell != 11).map(example).collect();
    let raw: Vec<_> = train
        .iter()
        .map(|e| Sample {
            cell: e.tokens[1] as usize + 5 * e.tokens[0] as usize,
            label: e.targets[1] as usize,
        })
        .collect();
    let discovery = coding::discover(5, &raw)?;
    let coding = discovery.coding.clone();
    let samples = if let Some(code) = &coding {
        code.samples(&raw)?
    } else {
        vec![]
    };
    Ok(Inputs {
        task: data::Task {
            vocab: 5,
            selection: "validation".into(),
            splits: BTreeMap::from([
                ("train".into(), train),
                ("validation".into(), vec![example(11)]),
            ]),
        },
        row: json!({"case":{"batch":5,"observe":5},"data_fingerprints":{"fixture":"categorical-withheld-v1"}}),
        coding,
        discovery,
        samples,
        explicit_bytes: memory_bound(5)?,
    })
}
fn local_inventory(name: &str, rejected: bool) -> Result<Value> {
    Ok(
        json!({"source_fingerprint":fingerprint(),"protocol":protocol()?,"producer_commit":"0000000000000000000000000000000000000000000","cases":cases()?,
        "tasks":[task_record(name,&fixture(rejected)?)]}),
    )
}

#[test]
fn runner_saved_replay_and_independent_audit_cover_all_modes() -> Result<()> {
    let root = std::env::temp_dir().join(format!("cm12-endpoint-{}", std::process::id()));
    fs::create_dir(&root)?;
    let name = "arithmetic-perm-add";
    let inv = local_inventory(name, false)?;
    write_new(&root.join("inventory.json"), &inv)?;
    for arm in ["full_batch", "minibatch", "forward_only"] {
        let c = Case {
            task: name.into(),
            arm: arm.into(),
            coding: "cycle".into(),
            budget: "rank_one".into(),
        };
        run::execute(
            &root,
            c.clone(),
            inv.clone(),
            protocol()?,
            fixture(false)?,
            Instant::now(),
        )?;
        let seal = replay::with_inputs(&root, &c.id(), fixture(false)?)?;
        assert_eq!(seal["audit"]["raw_coordinate_audit"], true);
        let report = read(&root.join(c.id()).join("result.json"))?;
        assert_eq!(report["metrics"]["validation"]["exact"], 1);
        if arm == "forward_only" {
            assert_eq!(report["training_gradient_evaluations"], 0);
        }
        write_new(&root.join(format!("{}.replay.json", c.id())), &seal)?;
        assert!(
            run::execute(
                &root,
                c.clone(),
                inv.clone(),
                protocol()?,
                fixture(false)?,
                Instant::now()
            )
            .is_err()
        );
        let path = root.join(c.id()).join("model.json");
        let mut model: crate::Model = serde_json::from_slice(&fs::read(&path)?)?;
        model.coding.encode.swap(0, 1);
        let changed = serde_json::to_vec(&model)?;
        fs::write(path, &changed)?;
        let mut tampered = report;
        tampered["model_sha256"] = hash(&changed).into();
        fs::write(
            root.join(c.id()).join("result.json"),
            serde_json::to_vec(&tampered)?,
        )?;
        assert!(replay::with_inputs(&root, &c.id(), fixture(false)?).is_err());
    }
    assert!(summarize(&root).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn rejected_proposals_save_and_replay_without_a_model() -> Result<()> {
    let root = std::env::temp_dir().join(format!("cm12-rejected-{}", std::process::id()));
    fs::create_dir(&root)?;
    let c = Case {
        task: "arithmetic-perm-add".into(),
        arm: "forward_only".into(),
        coding: "cycle".into(),
        budget: "rank_one".into(),
    };
    let inv = local_inventory(&c.task, true)?;
    write_new(&root.join("inventory.json"), &inv)?;
    run::execute(
        &root,
        c.clone(),
        inv,
        protocol()?,
        fixture(true)?,
        Instant::now(),
    )?;
    let seal = replay::with_inputs(&root, &c.id(), fixture(true)?)?;
    assert_eq!(seal["audit"]["rejected_proposal_reproduced"], true);
    assert!(!root.join(c.id()).join("model.json").exists());
    let mut changed = fixture(true)?;
    changed.discovery.reason = "invented".into();
    assert!(replay::with_inputs(&root, &c.id(), changed).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
