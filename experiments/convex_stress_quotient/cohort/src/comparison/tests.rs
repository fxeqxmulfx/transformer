use super::*;
fn fixture() -> (Value, Vec<String>, Vec<Offer>) {
    let tasks: Vec<_> = (0..35).map(|i| format!("fixture-{i}")).collect();
    let arms:Vec<_>=POLICIES.iter().enumerate().map(|(p,policy)| {
        let choices:Vec<_>=tasks.iter().flat_map(|task|MODES.iter().map(move |mode|
            json!({"task":task,"training_mode":mode,"status":"selected", "reference":format!("parent:{policy}:{task}:{mode}"),
                "train_ce":if p==0{1.0}else{0.2},"metrics":{"heldout":0.5}}))).collect();
        json!({"initializer":policy,"choices":choices})
    }).collect();
    let parent = json!({"arms":arms,"depth_outcomes":vec![json!({});336],"transformer_endpoints":vec![json!({});35],
        "all_parent_costs_charged_once":true,"choose_between_initializers":false});
    let offers = [
        training::Mode::Full,
        training::Mode::Mini,
        training::Mode::Forward,
    ]
    .iter()
    .map(|&mode| Offer {
        task: tasks[0].clone(),
        mode,
        prime: 3,
        train_ce: 0.25,
        model_sha256: "a".repeat(64),
        stage_sha256: "b".repeat(64),
    })
    .collect();
    (parent, tasks, offers)
}
#[test]
fn both105_arms_compare_their_own_parents_and_preserve_missing_models() -> Result<()> {
    let (mut parent, tasks, offers) = fixture();
    parent["arms"][0]["choices"][5] = json!({"task":tasks[1],"training_mode":"forward_only","status":"no_model","reference":null});
    let out = select(&parent, &offers, &tasks)?;
    assert_eq!(out["arms"][0]["choices"].as_array().unwrap().len(), 105);
    assert_eq!(out["arms"][1]["choices"].as_array().unwrap().len(), 105);
    for (i, offer) in offers.iter().enumerate() {
        assert_eq!(out["arms"][0]["choices"][i]["selected"], offer.reference());
        assert_eq!(
            out["arms"][1]["choices"][i]["selected"],
            parent["arms"][1]["choices"][i]["reference"]
        );
    }
    assert!(out["arms"][0]["choices"][5]["selected"].is_null());
    assert_eq!(out["select_between_initializers"], false);
    Ok(())
}
#[test]
fn earlier_ce_ties_ignore_every_heldout_or_certificate_field() -> Result<()> {
    let (mut parent, tasks, mut offers) = fixture();
    for o in &mut offers {
        o.train_ce = 1.;
    }
    let baseline = select(&parent, &offers, &tasks)?;
    for arm in parent["arms"].as_array_mut().unwrap() {
        for row in arm["choices"].as_array_mut().unwrap() {
            row["metrics"] = json!({"heldout":1.});
            row["primary_pass"] = json!(true);
            row["certificate"] = json!(true);
        }
    }
    assert_eq!(select(&parent, &offers, &tasks)?, baseline);
    for mode in 0..3 {
        assert_eq!(
            baseline["arms"][0]["choices"][mode]["selected"],
            parent["arms"][0]["choices"][mode]["reference"]
        );
    }
    offers[0].train_ce = 0.1;
    let mut late = offers[0].clone();
    late.prime = 5;
    offers.insert(0, late);
    assert_eq!(
        select(&parent, &offers, &tasks)?["arms"][0]["choices"][0]["selected"],
        offers[1].reference()
    );
    Ok(())
}
#[test]
fn complete_scope_and_actual_score_shapes_are_required() {
    let (parent, tasks, offers) = fixture();
    for mutation in 0..6 {
        let mut bad = parent.clone();
        match mutation {
            0 => {
                bad["arms"][0]["choices"].as_array_mut().unwrap().pop();
            }
            1 => bad["arms"][0]["choices"][0] = bad["arms"][0]["choices"][1].clone(),
            2 => bad["arms"][1]["initializer"] = json!("minimum_ce"),
            3 => {
                bad["depth_outcomes"].as_array_mut().unwrap().pop();
            }
            4 => {
                bad["transformer_endpoints"].as_array_mut().unwrap().pop();
            }
            _ => bad["arms"][0]["choices"][0]["train_ce"] = json!(-1.),
        }
        assert!(select(&bad, &offers, &tasks).is_err());
    }
    for mutation in 0..5 {
        let mut bad = offers.clone();
        match mutation {
            0 => bad.push(bad[0].clone()),
            1 => bad[0].prime = 4,
            2 => bad[0].train_ce = f64::NAN,
            3 => bad[0].stage_sha256 = "bad".into(),
            _ => bad[0].task = "unknown".into(),
        }
        assert!(select(&parent, &bad, &tasks).is_err());
    }
}
