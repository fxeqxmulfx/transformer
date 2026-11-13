use super::*;
#[test]
fn complete_binding_retains_all_modes_and_rejects_mutated_evidence() -> Result<()> {
    let request = Request {
        discovery: crate::tests::pair_request(),
        training: training::policies(),
    };
    let report = run(request.clone())?;
    replay(&request, &report)?;
    assert_eq!(report.endpoints.len(), 3);
    for endpoint in &report.endpoints {
        assert!(endpoint.error.is_none());
        assert_eq!(endpoint.audit.as_ref().unwrap().train_errors, 0);
    }
    let saved = serde_json::to_vec(&report)?;
    assert_eq!(serde_json::from_slice::<Report>(&saved)?, report);
    for i in 0..5 {
        let mut bad = report.clone();
        match i {
            0 => bad.endpoints[0].audit.as_mut().unwrap().train_ce += 0.01,
            1 => {
                bad.endpoints.pop();
            }
            2 => bad.endpoints[1].error = Some("invented refusal".into()),
            3 => bad.buffer_preflight_bytes += 1,
            _ => bad.unseen_identification = true,
        }
        assert!(check(&request, &bad).is_err());
    }
    Ok(())
}
#[test]
fn full_batch_work_cap_preserves_mini_and_forward_and_discovery_refusals_replay() -> Result<()> {
    let mut request = Request {
        discovery: crate::tests::pair_request(),
        training: training::policies(),
    };
    request.discovery.input.vocabulary = 1024;
    request.training[0].updates = 4096;
    request.training[1].updates = 1;
    let out = run(request.clone())?;
    check(&request, &out)?;
    assert!(out.endpoints[0].training.is_none());
    assert!(
        out.endpoints[0]
            .error
            .as_ref()
            .unwrap()
            .contains("scalar-work forecast")
    );
    assert!(out.endpoints[1].error.is_none());
    assert!(out.endpoints[2].error.is_none());
    assert_eq!(
        out.endpoints[2]
            .training
            .as_ref()
            .unwrap()
            .work
            .gradient_examples,
        0
    );
    let mut bad = out;
    bad.endpoints[0].error = None;
    assert!(check(&request, &bad).is_err());
    request.discovery.policy.visits = 0;
    let out = run(request.clone())?;
    assert!(out.initialization.is_none());
    assert!(out.initialization_error.is_some());
    assert!(out.discovery.reason.is_some());
    assert!(out.endpoints.is_empty());
    replay(&request, &out)?;
    Ok(())
}
