//! Conservative preallocation/finite-work guards, including rejected proposals.
//! All queries/classes count; no task-prime or canonical-label size shortcut.
use crate::*;

pub(crate) fn enforce_limits(buffer: u128, total: u128) -> Result<()> {
    if buffer > 56 * 1024 * 1024 * 1024 || total > WORK_CAP as u128 {
        return Err(
            "margin context exceeds56GiB memory or8T worst-case scalar-work forecast".into(),
        );
    }
    Ok(())
}

pub fn forecast(parent: &parent::Model) -> Result<Bounds> {
    let source = parent.source();
    let n = parent.records().len();
    let ep = parent.endpoint();
    let (j, g, s, a, r, v) = (
        parent.bank().len() as u128 + 2,
        GATE_COUNT as u128,
        parent.bounds().max_states as u128,
        parent.bounds().max_active_classes as u128,
        n as u128,
        source.vocabulary as u128,
    );
    let epochs = match ep.mode {
        Mode::Full => FULL_STEPS,
        Mode::Mini => MINI_EPOCHS,
        Mode::Forward => 0,
    };
    let batches = if ep.mode == Mode::Mini {
        n.div_ceil(ep.mixture_batch)
    } else {
        1
    };
    let max_gradient_batches = epochs * batches;
    if max_gradient_batches > 4096 {
        return Err("guarded gradient batch cap exceeded".into());
    }
    let steps = max_gradient_batches as u128;
    let cells = (source.n as u128).pow(2);
    let (mut q, mut qs, mut qs2) = (0u128, 0u128, 0u128);
    // Read-only scalar scan precedes every new cache allocation.
    for cell in 0..source.n * source.n {
        if parent
            .parent()
            .parent()
            .context()
            .neighbors(cell)?
            .is_some()
        {
            let states = source.blocks[source.location(cell).0].states() as u128;
            q += 1;
            qs += states;
            qs2 += states * states;
        }
    }
    let all_s: u128 = source.blocks.iter().map(|b| b.states() as u128).sum();
    let all_s2: u128 = source
        .blocks
        .iter()
        .map(|b| (b.states() as u128).pow(2))
        .sum();
    // Primitive caches plus Vec overhead, several live score states, JSON/digest
    // temporaries, actual features and reconstructed saved-model comparisons.
    let buffer = parent.bounds().buffer_bytes as u128
        + 2 * (q * (j * (17 * 24 + 48 + 8 * v) + 160 * v + 256)
            + 160 * qs
            + 16 * all_s * v
            + 128 * r * j * a
            + 128 * cells
            + 256 * (steps + 1) * g * j);
    let inherited = parent.bounds().total_scalar_work as u128;
    let construction = 16 * all_s2 * v
        + 16 * qs2
        + 16 * qs * v
        + 8 * q * j * 17 * v
        + 64 * (r + q) * (3 * s * (1 + parent.bounds().max_states.ilog2() as u128) + j * 17 + a)
        + 64 * (cells * (a + 1) + all_s * (a + 1) + r * j * a);
    let rounds = (steps + 1) * (BACKTRACKS as u128 + 1) + 3;
    let optimization = rounds * (16 * qs * v + 16 * qs2 + 8 * q * j * 17 + 32 * q * v)
        + 32 * (j + steps) * q * v
        + 64 * (j + 4 + (epochs as u128 + 1) * (LINE_SEARCH.len() + BACKTRACKS + 3) as u128)
            * r
            * j
            * a;
    let total = inherited + construction + optimization;
    enforce_limits(buffer, total)?;
    Ok(Bounds {
        parent_buffer_bytes: parent.bounds().buffer_bytes,
        buffer_bytes: buffer as u64,
        inherited_scalar_work: inherited as u64,
        construction_scalar_work: construction as u64,
        optimization_scalar_work: optimization as u64,
        total_scalar_work: total as u64,
        groups: g as usize,
        bank_size: j as usize,
        guarded_cells: q as usize,
        guard_classes: v as usize,
        max_active_classes: a as usize,
        max_states: s as usize,
        max_gradient_batches,
        max_guard_query_visits: (rounds * q) as u64,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    parent: serde_json::Value,
    binding: Binding,
    profiles: Vec<Profile>,
    gates: Vec<usize>,
    radius: f64,
    retention: f64,
    selection: Selection,
    rows: Vec<Option<FeatureRow>>,
    certificate: Certificate,
    bounds: Bounds,
}

impl Model {
    pub fn new(parent: parent::Model, expected_parent_sha256: &str) -> Result<Self> {
        let bounds = forecast(&parent)?;
        let parent_sha256 = digest(&parent)?;
        if parent_sha256 != expected_parent_sha256 {
            return Err("CM46 reference differs from external binding".into());
        }
        let profiles = gates::profiles(parent.source());
        let gates = gates::build(parent.source(), &profiles);
        let guard = guard::Guard::new(&parent, &gates)?;
        if guard.queries.len() != bounds.guarded_cells {
            return Err("guard domain differs from forecast".into());
        }
        let (selection, certificate) = selection::select(&parent, &gates, &guard)?;
        let rows = model::rows(&parent, &selection.weights, &gates)?;
        for r in parent.records() {
            if rows[r.cell].is_some() {
                return Err("guard changed an observed source row".into());
            }
        }
        let binding = Binding {
            parent_sha256,
            protocol_sha256: digest(&serde_json::from_str::<serde_json::Value>(PROTOCOL)?)?,
        };
        let mut model = Self {
            parent,
            binding,
            profiles,
            gates,
            radius: RADIUS,
            retention: RETENTION,
            selection,
            rows,
            certificate,
            bounds,
        };
        // Verify the final stored inference rows, independently of the proposal
        // cache. Bind exactly the scores returned by the public prediction path.
        let mut materialized = Vec::with_capacity(guard.queries.len());
        for q in &guard.queries {
            let z = model.logits_cell(q.cell)?;
            if !margins_hold(&q.reference, &z) {
                return Err("materialized inference changed raw margins".into());
            }
            model.selection.work.guard_query_visits += 1;
            model.selection.work.guard_class_visits += z.len();
            let s = model.source().blocks[model.source().location(q.cell).0].states();
            let residual = model.rows[q.cell]
                .as_ref()
                .is_some_and(|r| r.residual.iter().any(|p| p.re != 0. || p.im != 0.));
            model.selection.work.actual_head_terms += (1 + usize::from(residual)) * s * z.len();
            model.selection.work.feature_scalar_work_bound += 8 * s * s;
            materialized.push(z);
        }
        if digest(&materialized)? != model.certificate.final_scores_sha256 {
            return Err("materialized scores differ from checked endpoint".into());
        }
        if model.selection.work.guard_query_visits as u64 > model.bounds.max_guard_query_visits {
            return Err("actual guard visits exceed forecast".into());
        }
        Ok(model)
    }

    pub fn load(value: serde_json::Value, expected: &Binding) -> Result<Self> {
        let saved: Saved = serde_json::from_value(value)?;
        if &saved.binding != expected {
            return Err("saved guard binding differs".into());
        }
        let parent_binding = serde_json::from_value(saved.parent["binding"].clone())?;
        let parent = parent::Model::load(saved.parent, &parent_binding)?;
        let actual = Self::new(parent, &expected.parent_sha256)?;
        if &actual.binding != expected
            || actual.profiles != saved.profiles
            || actual.gates != saved.gates
            || actual.radius != saved.radius
            || actual.retention != saved.retention
            || actual.selection != saved.selection
            || actual.rows != saved.rows
            || actual.certificate != saved.certificate
            || actual.bounds != saved.bounds
        {
            return Err("saved guard architecture,coefficients,margins or costs changed".into());
        }
        Ok(actual)
    }
}
