//! Rebuild every derived tensor and cost on load. External source and record
//! hashes bind the learned decoder and the original confidence-selection data.
use crate::*;
use std::collections::BTreeSet;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    source: tagged::Model,
    records: Vec<records::Record>,
    endpoint: Endpoint,
    binding: Binding,
    selection: Selection,
    bounds: Bounds,
    code: Vec<C>,
    coefficients: Vec<f64>,
    table: Vec<f64>,
    radius: f64,
}

impl Model {
    pub fn new(
        source: tagged::Model,
        records: Vec<records::Record>,
        endpoint: Endpoint,
        expected_source_sha256: &str,
        expected_records_sha256: &str,
    ) -> Result<Self> {
        let bounds = model::forecast(source.n, source.q, source.vocab, records.len())?;
        source.validate()?;
        if endpoint.source_reference.is_empty()
            || endpoint.source_reference.len() > 1024
            || (!endpoint.source_gradient_training_run
                && endpoint.source_optimizer_updates.is_some_and(|n| n != 0))
        {
            return Err("invalid inherited endpoint metadata".into());
        }
        let mut ids = BTreeSet::new();
        for r in &records {
            if r.cell >= source.n * source.n || r.label >= source.vocab || !ids.insert(r.id) {
                return Err("invalid original record ID, cell or target".into());
            }
        }
        let binding = Binding {
            source_sha256: digest(&source)?,
            records_sha256: digest(&records)?,
            endpoint_sha256: digest(&endpoint)?,
            protocol_sha256: digest(&serde_json::from_str::<serde_json::Value>(PROTOCOL)?)?,
        };
        if binding.source_sha256 != expected_source_sha256
            || binding.records_sha256 != expected_records_sha256
        {
            return Err("source or original records differ from external binding".into());
        }
        let selection = selection::select(&source, &records);
        let code = model::code(source.q);
        let coefficients = model::initializer(&source, &code, selection.selected_margin);
        let radius = 64. * source.q as f64;
        for c in 0..source.vocab {
            let charge: f64 = (0..2 * source.q)
                .map(|k| coefficients[k * source.vocab + c].abs())
                .sum();
            if !charge.is_finite() || charge > radius + 1e-8 {
                return Err("complete complex coefficient budget exceeded".into());
            }
        }
        let mut table = Vec::with_capacity(source.q * source.vocab);
        for state in 0..source.q {
            let z = model::product(
                &code[state * source.q..(state + 1) * source.q],
                &coefficients,
                source.vocab,
            );
            for (c, &actual) in z.iter().enumerate() {
                let declared = if c == source.decoder[state] {
                    selection.selected_margin
                } else {
                    0.
                };
                if !actual.is_finite() || (actual - declared).abs() > 1e-8 {
                    return Err("actual complex products fail the declared margin identity".into());
                }
            }
            table.extend(z);
        }
        let actual_ce = records
            .iter()
            .map(|r| {
                let start = source.latent_labels[r.cell] * source.vocab;
                tagged::loss(&table[start..start + source.vocab], r.label)
            })
            .sum::<f64>()
            / records.len() as f64;
        let expected_ce = selection
            .candidates
            .iter()
            .find(|c| c.margin == selection.selected_margin)
            .ok_or("missing selected margin")?
            .mean_ce;
        if !actual_ce.is_finite() || (actual_ce - expected_ce).abs() > 1e-8 {
            return Err("actual original-record CE fails the CM44 error identity".into());
        }
        Ok(Self {
            source,
            records,
            endpoint,
            binding,
            selection,
            bounds,
            code,
            coefficients,
            table,
            radius,
        })
    }

    pub fn load(value: serde_json::Value, expected: &Binding) -> Result<Self> {
        let saved: Saved = serde_json::from_value(value)?;
        if &saved.binding != expected {
            return Err("saved decoder-head binding differs from external binding".into());
        }
        let actual = Self::new(
            saved.source,
            saved.records,
            saved.endpoint,
            &expected.source_sha256,
            &expected.records_sha256,
        )?;
        if &actual.binding != expected
            || actual.selection != saved.selection
            || actual.bounds != saved.bounds
            || actual.code != saved.code
            || actual.coefficients != saved.coefficients
            || actual.table != saved.table
            || actual.radius != saved.radius
        {
            return Err("saved decoder head, selection, metadata or costs changed".into());
        }
        Ok(actual)
    }
}
