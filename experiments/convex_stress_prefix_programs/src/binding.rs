//! Frozen training-only priority: preserve the complete reviewed task328
//! binder. Only lack of its usable initialization permits the new family.
use crate::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub input: Input,
    pub parent: Option<parent::binding::Report>,
    pub parent_error: Option<String>,
    pub latent: Option<latent::Report>,
    pub latent_error: Option<String>,
}
impl Report {
    pub fn parent_usable(&self) -> bool {
        self.parent.as_ref().is_some_and(|p| {
            p.parent_usable()
                || p.latent
                    .as_ref()
                    .is_some_and(|l| l.initialization_error.is_none())
        })
    }
}
pub fn run(input: Input) -> Result<Report> {
    let (parent, parent_error) = match parent::binding::run(input.clone()) {
        Ok(p) => (Some(p), None),
        Err(e) => (None, Some(e.to_string())),
    };
    let mut out = Report {
        input,
        parent,
        parent_error,
        latent: None,
        latent_error: None,
    };
    if !out.parent_usable() {
        match latent::run(out.input.clone()) {
            Ok(model) => out.latent = Some(model),
            Err(e) => out.latent_error = Some(e.to_string()),
        }
    }
    Ok(out)
}
pub fn check(input: &Input, out: &Report) -> Result<()> {
    if *out != run(input.clone())? {
        return Err("complete family priority/input/state replay differs".into());
    }
    if let Some(p) = &out.parent {
        parent::binding::check(input, p)?;
    }
    if let Some(l) = &out.latent {
        latent::check(input, l)?;
    }
    Ok(())
}
