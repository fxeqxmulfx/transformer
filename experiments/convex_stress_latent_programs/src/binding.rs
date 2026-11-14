//! Frozen training-only family priority. Run the reviewed CM52 binder first;
//! use CM53 only if that entire copy construction has no usable initialization.
use crate::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub input: Input,
    pub copy: copy::binding::Report,
    pub latent: Option<latent::Report>,
    pub latent_error: Option<String>,
}
pub fn run(input: Input) -> Result<Report> {
    let copy = copy::binding::run(input.clone())?;
    let (latent, latent_error) = if copy.initialization_error.is_some() {
        match latent::run(input.clone()) {
            Ok(model) => (Some(model), None),
            Err(e) => (None, Some(e.to_string())),
        }
    } else {
        (None, None)
    };
    Ok(Report {
        input,
        copy,
        latent,
        latent_error,
    })
}
pub fn check(input: &Input, out: &Report) -> Result<()> {
    if *out != run(input.clone())? {
        return Err("training-only family selection or complete bound replay differs".into());
    }
    copy::binding::check(input, &out.copy)?;
    if let Some(model) = &out.latent {
        latent::check(input, model)?;
    }
    Ok(())
}
