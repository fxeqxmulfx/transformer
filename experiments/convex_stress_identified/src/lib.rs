//! CM28 / Lean9cd95708: replay actual ring row operations to certify a zero
//! anchored kernel. This is an observed structural certificate, conditional
//! on the declared categorical model; it does not establish structural truth.
pub mod ring;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
