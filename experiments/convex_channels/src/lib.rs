//! Joint convex routing/output channels: FlatSpectrum.ConvexChannel, CC1-CC8.
//! Inputs use categorical identities and supplied key/value bindings. Every
//! routing/output interaction is trained directly; no task rule enters forward.

pub mod channel;
pub mod provenance;
pub mod task;

#[cfg(test)]
mod tests;
