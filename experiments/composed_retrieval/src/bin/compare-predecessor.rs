//! Compare the learned predecessor scalar with the frozen CM117 parent.
#[path = "shared/pilot.rs"]
mod pilot;

fn main() -> composed_retrieval::Result<()> {
    pilot::compare(true)
}
