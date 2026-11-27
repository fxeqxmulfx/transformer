//! Complex skip and write gates around each residual sublayer.
use super::{
    Result,
    tape::{Context, Pair, Tape},
    weights::{Layer, Parameters},
};

/// One gated sublayer. The skip and the write each multiply every token by
/// `1 + <normalized input, gate vector>`, so zero gate vectors give ordinary
/// residual addition.
pub(super) struct Sublayer {
    skip: Pair,
    write_gate: Pair,
    /// The RMS-normalized block input, which also drives both gates.
    pub input: Pair,
}

impl Sublayer {
    pub fn read(
        tape: &mut Tape,
        ctx: &Context<'_>,
        params: &Parameters,
        layer: &Layer,
        sublayer: usize,
        state: Pair,
        width: usize,
    ) -> Result<Self> {
        let input = tape.rms(ctx, state, width)?;
        let gate = |offsets: [usize; 2]| params.pair(offsets[sublayer], 2 * width);
        Ok(Self {
            skip: tape.gate_rows(ctx, state, input, gate(layer.skip_gates))?,
            write_gate: gate(layer.write_gates),
            input,
        })
    }

    pub fn write(
        self,
        tape: &mut Tape,
        ctx: &Context<'_>,
        branch: Pair,
        scale: f32,
    ) -> Result<Pair> {
        let write = tape.gate_rows(ctx, branch, self.input, self.write_gate)?;
        tape.sum(ctx, self.skip, write, scale)
    }
}
