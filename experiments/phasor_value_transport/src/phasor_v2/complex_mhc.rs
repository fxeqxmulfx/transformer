//! Complex read/write gates and phase-rotated Sinkhorn mixing.
use super::{
    Result, grid,
    tape::{Context, Pair, Tape},
    weights::{Parameters, Projection},
};
use kernite_cuda::{Cuda, Grid};

pub const STREAMS: usize = 4;
pub const AMPLITUDES: usize = 24;

#[derive(Clone, Copy)]
pub enum Routing {
    Single,
    Real,
    Conjugated,
    EntryPhase,
}

impl Routing {
    pub fn selected() -> Result<Self> {
        match std::env::var("PHASOR_V2_MHC").as_deref() {
            Err(std::env::VarError::NotPresent) => Ok(Self::Single),
            Ok("real") => Ok(Self::Real),
            Ok("conjugated") => Ok(Self::Conjugated),
            Ok("entry-phase") => Ok(Self::EntryPhase),
            _ => Err("PHASOR_V2_MHC takes real, conjugated or entry-phase".into()),
        }
    }

    pub fn model(self) -> &'static str {
        match self {
            Self::Single => "phasor-v2-single-stream-tied-vocab-fp32",
            Self::Real => "phasor-v2-real-mhc-tied-vocab-fp32",
            Self::Conjugated => "phasor-v2-conjugated-mhc-tied-vocab-fp32",
            Self::EntryPhase => "phasor-v2-entry-phase-mhc-tied-vocab-fp32",
        }
    }

    pub(super) fn router_width(self) -> Option<usize> {
        match self {
            Self::Single => None,
            Self::Real => Some(12),
            Self::Conjugated => Some(18),
            Self::EntryPhase => Some(24),
        }
    }
}

pub(super) struct Kernels {
    pub(super) real: crate::mhc::kernels::Kernels,
    phase: device::Kernels,
}

impl Kernels {
    pub fn load(cuda: &Cuda) -> Result<Self> {
        Ok(Self {
            real: crate::mhc::kernels::Kernels::load(cuda)?,
            phase: device::Kernels::load(cuda)?,
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct Route {
    pub logits: Pair,
    pub packed: Pair,
    pub amplitudes: Pair,
    pub pre: Pair,
    pub post: Pair,
    pub residual: Pair,
    pub edge_gradient: Option<Pair>,
}

impl Route {
    pub fn apply<const BACK: bool>(self, k: &Kernels) -> Result<()> {
        let rows = self.packed.len / AMPLITUDES;
        let stride = (self.logits.len / rows) as u32;
        let count = self.packed.len as u32;
        let launch = grid(count as usize);
        let sinkhorn = Grid::new(rows.div_ceil(2) as u32, 32);
        let edges = self.edge_gradient.map_or(self.logits.dx, |edge| edge.x);
        // SAFETY: the packed and phase arrays cover the same rows. Each row's
        // gates and adjoints are packed independently in the default stream.
        unsafe {
            if BACK {
                k.phase.convert::<true>(
                    launch,
                    self.pre.dx,
                    self.post.dx,
                    self.residual.dx,
                    self.amplitudes.x,
                    self.amplitudes.dx,
                    self.logits.x,
                    self.logits.dx,
                    edges,
                    stride,
                    count,
                )?;
                if let Some(edge) = self.edge_gradient {
                    k.phase.reduce_conjugated(
                        grid(rows * STREAMS),
                        self.logits.dx,
                        edge.x,
                        (rows * STREAMS) as u32,
                    )?;
                }
                k.real.sinkhorn::<true>(
                    sinkhorn,
                    self.packed.dx,
                    self.packed.x,
                    self.amplitudes.dx,
                    rows as u32,
                )?;
                k.phase
                    .pack::<true>(launch, self.logits.dx, self.packed.dx, stride, count)?;
            } else {
                k.phase
                    .pack::<false>(launch, self.packed.x, self.logits.x, stride, count)?;
                k.real.sinkhorn::<false>(
                    sinkhorn,
                    self.amplitudes.x,
                    self.packed.x,
                    self.amplitudes.dx,
                    rows as u32,
                )?;
                k.phase.convert::<false>(
                    launch,
                    self.pre.x,
                    self.post.x,
                    self.residual.x,
                    self.amplitudes.x,
                    self.amplitudes.dx,
                    self.logits.x,
                    self.logits.dx,
                    edges,
                    stride,
                    count,
                )?;
            }
        }
        Ok(())
    }
}

pub(super) struct Sublayer {
    state: Pair,
    route: Option<Route>,
    pub input: Pair,
}

impl Sublayer {
    pub fn read(
        tape: &mut Tape,
        ctx: &Context<'_>,
        params: &Parameters,
        router: Option<&Projection>,
        state: Pair,
        rows: usize,
        width: usize,
    ) -> Result<Self> {
        let Some(router) = router else {
            return Ok(Self {
                state,
                route: None,
                input: state,
            });
        };
        let normalized = tape.rms(ctx, state, STREAMS * width)?;
        let logits = router.project(tape, ctx, params, normalized, rows)?;
        let route = tape.route(ctx, logits, rows)?;
        let input = tape.mm(
            ctx,
            route.pre.matrix(rows, 1, STREAMS),
            state.matrix(rows, STREAMS, width),
        )?;
        Ok(Self {
            state,
            route: Some(route),
            input,
        })
    }

    pub fn write(
        self,
        tape: &mut Tape,
        ctx: &Context<'_>,
        branch: Pair,
        width: usize,
        scale: f32,
    ) -> Result<Pair> {
        if let Some(route) = self.route {
            let rows = branch.len / (2 * width);
            let skip = tape.mm(
                ctx,
                route.residual.matrix(rows, STREAMS, STREAMS),
                self.state.matrix(rows, STREAMS, width),
            )?;
            let write = tape.mm(
                ctx,
                route.post.matrix(rows, STREAMS, 1),
                branch.matrix(rows, 1, width),
            )?;
            tape.sum(ctx, skip, write, scale)
        } else {
            tape.sum(ctx, self.state, branch, scale)
        }
    }
}

#[kernite_macros::kernels]
mod device {
    fn index(i: u32) {
        let (block, lane, threads): (u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(lane, tid.x);
        mov.u32(threads, ntid.x);
        mad.lo.u32(i, block, threads, lane);
    }

    fn load(v: f32, p: u64, i: u32) {
        let at: u64;
        mad.wide.u32(at, i, 4, p);
        ld.global.f32(v, [at]);
    }

    fn store(p: u64, i: u32, v: f32) {
        let at: u64;
        mad.wide.u32(at, i, 4, p);
        st.global.f32([at], v);
    }

    // The first 24 raw coordinates supply amplitudes under every rotation law.
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn pack<const BACK: bool>(
        out: *mut f32,
        values: *const f32,
        stride: u32,
        count: u32,
    ) {
        let (i, row, gate, raw): (u32, u32, u32, u32);
        index(i);
        let stop: pred;
        setp.ge.u32(stop, i, count);
        if stop {
            ret();
        }
        div.u32(row, i, 24);
        rem.u32(gate, i, 24);
        mad.lo.u32(raw, row, stride, gate);
        let value: f32;
        if BACK {
            load(value, values, i);
            store(out, raw, value);
        } else {
            load(value, values, raw);
            let diag: pred;
            setp.eq.u32(diag, gate, 8);
            setp.eq.or.u32(diag, gate, 13, diag);
            setp.eq.or.u32(diag, gate, 18, diag);
            setp.eq.or.u32(diag, gate, 23, diag);
            if diag {
                add.rn.f32(value, value, 2.0);
            }
            store(out, i, value);
        }
    }

    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn convert<const BACK: bool>(
        a: *mut f32,
        b: *mut f32,
        r: *mut f32,
        amplitudes: *const f32,
        damp: *mut f32,
        raw: *const f32,
        draw: *mut f32,
        edge_grad: *mut f32,
        stride: u32,
        count: u32,
    ) {
        let (i, row, gate, at, angle_at, edge, src, dst): (u32, u32, u32, u32, u32, u32, u32, u32);
        index(i);
        let (stop, pre, post, mix, real, conjugated): (pred, pred, pred, pred, pred, pred);
        setp.ge.u32(stop, i, count);
        if stop {
            ret();
        }
        div.u32(row, i, 24);
        rem.u32(gate, i, 24);
        setp.lt.u32(pre, gate, 4);
        setp.lt.u32(mix, gate, 8);
        and.pred(post, mix, !pre);
        let out: u64;
        sub.u32(edge, gate, 8);
        selp.b64(out, a, b, pre);
        mov.u32(src, gate);
        if post {
            sub.u32(src, gate, 4);
        }
        mad.lo.u32(at, row, 4, src);
        if !mix {
            mov.u64(out, r);
            mad.lo.u32(at, row, 16, edge);
        }
        mul.lo.u32(at, at, 2);
        let (angle, other, c, s, g, gr, gi, value): (f32, f32, f32, f32, f32, f32, f32, f32);
        mov.f32(angle, 0.0);
        mov.u32(angle_at, 0);
        setp.eq.u32(real, stride, 24);
        setp.eq.u32(conjugated, stride, 36);
        if real {
            bra(trig);
        }
        // Four stream phases encode every edge through theta_i - theta_j.
        and.pred(stop, conjugated, !mix);
        if stop {
            bra(stream_angles);
        }
        mul.lo.u32(dst, row, stride);
        add.u32(angle_at, dst, 24);
        add.u32(angle_at, angle_at, gate);
        load(angle, raw, angle_at);
        bra(trig);
        label(stream_angles);
        div.u32(dst, edge, 4);
        rem.u32(src, edge, 4);
        mad.lo.u32(angle_at, row, stride, 32);
        add.u32(src, angle_at, src);
        add.u32(angle_at, angle_at, dst);
        load(angle, raw, angle_at);
        load(other, raw, src);
        sub.rn.f32(angle, angle, other);
        label(trig);
        cos.approx.f32(c, angle);
        sin.approx.f32(s, angle);
        load(g, amplitudes, i);
        if BACK {
            load(gr, out, at);
            add.u32(at, at, 1);
            load(gi, out, at);
            mul.rn.f32(value, gi, s);
            fma.rn.f32(value, gr, c, value);
            store(damp, i, value);
            if real {
                ret();
            }
            mul.rn.f32(value, gi, c);
            neg.f32(s, s);
            fma.rn.f32(value, gr, s, value);
            mul.rn.f32(value, value, g);
            if stop {
                mad.lo.u32(at, row, 16, edge);
                store(edge_grad, at, value);
                ret();
            }
            store(draw, angle_at, value);
        } else {
            mul.rn.f32(c, c, g);
            mul.rn.f32(s, s, g);
            store(out, at, c);
            add.u32(at, at, 1);
            store(out, at, s);
        }
    }

    // Pull edge phases theta_i - theta_j back to one angle per stream.
    pub unsafe extern "ptx-kernel" fn reduce_conjugated(
        draw: *mut f32,
        edge_grad: *const f32,
        count: u32,
    ) {
        let (i, row, col, base, at): (u32, u32, u32, u32, u32);
        index(i);
        let stop: pred;
        setp.ge.u32(stop, i, count);
        if stop {
            ret();
        }
        div.u32(row, i, 4);
        rem.u32(col, i, 4);
        mul.lo.u32(base, row, 16);
        let (total, a, b): (f32, f32, f32);
        mov.f32(total, 0.0);
        for j in 0..4 {
            mad.lo.u32(at, col, 4, base);
            add.u32(at, at, j);
            load(a, edge_grad, at);
            add.u32(at, base, col);
            add.u32(at, at, 4 * j);
            load(b, edge_grad, at);
            sub.rn.f32(a, a, b);
            add.rn.f32(total, total, a);
        }
        mad.lo.u32(at, row, 36, 32);
        add.u32(at, at, col);
        store(draw, at, total);
    }
}
