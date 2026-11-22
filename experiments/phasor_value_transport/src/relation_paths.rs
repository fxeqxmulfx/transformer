//! Causal two-hop values, specified by Transformer.Phasor.RelationalPaths.
//! A triple writes its last token under its first two tokens. The current
//! query follows two edges through each relation in this prefix only.
use crate::engine::Result;
use kernite_cuda::{Buffer, Cuda};
use std::collections::BTreeMap;

pub struct RelationRead {
    pub kernels: kernels::Kernels,
    pub endpoints: Buffer<u32>,
    host: Vec<u32>,
    vocab: usize,
}

impl RelationRead {
    pub fn new(cuda: &Cuda, rows: usize, vocab: usize) -> Result<Self> {
        Ok(Self {
            kernels: kernels::Kernels::load(cuda)?,
            endpoints: cuda.alloc(rows * vocab)?,
            host: vec![0; rows * vocab],
            vocab,
        })
    }

    pub fn prepare(&mut self, tokens: &[u32], length: usize) -> Result<()> {
        if length <= 2 {
            return Ok(());
        }
        let mut edges = BTreeMap::new();
        let batches = self.host.chunks_exact_mut(length * self.vocab);
        for (prompt, batch) in tokens.chunks_exact(length).zip(batches) {
            edges.clear();
            for (i, (&source, row)) in prompt
                .iter()
                .zip(batch.chunks_exact_mut(self.vocab))
                .enumerate()
            {
                if i >= 3 {
                    edges.insert((prompt[i - 3], prompt[i - 2]), prompt[i - 1]);
                }
                row.fill(source);
                for (&(_, relation), &middle) in edges.range((source, 0)..=(source, u32::MAX)) {
                    if let Some(&end) = edges.get(&(middle, relation)) {
                        row[relation as usize] = end;
                    }
                }
            }
        }
        self.endpoints.write(&self.host)?;
        Ok(())
    }
}

#[kernite_macros::kernels]
pub mod kernels {
    // One lane owns every coefficient of a row. Relations sharing an
    // endpoint accumulate in vocabulary order, including signed amplitudes.
    pub unsafe extern "ptx-kernel" fn add(
        weights: *mut f32,
        endpoints: *const u32,
        tokens: *const u32,
        amplitude: *const f32,
        vocab: u32,
        rows: u32,
    ) {
        let (row, width, lane, base, source, dest, relation, at): (
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
        );
        mov.u32(row, ctaid.x);
        mov.u32(width, ntid.x);
        mov.u32(lane, tid.x);
        mad.lo.u32(row, row, width, lane);
        let (done, skip): (pred, pred);
        setp.ge.u32(done, row, rows);
        if done {
            ret();
        }
        let (ptr, source_ptr): (u64, u64);
        let (current, value, a): (f32, f32, f32);
        mad.wide.u32(ptr, row, 4, tokens);
        ld.global.u32(source, [ptr]);
        mul.lo.u32(base, row, vocab);
        add.u32(at, base, source);
        mad.wide.u32(source_ptr, at, 4, weights);
        ld.global.f32(current, [source_ptr]);
        mov.u32(relation, 0);
        label(scan);
        setp.ge.u32(done, relation, vocab);
        if done {
            bra(finish);
        }
        add.u32(at, base, relation);
        mad.wide.u32(ptr, at, 4, endpoints);
        ld.global.u32(dest, [ptr]);
        setp.eq.u32(skip, dest, source);
        if skip {
            bra(next);
        }
        mad.wide.u32(ptr, relation, 4, amplitude);
        ld.global.f32(a, [ptr]);
        setp.eq.f32(skip, a, 0.0);
        if skip {
            bra(next);
        }
        add.u32(at, base, dest);
        mad.wide.u32(ptr, at, 4, weights);
        ld.global.f32(value, [ptr]);
        add.rn.f32(value, value, a);
        st.global.f32([ptr], value);
        sub.rn.f32(current, current, a);
        label(next);
        add.u32(relation, relation, 1);
        bra(scan);
        label(finish);
        st.global.f32([source_ptr], current);
    }

    // One warp reduces each relation across positions. An empty active
    // prefix overwrites all amplitudes' derivatives with zero.
    pub unsafe extern "ptx-kernel" fn adjoint(
        gradient: *mut f32,
        dweights: *const f32,
        endpoints: *const u32,
        tokens: *const u32,
        vocab: u32,
        rows: u32,
    ) {
        let (relation, lane, row, base, source, dest, at): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(relation, ctaid.x);
        mov.u32(lane, tid.x);
        mov.u32(row, lane);
        let (done, skip): (pred, pred);
        let ptr: u64;
        let (sum, x, y): (f32, f32, f32);
        mov.f32(sum, 0.0);
        label(scan);
        setp.ge.u32(done, row, rows);
        if done {
            bra(reduce);
        }
        mul.lo.u32(base, row, vocab);
        add.u32(at, base, relation);
        mad.wide.u32(ptr, at, 4, endpoints);
        ld.global.u32(dest, [ptr]);
        mad.wide.u32(ptr, row, 4, tokens);
        ld.global.u32(source, [ptr]);
        setp.eq.u32(skip, dest, source);
        if skip {
            bra(next);
        }
        add.u32(at, base, dest);
        mad.wide.u32(ptr, at, 4, dweights);
        ld.global.f32(x, [ptr]);
        add.u32(at, base, source);
        mad.wide.u32(ptr, at, 4, dweights);
        ld.global.f32(y, [ptr]);
        sub.rn.f32(x, x, y);
        add.rn.f32(sum, sum, x);
        label(next);
        add.u32(row, row, 32);
        bra(scan);
        label(reduce);
        for j in 0..5 {
            shfl.sync.bfly.b32(x, sum, 1 << j, 31, 0xffffffff);
            add.rn.f32(sum, sum, x);
        }
        setp.ne.u32(skip, lane, 0);
        if skip {
            ret();
        }
        mad.wide.u32(ptr, relation, 4, gradient);
        st.global.f32([ptr], sum);
    }
}
