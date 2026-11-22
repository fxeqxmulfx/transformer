//! One CUDA lane scans one vocabulary column of one prompt. At position i
//! the counter includes exactly tokens 0 through i-2. The signed embedding
//! coefficients are reused by the tied-embedding adjoint.
#[kernite_macros::kernels]
pub mod kernels {
    pub unsafe extern "ptx-kernel" fn center(
        weights: *mut f32,
        tokens: *const u32,
        vocab: u32,
        length: u32,
        columns: u32,
        strength: f32,
    ) {
        let (column, row, class, pos, count, at, token): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(column, ctaid.x);
        mov.u32(at, ntid.x);
        mov.u32(pos, tid.x);
        mad.lo.u32(column, column, at, pos);
        let (done, matches): (pred, pred);
        setp.ge.u32(done, column, columns);
        if done {
            ret();
        }
        div.u32(row, column, vocab);
        rem.u32(class, column, vocab);
        mul.lo.u32(row, row, length);
        mov.u32(pos, 2);
        mov.u32(count, 0);
        let ptr: u64;
        let (freq, size, value): (f32, f32, f32);
        label(position);
        setp.ge.u32(done, pos, length);
        if done {
            ret();
        }
        add.u32(at, row, pos);
        sub.u32(at, at, 2);
        mad.wide.u32(ptr, at, 4, tokens);
        ld.global.u32(token, [ptr]);
        setp.eq.u32(matches, token, class);
        if matches {
            add.u32(count, count, 1);
        }
        cvt.rn.f32.u32(freq, count);
        sub.u32(at, pos, 1);
        cvt.rn.f32.u32(size, at);
        div.rn.f32(freq, freq, size);
        add.u32(at, row, pos);
        mad.wide.u32(ptr, at, 4, tokens);
        ld.global.u32(token, [ptr]);
        setp.eq.u32(matches, token, class);
        if matches {
            sub.rn.f32(freq, freq, 1.0);
        }
        mad.lo.u32(at, at, vocab, class);
        mad.wide.u32(ptr, at, 4, weights);
        ld.global.f32(value, [ptr]);
        fma.rn.f32(value, strength, freq, value);
        st.global.f32([ptr], value);
        add.u32(pos, pos, 1);
        bra(position);
    }
}
