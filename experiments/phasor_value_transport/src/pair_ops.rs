//! Add selected memory embeddings before RMS. In the transpose each parameter
//! has one writer and accumulates its examples in a deterministic order.
#[kernite_macros::kernels]
pub mod kernels {
    fn word(v: u32, ptr: u64, i: u32) {
        let address: u64;
        mad.wide.u32(address, i, 4, ptr);
        ld.global.u32(v, [address]);
    }
    fn pair_at(at: u32, row: u32, col: u32, width: u32) {
        mad.lo.u32(at, row, 2, 1);
        mad.lo.u32(at, at, width, col);
    }
    pub unsafe extern "ptx-kernel" fn read(
        hidden: *mut f32,
        values: *const f32,
        keys: *const u32,
        width: u32,
        tables: u32,
    ) {
        let (row, col, at, ix, end, key): (u32, u32, u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mov.u32(col, tid.x);
        pair_at(at, row, col, width);
        let (dest, ptr): (u64, u64);
        let (sum, val): (f32, f32);
        mad.wide.u32(dest, at, 4, hidden);
        ld.global.f32(sum, [dest]);
        mul.lo.u32(ix, row, tables);
        add.u32(end, ix, tables);
        let more: pred;
        label(next);
        word(key, keys, ix);
        mad.lo.u32(at, key, width, col);
        mad.wide.u32(ptr, at, 4, values);
        ld.global.f32(val, [ptr]);
        add.rn.f32(sum, sum, val);
        add.u32(ix, ix, 1);
        setp.lt.u32(more, ix, end);
        if more {
            bra(next);
        }
        st.global.f32([dest], sum);
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn backward<const CLEAR: bool>(
        gradients: *mut f32,
        dhidden: *const f32,
        offsets: *const u32,
        owners: *const u32,
        width: u32,
    ) {
        let (key, col, ix, end, row, at): (u32, u32, u32, u32, u32, u32);
        mov.u32(key, ctaid.x);
        mov.u32(col, tid.x);
        let (sum, val): (f32, f32);
        let ptr: u64;
        mov.f32(sum, 0.0);
        if !CLEAR {
            word(ix, offsets, key);
            add.u32(at, key, 1);
            word(end, offsets, at);
            let done: pred;
            label(next);
            setp.ge.u32(done, ix, end);
            if done {
                bra(write);
            }
            word(row, owners, ix);
            pair_at(at, row, col, width);
            mad.wide.u32(ptr, at, 4, dhidden);
            ld.global.f32(val, [ptr]);
            add.rn.f32(sum, sum, val);
            add.u32(ix, ix, 1);
            bra(next);
        }
        label(write);
        mad.lo.u32(at, key, width, col);
        mad.wide.u32(ptr, at, 4, gradients);
        st.global.f32([ptr], sum);
    }
    // Batch by class. Preserve the first causal output and its direct gradient.
    #[instances((true, false), (false, false), (false, true))]
    pub unsafe extern "ptx-kernel" fn logits<const COPY_ANSWER: bool, const RESTORE: bool>(
        ordinary: *mut f32,
        temp: *mut f32,
        vocab: u32,
    ) {
        let (row, col, at): (u32, u32, u32);
        let stride: u32;
        mov.u32(stride, ntid.x);
        mov.u32(row, ctaid.x);
        mov.u32(col, tid.x);
        mul.lo.u32(row, row, 2);
        if COPY_ANSWER {
            add.u32(row, row, 1);
        }
        let (ptr, other): (u64, u64);
        let (val, saved): (f32, f32);
        label(next);
        let done: pred;
        setp.ge.u32(done, col, vocab);
        if done {
            ret();
        }
        mad.lo.u32(at, row, vocab, col);
        mad.wide.u32(ptr, at, 4, ordinary);
        mad.wide.u32(other, at, 4, temp);
        if COPY_ANSWER {
            ld.global.f32(val, [other]);
            st.global.f32([ptr], val);
        } else {
            ld.global.f32(val, [ptr]);
            if RESTORE {
                ld.global.f32(saved, [other]);
                add.rn.f32(val, val, saved);
            } else {
                st.global.f32([other], val);
                mov.f32(val, 0.0);
            }
            st.global.f32([ptr], val);
        }
        add.u32(col, col, stride);
        bra(next);
    }
}
