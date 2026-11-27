//! A register-tiled complex matrix product, after the FP32 SGEMM of
//! kernite's `examples/matmul` (a1b6bc2, MIT), as ported to
//! `research/complex_geometry_suite/native/src/gemm.rs` of the pinned GPT
//! checkout: tiles pass through shared memory in two buffers, a block waits
//! at one barrier for each, and each thread keeps a 4 × 4 block of complex
//! outputs in registers.
//!
//! Without a split of `k`, each output is one chain of fused multiply-adds
//! over `k` in order, the chain of `matmul` in `kernels.rs`, so the two
//! kernels agree bit for bit. A split adds its chains in order instead.
#[kernite_macros::kernels]
pub(super) mod device {
    /// Complex rows and columns of a block's tile of the output.
    pub const TILE: usize = 64;
    /// Complex steps of `k` in a stage of shared memory.
    pub const DEPTH: usize = 8;
    /// Complex numbers from one step's row of a shared tile to the next: the
    /// tile and four more, so that the 8-byte stores of a warp take two
    /// passes over the banks, the fewest they can.
    const STRIDE: usize = TILE + 4;
    /// Bytes of one stage of one operand in shared memory.
    const STAGE: usize = DEPTH * STRIDE * 8;
    /// Floats of the two stages of one operand.
    const FLOATS: usize = 2 * DEPTH * STRIDE * 2;

    /// Flags of `cgemm` and `reduce`.
    pub const CONJUGATE_A: u32 = 1;
    pub const CONJUGATE_B: u32 = 2;
    pub const CONJUGATE_Y: u32 = 4;
    pub const TRANSPOSE_Y: u32 = 8;
    pub const ADD: u32 = 16;
    /// Store the raw chains of this block's part of `k` to the workspace
    /// at `y`, part after part, each as `nctaid.z` row-major `m × n` matrices.
    pub const PARTIAL: u32 = 32;

    /// `Y = op(A) op(B)` for `nctaid.z` batches of complex matrices, `m × k`
    /// by `k × n`, stored as interleaved real and imaginary parts. `TA` says
    /// that the batch of `A` is stored `k × m`, and `TB` that that of `B` is
    /// stored `n × k`. Blocks run along x over the `tiles_n`-wide row-major
    /// grid of output tiles, along y over the parts of `k`, `span` steps
    /// each, and along z over the batches.
    #[instances((false, false), (true, false), (false, true), (true, true))]
    pub unsafe extern "ptx-kernel" fn cgemm<const TA: bool, const TB: bool>(
        y: *mut f32,
        a: *const f32,
        b: *const f32,
        m: u32,
        n: u32,
        k: u32,
        tiles_n: u32,
        span: u32,
        flags: u32,
    ) {
        let a_tiles: shared<[f32; FLOATS], 16>;
        let b_tiles: shared<[f32; FLOATS], 16>;
        let (t, tx, ty, u, tile, i0, j0, l, lend, batch): (u32, u32, u32, u32, u32, u32, u32, u32, u32, u32);
        mov.u32(t, tid.x);
        lanes(tx, ty);
        mov.u32(tile, ctaid.x);
        div.u32(i0, tile, tiles_n);
        rem.u32(j0, tile, tiles_n);
        mul.lo.u32(i0, i0, TILE);
        mul.lo.u32(j0, j0, TILE);
        mov.u32(u, ctaid.y);
        mul.lo.u32(l, u, span);
        add.u32(lend, l, span);
        min.u32(lend, lend, k);
        mov.u32(batch, ctaid.z);

        note("the two consecutive complex numbers this thread loads of each operand");
        let (ra, ka, rb, kb): (u32, u32, u32, u32);
        if TA {
            shr.u32(ka, t, 5);
            and.b32(ra, t, 31);
            shl.b32(ra, ra, 1);
        } else {
            shr.u32(ra, t, 2);
            and.b32(ka, t, 3);
            shl.b32(ka, ka, 1);
        }
        if TB {
            shr.u32(rb, t, 2);
            and.b32(kb, t, 3);
            shl.b32(kb, kb, 1);
        } else {
            shr.u32(kb, t, 5);
            and.b32(rb, t, 31);
            shl.b32(rb, rb, 1);
        }
        let (fixed_a, fixed_b): ([pred; 2], [pred; 2]);
        let (row, col): (u32, u32);
        add.u32(row, i0, ra);
        add.u32(col, j0, rb);
        setp.lt.u32(fixed_a[0], row, m);
        setp.lt.u32(fixed_b[0], col, n);
        if TA {
            add.u32(u, row, 1);
            setp.lt.u32(fixed_a[1], u, m);
        }
        if !TB {
            add.u32(u, col, 1);
            setp.lt.u32(fixed_b[1], u, n);
        }

        note("the complex offsets of those numbers at the first stage, and the advance per stage");
        let (oa, ob, da, db, base): (u32, u32, u32, u32, u32);
        mul.lo.u32(base, m, k);
        mul.lo.u32(oa, base, batch);
        mul.lo.u32(base, k, n);
        mul.lo.u32(ob, base, batch);
        add.u32(u, l, ka);
        if TA {
            mad.lo.u32(oa, u, m, oa);
            add.u32(oa, oa, row);
            mul.lo.u32(da, m, DEPTH);
        } else {
            mad.lo.u32(oa, row, k, oa);
            add.u32(oa, oa, u);
            mov.u32(da, DEPTH);
        }
        add.u32(u, l, kb);
        if TB {
            mad.lo.u32(ob, col, k, ob);
            add.u32(ob, ob, u);
            mov.u32(db, DEPTH);
        } else {
            mad.lo.u32(ob, u, n, ob);
            add.u32(ob, ob, col);
            mul.lo.u32(db, n, DEPTH);
        }

        note("shared addresses: where this thread stores, and where it reads its fragments");
        let (a_wr, b_wr, a_rd, b_rd): (u32, u32, u32, u32);
        mad.lo.u32(a_wr, ka, STRIDE, ra);
        mov.u32(base, a_tiles);
        mad.lo.u32(a_wr, a_wr, 8, base);
        mad.lo.u32(a_rd, ty, 16, base);
        mad.lo.u32(b_wr, kb, STRIDE, rb);
        mov.u32(base, b_tiles);
        mad.lo.u32(b_wr, b_wr, 8, base);
        mad.lo.u32(b_rd, tx, 16, base);

        let (ca, cb): (pred, pred);
        flag::<CONJUGATE_A>(ca, flags);
        flag::<CONJUGATE_B>(cb, flags);
        let (va, vb): ([f32; 4], [f32; 4]);
        let acc: [f32; 32];
        for i in 0..32 {
            mov.f32(acc[i], 0.0);
        }
        let (fa0, fb0, fa1, fb1): ([f32; 8], [f32; 8], [f32; 8], [f32; 8]);
        let (more, past): (pred, pred);
        setp.ge.u32(past, l, lend);
        if past {
            bra(done);
        }
        fetch::<TA, TB>(va, vb, a, b, oa, ob, l, lend, ka, kb, fixed_a, fixed_b, ca, cb);
        store::<TA, TB>(va, vb, a_wr, b_wr);
        bar.sync(0);
        let d: u32;
        mov.u32(d, STAGE);
        add.u32(a_wr, a_wr, d);
        add.u32(b_wr, b_wr, d);

        label(stage);
        note("load the next stage while this one is multiplied: past lend, zeros that no stage reads");
        add.u32(l, l, DEPTH);
        add.u32(oa, oa, da);
        add.u32(ob, ob, db);
        setp.lt.u32(more, l, lend);
        fetch::<TA, TB>(va, vb, a, b, oa, ob, l, lend, ka, kb, fixed_a, fixed_b, ca, cb);
        frags::<0>(fa0, fb0, a_rd, b_rd);
        for s in 0..DEPTH {
            let next = s + 1;
            if s % 2 == 0 {
                if next < DEPTH {
                    frags::<next>(fa1, fb1, a_rd, b_rd);
                }
                products(acc, fa0, fb0);
            } else {
                if next < DEPTH {
                    frags::<next>(fa0, fb0, a_rd, b_rd);
                }
                products(acc, fa1, fb1);
            }
        }
        store::<TA, TB>(va, vb, a_wr, b_wr);
        add.u32(a_rd, a_rd, d);
        sub.u32(a_wr, a_wr, d);
        add.u32(b_rd, b_rd, d);
        sub.u32(b_wr, b_wr, d);
        sub.u32(d, 0, d);
        bar.sync(0);
        if more {
            bra(stage);
        }

        label(done);
        note("rows ty * 2 + r and 32 + ty * 2 + r, columns tx * 2 + q and 32 + tx * 2 + q");
        lanes(tx, ty);
        note("a partial product is stored raw, whatever the output flags say");
        let (partial, ty_, cy, add, load): (pred, pred, pred, pred, pred);
        let out: u32;
        flag::<PARTIAL>(partial, flags);
        selp.u32(out, 0, flags, partial);
        flag::<TRANSPOSE_Y>(ty_, out);
        flag::<CONJUGATE_Y>(cy, out);
        flag::<ADD>(add, out);
        let (block, plane, at, r, c, p): (u32, u32, u32, u32, u32, u32);
        mov.u32(tile, ctaid.x);
        div.u32(i0, tile, tiles_n);
        rem.u32(j0, tile, tiles_n);
        mul.lo.u32(i0, i0, TILE);
        mul.lo.u32(j0, j0, TILE);
        mul.lo.u32(plane, m, n);
        mov.u32(block, ctaid.z);
        mov.u32(u, ctaid.y);
        selp.u32(u, u, 0, partial);
        mov.u32(p, nctaid.z);
        mad.lo.u32(block, u, p, block);
        mul.lo.u32(block, block, plane);
        let (re, im, old_re, old_im): (f32, f32, f32, f32);
        let (ok, okc): (pred, pred);
        let ptr: u64;
        for i in 0..4 {
            mad.lo.u32(r, ty, 2, i0);
            add.u32(r, r, i % 2 + 32 * (i / 2));
            setp.lt.u32(ok, r, m);
            for j in 0..4 {
                mad.lo.u32(c, tx, 2, j0);
                add.u32(c, c, j % 2 + 32 * (j / 2));
                setp.lt.u32(okc, c, n);
                and.pred(okc, okc, ok);
                mov.f32(re, acc[8 * i + 2 * j]);
                mov.f32(im, acc[8 * i + 2 * j + 1]);
                mad.lo.u32(at, r, n, c);
                if ty_ {
                    mad.lo.u32(at, c, m, r);
                }
                add.u32(at, at, block);
                mad.wide.u32(ptr, at, 8, y);
                if cy {
                    neg.f32(im, im);
                }
                and.pred(load, okc, add);
                if load {
                    ld.global.v2.f32((old_re, old_im), [ptr]);
                    add.rn.f32(re, re, old_re);
                    add.rn.f32(im, im, old_im);
                }
                if okc {
                    st.global.v2.f32([ptr], (re, im));
                }
            }
        }
    }

    /// Adds the `parts` partial products in the workspace `w` in order, then
    /// stores them to `y` as `cgemm` stores a whole product.
    pub unsafe extern "ptx-kernel" fn reduce(
        y: *mut f32,
        w: *const f32,
        m: u32,
        n: u32,
        count: u32,
        parts: u32,
        flags: u32,
    ) {
        let (i, block, threads, t, at, rest, row, col, batch, part): (u32, u32, u32, u32, u32, u32, u32, u32, u32, u32);
        mov.u32(t, tid.x);
        mov.u32(block, ctaid.x);
        mov.u32(threads, ntid.x);
        mad.lo.u32(i, block, threads, t);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let (re, im, x, z): (f32, f32, f32, f32);
        let ptr: u64;
        mad.wide.u32(ptr, i, 8, w);
        ld.global.v2.f32((re, im), [ptr]);
        mov.u32(part, 1);
        mov.u32(at, i);
        label(sum);
        setp.ge.u32(past, part, parts);
        if past {
            bra(summed);
        }
        add.u32(at, at, count);
        mad.wide.u32(ptr, at, 8, w);
        ld.global.v2.f32((x, z), [ptr]);
        add.rn.f32(re, re, x);
        add.rn.f32(im, im, z);
        add.u32(part, part, 1);
        bra(sum);
        label(summed);
        let (ty_, cy, add): (pred, pred, pred);
        flag::<TRANSPOSE_Y>(ty_, flags);
        flag::<CONJUGATE_Y>(cy, flags);
        flag::<ADD>(add, flags);
        mov.u32(at, i);
        if ty_ {
            rem.u32(col, i, n);
            div.u32(rest, i, n);
            rem.u32(row, rest, m);
            div.u32(batch, rest, m);
            mul.lo.u32(at, batch, m);
            mad.lo.u32(at, at, n, row);
            mad.lo.u32(at, col, m, at);
        }
        mad.wide.u32(ptr, at, 8, y);
        if cy {
            neg.f32(im, im);
        }
        if add {
            ld.global.v2.f32((x, z), [ptr]);
            add.rn.f32(re, re, x);
            add.rn.f32(im, im, z);
        }
        st.global.v2.f32([ptr], (re, im));
    }

    fn flag<const BIT: u32>(value: pred, flags: u32) {
        let x: u32;
        and.b32(x, flags, BIT);
        setp.ne.u32(value, x, 0);
    }

    /// This thread's place among the 256: `tx` and `ty`, 16 each, where a
    /// warp is 8 `tx` by 4 `ty`, as in the SGEMM.
    fn lanes(tx: u32, ty: u32) {
        let (t, hi): (u32, u32);
        mov.u32(t, tid.x);
        and.b32(tx, t, 7);
        and.b32(hi, t, 32);
        shr.u32(hi, hi, 2);
        or.b32(tx, tx, hi);
        shr.u32(ty, t, 3);
        and.b32(ty, ty, 3);
        shr.u32(hi, t, 6);
        shl.b32(hi, hi, 2);
        or.b32(ty, ty, hi);
    }

    /// Loads this thread's two complex numbers of each operand for the stage
    /// at step `l`, zeros where they lie outside the matrices or past `lend`,
    /// and conjugates them as the flags ask.
    fn fetch<const TA: bool, const TB: bool>(
        va: [f32; 4],
        vb: [f32; 4],
        a: u64,
        b: u64,
        oa: u32,
        ob: u32,
        l: u32,
        lend: u32,
        ka: u32,
        kb: u32,
        fixed_a: [pred; 2],
        fixed_b: [pred; 2],
        ca: pred,
        cb: pred,
    ) {
        let (u, w): (u32, u32);
        let (oka, okb): ([pred; 2], [pred; 2]);
        let p: u64;
        add.u32(u, l, ka);
        setp.lt.u32(oka[0], u, lend);
        if TA {
            and.pred(oka[1], oka[0], fixed_a[1]);
        } else {
            add.u32(w, u, 1);
            setp.lt.u32(oka[1], w, lend);
            and.pred(oka[1], oka[1], fixed_a[0]);
        }
        and.pred(oka[0], oka[0], fixed_a[0]);
        add.u32(u, l, kb);
        setp.lt.u32(okb[0], u, lend);
        if TB {
            add.u32(w, u, 1);
            setp.lt.u32(okb[1], w, lend);
            and.pred(okb[1], okb[1], fixed_b[0]);
        } else {
            and.pred(okb[1], okb[0], fixed_b[1]);
        }
        and.pred(okb[0], okb[0], fixed_b[0]);
        for j in 0..4 {
            mov.f32(va[j], 0.0);
            mov.f32(vb[j], 0.0);
        }
        mad.wide.u32(p, oa, 8, a);
        if oka[0] {
            ld.global.v2.f32((va[0], va[1]), [p]);
        }
        if oka[1] {
            ld.global.v2.f32((va[2], va[3]), [p + 8]);
        }
        mad.wide.u32(p, ob, 8, b);
        if okb[0] {
            ld.global.v2.f32((vb[0], vb[1]), [p]);
        }
        if okb[1] {
            ld.global.v2.f32((vb[2], vb[3]), [p + 8]);
        }
        if ca {
            neg.f32(va[1], va[1]);
            neg.f32(va[3], va[3]);
        }
        if cb {
            neg.f32(vb[1], vb[1]);
            neg.f32(vb[3], vb[3]);
        }
    }

    /// Stores what `fetch` loaded: a step's row of a shared tile holds the
    /// operand's numbers along `m` for `A` and along `n` for `B`.
    fn store<const TA: bool, const TB: bool>(va: [f32; 4], vb: [f32; 4], a_wr: u32, b_wr: u32) {
        if TA {
            st.shared.v4.f32([a_wr], (va[0], va[1], va[2], va[3]));
        } else {
            st.shared.v2.f32([a_wr], (va[0], va[1]));
            st.shared.v2.f32([a_wr + 8 * STRIDE], (va[2], va[3]));
        }
        if TB {
            st.shared.v2.f32([b_wr], (vb[0], vb[1]));
            st.shared.v2.f32([b_wr + 8 * STRIDE], (vb[2], vb[3]));
        } else {
            st.shared.v4.f32([b_wr], (vb[0], vb[1], vb[2], vb[3]));
        }
    }

    /// The numbers of step `T` of a stage for this thread's four rows and
    /// four columns.
    fn frags<const T: usize>(fa: [f32; 8], fb: [f32; 8], a: u32, b: u32) {
        ld.shared.v4.f32((fa[0], fa[1], fa[2], fa[3]), [a + 8 * STRIDE * T]);
        ld.shared.v4.f32((fa[4], fa[5], fa[6], fa[7]), [a + 8 * STRIDE * T + 256]);
        ld.shared.v4.f32((fb[0], fb[1], fb[2], fb[3]), [b + 8 * STRIDE * T]);
        ld.shared.v4.f32((fb[4], fb[5], fb[6], fb[7]), [b + 8 * STRIDE * T + 256]);
    }

    /// One complex step of all 16 chains, in the order of `matmul`:
    /// `re += ar br`, `im += ai br`, `im += ar bi`, `re += -ai bi`.
    fn products(acc: [f32; 32], fa: [f32; 8], fb: [f32; 8]) {
        let minus: f32;
        for i in 0..4 {
            neg.f32(minus, fa[2 * i + 1]);
            for j in 0..4 {
                fma.rn.f32(acc[8 * i + 2 * j], fa[2 * i], fb[2 * j], acc[8 * i + 2 * j]);
                fma.rn.f32(acc[8 * i + 2 * j + 1], fa[2 * i + 1], fb[2 * j], acc[8 * i + 2 * j + 1]);
                fma.rn.f32(acc[8 * i + 2 * j + 1], fa[2 * i], fb[2 * j + 1], acc[8 * i + 2 * j + 1]);
                fma.rn.f32(acc[8 * i + 2 * j], minus, fb[2 * j + 1], acc[8 * i + 2 * j]);
            }
        }
    }
}
