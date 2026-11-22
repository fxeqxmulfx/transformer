//! Prototype searches for the adjacent addresses and the trusted triple bank.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(value: f32, buffer: u64, index: u32) {
        let address: u64;
        mad.wide.u32(address, index, 4, buffer);
        ld.global.f32(value, [address]);
    }

    // A positive maximum must be unique. A later larger value supersedes a tie.
    fn consider(best: f32, winner: u32, score: f32, slot: u32) {
        let (greater, equal): (pred, pred);
        setp.gt.f32(greater, score, best);
        setp.eq.f32(equal, score, best);
        selp.u32(winner, 0xffffffff, winner, equal);
        selp.u32(winner, slot, winner, greater);
        selp.f32(best, score, best, greater);
    }

    // Each warp scans one cell's prototype scores. A lane remembers ties
    // among its own slots. The ballot detects ties between different lanes.
    pub unsafe extern "ptx-kernel" fn trusted_bank(
        output: *mut u32,
        products: *const f32,
        inverse: *const f32,
        scales: *const f32,
        slots: u32,
        cells: u32,
        threshold: f32,
    ) {
        let (thread, cell, lane, i, start): (u32, u32, u32, u32, u32);
        let (block, size, local): (u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(size, ntid.x);
        mov.u32(local, tid.x);
        mad.lo.u32(thread, block, size, local);
        shr.u32(cell, thread, 5);
        and.b32(lane, thread, 31);
        let (outside, leader): (pred, pred);
        setp.ge.u32(outside, cell, cells);
        if outside {
            ret();
        }
        setp.eq.u32(leader, lane, 0);
        let (scale, best, score, inv): (f32, f32, f32, f32);
        let (winner, absent): (u32, u32);
        let out_at: u64;
        mad.wide.u32(out_at, cell, 4, output);
        mov.u32(absent, 0xffffffff);
        load(scale, scales, cell);
        let (masked, masked_leader): (pred, pred);
        setp.eq.f32(masked, scale, 0.0);
        and.pred(masked_leader, masked, leader);
        if masked_leader {
            st.global.u32([out_at], absent);
        }
        if masked {
            ret();
        }
        mov.f32(best, threshold);
        mov.u32(winner, absent);
        mul.lo.u32(start, cell, slots);
        mov.u32(i, lane);
        let done: pred;
        let offset: u32;
        label(columns);
        setp.ge.u32(done, i, slots);
        if done {
            bra(peak);
        }
        add.u32(offset, start, i);
        load(score, products, offset);
        load(inv, inverse, i);
        mul.rn.f32(score, score, inv);
        mul.rn.f32(score, score, scale);
        consider(best, winner, score, i);
        add.u32(i, i, 32);
        bra(columns);

        label(peak);
        let (highest, other): (f32, f32);
        mov.f32(highest, best);
        for k in 0..5 {
            shfl.sync.bfly.b32(other, highest, 1 << k, 31, 0xffffffff);
            max.f32(highest, highest, other);
        }
        let (above, equal, attains, single): (pred, pred, pred, pred);
        setp.gt.f32(above, best, threshold);
        setp.eq.f32(equal, best, highest);
        and.pred(attains, above, equal);
        let (mask, count, source, selected): (u32, u32, u32, u32);
        vote.sync.ballot.b32(mask, attains, 0xffffffff);
        popc.b32(count, mask);
        bfind.u32(source, mask);
        shfl.sync.idx.b32(selected, winner, source, 31, 0xffffffff);
        setp.eq.u32(single, count, 1);
        selp.u32(selected, selected, absent, single);
        if leader {
            st.global.u32([out_at], selected);
        }
    }

    // One block per position, one lane per prototype. The dot products use
    // the scalar oracle's order and its individually rounded operations.
    pub unsafe extern "ptx-kernel" fn cells(
        output: *mut u32,
        input: *const f32,
        hidden: *const f32,
        dictionary_t: *const f32,
        inverse_norms: *const f32,
        width: u32,
        length: u32,
        units: f32,
    ) {
        let pairs: shared<[f32; 512], 8>;
        let (row, slot, slots, position): (u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mov.u32(slot, tid.x);
        mov.u32(slots, ntid.x);
        rem.u32(position, row, length);
        let (out_at, absent): (u64, u32);
        mad.wide.u32(out_at, row, 4, output);
        mov.u32(absent, 0xffffffff);
        let (first, leader, first_leader): (pred, pred, pred);
        setp.eq.u32(first, position, 0);
        setp.eq.u32(leader, slot, 0);
        and.pred(first_leader, first, leader);
        if first_leader {
            st.global.u32([out_at], absent);
        }
        if first {
            ret();
        }

        let (current_at, previous_at, dictionary_at, end): (u32, u32, u32, u32);
        mul.lo.u32(current_at, row, width);
        sub.u32(previous_at, current_at, width);
        add.u32(end, current_at, width);
        mov.u32(dictionary_at, slot);
        let (left, right, key, query, context, prototype, product, inverse): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        mov.f32(left, 0.0);
        mov.f32(right, 0.0);
        let more: pred;
        label(coordinates);
        load(key, input, previous_at);
        load(context, hidden, previous_at);
        mul.rn.f32(context, units, context);
        add.rn.f32(key, key, context);
        load(query, input, current_at);
        load(prototype, dictionary_t, dictionary_at);
        mul.rn.f32(product, prototype, key);
        add.rn.f32(left, left, product);
        mul.rn.f32(product, prototype, query);
        add.rn.f32(right, right, product);
        add.u32(current_at, current_at, 1);
        add.u32(previous_at, previous_at, 1);
        add.u32(dictionary_at, dictionary_at, slots);
        setp.lt.u32(more, current_at, end);
        if more {
            bra(coordinates);
        }
        load(inverse, inverse_norms, slot);
        mul.rn.f32(left, left, inverse);
        mul.rn.f32(right, right, inverse);
        let (base, pair): (u32, u32);
        mov.u32(base, pairs);
        mad.lo.u32(pair, slot, 8, base);
        st.shared.v2.f32([pair], (left, right));
        bar.sync(0);

        if !leader {
            ret();
        }
        let (left_best, right_best): (f32, f32);
        let (a, b, i, cell): (u32, u32, u32, u32);
        mov.f32(left_best, 0.0);
        mov.f32(right_best, 0.0);
        mov.u32(a, absent);
        mov.u32(b, absent);
        mov.u32(i, 0);
        mov.u32(pair, base);
        label(candidates);
        ld.shared.v2.f32((left, right), [pair]);
        consider(left_best, a, left, i);
        consider(right_best, b, right, i);
        add.u32(i, i, 1);
        add.u32(pair, pair, 8);
        setp.lt.u32(more, i, slots);
        if more {
            bra(candidates);
        }
        let (missing_a, missing_b, missing): (pred, pred, pred);
        setp.eq.u32(missing_a, a, absent);
        setp.eq.u32(missing_b, b, absent);
        or.pred(missing, missing_a, missing_b);
        mad.lo.u32(cell, a, slots, b);
        selp.u32(cell, absent, cell, missing);
        st.global.u32([out_at], cell);
    }
}
