Even for division modulo 97, it remains open whether a single uninterrupted
run of the full Argand model trained with bias-corrected AdamW from its stock
random initialization learns all hidden answers permanently with probability
greater than 3/4. Permanent recovery means that from some training step onward,
the correct token has a strictly higher logit than every other token on every
held-out input.

For division modulo 97 in real arithmetic, **3 of 5 equally weighted milestones
are complete (60%)**. Lean proves that the observed equations identify the target
within its class, that Argand at width 64 can represent it, and that two bounds
on AdamW training imply the recovery guarantee above. The two open milestones
are to bound the change in the recovery certificate along actual AdamW updates
after burn-in and to bound the expected error in that estimate under the stock
training law.
