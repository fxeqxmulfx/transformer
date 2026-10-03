# AMSGradW training reproduction

Six complete fresh runs use the migrated application, CUDA Graphs, original model/data/settings,
both attention modes and seeds 0, 1, 2. Historical runs remain unchanged.

| Run | Old test CE | New test CE | Absolute delta | Best step old/new | Exact reproduction |
| --- | ---: | ---: | ---: | --- | --- |
| softmax:amsgradw:0 | 1.623400612622 | 1.623400612622 | 0 | 12250/12250 | True |
| softmax:amsgradw:1 | 1.626090363985 | 1.626090363985 | 0 | 15250/15250 | True |
| softmax:amsgradw:2 | 1.626633163435 | 1.626633163435 | 0 | 8500/8500 | True |
| sparsemax:amsgradw:0 | 1.656125687294 | 1.656125687294 | 0 | 9000/9000 | True |
| sparsemax:amsgradw:1 | 1.622563498986 | 1.622563498986 | 0 | 13250/13250 | True |
| sparsemax:amsgradw:2 | 1.638117022684 | 1.638117022684 | 0 | 12250/12250 | True |

Exact reproduction requires equal validation/training curves, stopping decisions,
initial weights, full/used batch plans, and the selected model SHA256. Wall-clock
timings and source/protocol fingerprints are expected to differ after migration.
