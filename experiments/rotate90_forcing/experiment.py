"""Matched full-state continuations separating gradients from query decay.

Source: review/flat_spectrum.md, Gradient forcing. Every source is update400
of the frozen transfer study. Both arms retain the complete source optimizer
and sampler state and continue to absolute update1500 with CUDA max-autotune.
"""

from lab.dsl import *

sources = [('p97k2-w16-s31', '242640977358e74a9ca24948bffe1d25dfa8c6fd3412d059e3c6b59fea57ab36'),
 ('p97k2-w16-s37', '4c92814e811a20e36905ab8d1e8f777fbdd6ff192f9f6d43ff43facaacd76639'),
 ('p97k2-w16-s41', 'c362dbfd7820e79ee88ff36ee561f800f8104637ecaaae7355e47901b85107a3'),
 ('p97k2-w64-s31', '42aaadd96105bc788a948eea75a36c724a5402ff6c324e69cce63d7420b2c563'),
 ('p97k2-w64-s37', '5c7bbcf743503f15f96497233957ef482a7eebd74a3a37d3e3af722b97db91f0'),
 ('p97k2-w64-s41', '8d25efd637ff574b2e9fe03b31471dad7ea1514fccb264f901f8bb9e406a0bf2'),
 ('p193k3-w16-s31', '950aac3d9fcb78fe35d2a84111982ae0e03bb995c2ff2a159d636c2800bc8ce0'),
 ('p193k3-w16-s37', '35b325f10dfcffcc4a02634114ac74c892a01db7d200bf968d635557ccf5e31d'),
 ('p193k3-w16-s41', '4f2913d43ea9047f781414a6b5038b54ed98c850d4fa0d46057c992f7f868c15'),
 ('p193k3-w64-s31', '2e6bcf7f6d50938d0791f69657895b485c57a7418b54217b4b7ba8d56e2e5d06'),
 ('p193k3-w64-s37', 'e29fd24e1d2d415786c8a2835c2f84d1dc9e33d916a01d87cf8c8976dd27b150'),
 ('p193k3-w64-s41', '0117010f73adfe2d6475bd59b8478f2d41059c9b2c5c5519418c961624e4b841'),
 ('p17k1-w64-s31', '8dedcedb3623fadb569816652432e456c9fa721e078e643a83d16ad462de365c'),
 ('p17k1-w64-s37', '6d5f1d423a4a620f42bc742cbdb021cad4ee6c07020e46223feb016bd4a972f2'),
 ('p17k1-w64-s41', 'dd406dad50599f8b5ea8c622283c368ed8973d4a4d2e944e9e2f5cf7c79cadd7')]

experiments = {}
for source, digest in sources:
    task, width_text, seed_text = source.split("-")
    prime_text, exponent_text = task[1:].split("k")
    base = Experiment(
        model=PeriRotate90(int(width_text[1:]), 1),
        benchmark=PowerDivision(int(prime_text), int(exponent_text), .5),
        optimizer=ScheduleFreeAdamW(lr=.003, betas=(.9, .98), weight_decay=1., warmup=0),
        schedule=Schedule(warmup=10, inclusive=True), budget=Budget(1500, 512),
        seeds=Seeds(int(seed_text[1:]), 11, 11), evaluate=Evaluate(50, 1024),
        checkpoint=ScheduleFreeContinuation(50,
            f"experiments/rotate90_transfer/runs/{source}/checkpoints/step-00000400.pt", digest),
        execution=MaxAutotune(device="cuda"),
    )
    experiments[f"full-{source}"] = base
    experiments[f"no-decay-{source}"] = swap(base, "optimizer.weight_decay", 0.)
