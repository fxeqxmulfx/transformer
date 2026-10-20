`schedulefree-1.4.1.tar.gz` is the unmodified source distribution published at
https://pypi.org/project/schedulefree/1.4.1/ on 2025-03-24, including its Apache
2.0 license. SHA256:
`69ef25601d1fc0d8dd00cb36f9af78833f88b7846f1bb6ddecc9f144f3e9f7cb`.

Download URL:
https://files.pythonhosted.org/packages/bd/51/d78369c32b0f3818ee7f5a04e77ed02a6bd0ba8642462f780623af0010f5/schedulefree-1.4.1.tar.gz

Its `schedulefree/adamw_schedulefree.py` has SHA256
`a51a9946b729d3cac96037ecdc0cb626e9278d8a24563c84836ff7d7a2a0cc28`,
identical to the file at facebookresearch/schedule_free stability commit
`59ed9b662d5ec5799aded7390b2482758423653c` (version 1.3).

Tests verify the archive hash before importing that source directly. They
compare both upstream execution paths with the local optimizer, offline,
and keep rational checkpoints shared with `scripts/ScheduleFreeAdamWParity.lean`.
No expected values are generated from the local implementation.
