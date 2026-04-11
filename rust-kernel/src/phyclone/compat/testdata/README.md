# PhyClone Oracle Fixtures

`phyclone_oracle_fixture.json` is consumed by the Rust test:
- `phyclone::compat::loader::tests::matches_phyclone_oracle_fixture_when_present`

Generate it with:

```bash
python scripts/export_phyclone_oracle_fixture.py \
  --input-tsv <path/to/input.tsv> \
  --cluster-tsv <path/to/cluster.tsv> \
  --output rust-kernel/src/phyclone/compat/testdata/phyclone_oracle_fixture.json
```

Notes:
- The script imports from the workspace's `PhyClone` directory by default.
- It requires Python deps used by `Roth-Lab/PhyClone` (`numpy`, `pandas`, `numba`, `scipy`, etc.).
- If the fixture is absent, the Rust parity test exits early (treated as skipped).

Current fixture set includes:
- clustered beta-binomial baseline
- clustered binomial baseline
- outlier-split clustered beta-binomial case
- CN/purity sensitivity case
- single-sample low-precision beta-binomial case
