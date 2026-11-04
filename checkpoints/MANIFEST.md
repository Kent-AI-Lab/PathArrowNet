# Checkpoints

Download `patharrownet_checkpoints.tar` (1.5 GB) from the v1.0 release of this repository:
https://github.com/Kent-AI-Lab/PathArrowNet/releases/download/v1.0/patharrownet_checkpoints.tar

Extract it here and check the files:

```
cd checkpoints
tar -xf patharrownet_checkpoints.tar
awk -F'\t' 'NR>1 {print $7"  "$5}' MANIFEST.tsv | sha256sum -c
```

`MANIFEST.tsv` lists 50 files (1.5 GB) of the primary model: `group`, `setting`, `method` (score-dump name), `seed`, `release_name`,
`size_bytes`, `sha256` (computed from the files of the reported runs) and a note.

| group | model | files |
|---|---|---|
| `patharrownet` | PathArrowNet with KGML relation-level weights (primary), five strict splits x seeds 42-46 | `best.pt` + `config.json` per run |

Notes

- PathArrowNet `best.pt` files are plain `state_dict`s; the frozen ESM2 feature table is stored inside as a buffer
  (`dense_X0`), which is why each is 60 MB. `patharrownet.predict` rebuilds the model from `config.json` and re-roots the
  split directory to this repository. `config.json` also records training options, which inference ignores.
