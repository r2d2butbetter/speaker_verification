# Speaker Verification (TIMIT)

This repo contains a simple GMM-UBM speaker verification pipeline using the TIMIT corpus.

## Data Placement
- Expected TIMIT root: `data/raw_timit/data`
- After unzipping, you should have these folders:
  - `data/raw_timit/data/TRAIN/DR1/...`
  - `data/raw_timit/data/TEST/DR1/...`

```
/data/raw_timit/data/TRAIN
/data/raw_timit/data/TEST
```

Example tree:
```
data/
  raw_timit/
    TIMITDIC.TXT
    SPKRINFO.TXT
    data/
      TRAIN/
        DR1/...
      TEST/
        DR1/...
```