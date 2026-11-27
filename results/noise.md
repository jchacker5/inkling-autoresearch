# Seed noise of the baseline

9 valid baseline runs (base + 5 minutes of unchanged training, differing only in seed: data order and loop sampling), screening eval set.

| metric | mean | std | n |
|---|---|---|---|
| val_bpb | 1.4676 | 0.0050 | 9 |
| kappa_id | -0.0667 | 0.0000 | 9 |
| kappa_ood | -0.0667 | 0.0000 | 9 |
| reason_id | 0.0000 | 0.0000 | 9 |
| steps | 143.4444 | 29.4114 | 9 |

Screening rule used by swarm/run.py (candidate iff t >= 0.92 on the lane's primary metric):

- kappa_id-primary lanes: candidate iff kappa_id >= -0.0667
- val_bpb-primary lanes: candidate iff val_bpb <= 1.4630
- gate for every run: val_bpb <= 1.4876 (baseline mean + 0.02)

Power state per run: AC
