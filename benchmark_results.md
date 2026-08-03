# Hadamard Benchmark

Backend: cupy
Seed: 42
Step budgets: n=4: 2000, n=8: 2000, n=12: 2000, n=16: 2000, n=20: 2000, n=668: 5000
Timeout per case: 60 seconds

Cell format: `status; e=energy; rms=root-mean-square correlation; orth=orthogonal pairs; t=seconds`.

## Individual strategies

| Strategy | 4 | 8 | 12 | 16 | 20 | 668 |
|---|---|---|---|---|---|---|
| Circulant | YES 1.2s | YES 1.2s | YES 1.2s | YES 1.1s | YES 1.2s | OK; e=4692032; rms=4.59; orth=175350/222778; t=1.2s |
| Annealing | YES 1.2s | YES 1.2s | YES 1.1s | YES 1.2s | YES 1.2s | OK; e=8614528; rms=6.22; orth=174014/222778; t=1.2s |
| BaumertHall | YES 1.2s | N/A | YES 1.2s | N/A | YES 1.2s | OK; e=11607168; rms=7.22; orth=171342/222778; t=1.3s |
| Diffset | YES 1.2s | YES 1.2s | YES 1.2s | YES 1.1s | YES 1.1s | OK; e=887104; rms=2.00; orth=167334/222778; t=1.2s |
| MonteCarlo | YES 1.2s | YES 1.2s | YES 1.1s | YES 1.1s | YES 1.1s | OK; e=855040; rms=1.96; orth=187374/222778; t=1.8s |
| DirectSearch | OK; e=12; rms=1.41; orth=3/6; t=0.9s | OK; e=28; rms=1.00; orth=21/28; t=1.0s | OK; e=80; rms=1.10; orth=46/66; t=0.9s | OK; e=112; rms=0.97; orth=92/120; t=1.0s | OK; e=640; rms=1.84; orth=78/190; t=0.9s | OK; e=138652912; rms=24.95; orth=7141/222778; t=1.1s |
| RepairSearch | YES 0.9s | OK; e=76; rms=1.65; orth=12/28; t=0.9s | YES 0.9s | OK; e=588; rms=2.21; orth=36/120; t=0.9s | OK; e=940; rms=2.22; orth=57/190; t=0.9s | OK; e=132473056; rms=24.39; orth=7393/222778; t=1.2s |
| Ising | YES 1.2s | YES 1.2s | YES 1.1s | OK; e=128; rms=1.03; orth=112/120; t=1.3s | OK; e=320; rms=1.30; orth=170/190; t=1.2s | OK; e=4702720; rms=4.59; orth=172678/222778; t=1.7s |
| Spectral | YES 1.5s | YES 1.4s | OK; e=192; rms=1.71; orth=54/66; t=1.5s | OK; e=128; rms=1.03; orth=112/120; t=1.5s | OK; e=320; rms=1.30; orth=170/190; t=1.6s | OK; e=2426176; rms=3.30; orth=178690/222778; t=4.2s |
