# Turyn Sieve Comparison

Seed: 42
Steps per run: 20
Timeout per run: 60s

| solver | n | order | random start | sieved start |
| --- | ---: | ---: | --- | --- |
| TurynGreedy | 2 | 20 | YES; algo=0.2s | YES; algo=0.0s |
| TurynGreedy | 3 | 32 | OK; e=512; rms=1.02; orth=464/496; algo=0.0s | SIEVED OUT |
| TurynGreedy | 4 | 44 | YES; algo=0.0s | YES; algo=0.0s |
| TurynGreedy | 5 | 56 | OK; e=1792; rms=1.08; orth=1428/1540; algo=0.0s | OK; e=1792; rms=1.08; orth=1428/1540; algo=0.0s |
| TurynGreedy | 6 | 68 | OK; e=4352; rms=1.38; orth=2210/2278; algo=0.0s | OK; e=8704; rms=1.95; orth=2142/2278; algo=0.0s |
| TurynGreedy | 7 | 80 | OK; e=14080; rms=2.11; orth=2920/3160; algo=0.0s | SIEVED OUT |
| TurynGreedy | 8 | 92 | OK; e=19136; rms=2.14; orth=3726/4186; algo=0.0s | OK; e=17664; rms=2.05; orth=3634/4186; algo=0.0s |
| TurynGreedy | 9 | 104 | OK; e=19968; rms=1.93; orth=4940/5356; algo=0.0s | OK; e=6656; rms=1.11; orth=4940/5356; algo=0.0s |
| TurynGreedy | 36 | 428 | OK; e=3985536; rms=6.60; orth=78538/91378; algo=0.0s | OK; e=3554112; rms=6.24; orth=77254/91378; algo=0.2s |
| TurynGreedy | 56 | 668 | OK; e=20339264; rms=9.56; orth=188042/222778; algo=0.0s | OK; e=18650560; rms=9.15; orth=188710/222778; algo=0.5s |
| TurynAnnealing | 2 | 20 | YES; algo=0.0s | YES; algo=0.0s |
| TurynAnnealing | 3 | 32 | OK; e=512; rms=1.02; orth=464/496; algo=0.0s | SIEVED OUT |
| TurynAnnealing | 4 | 44 | OK; e=2112; rms=1.49; orth=814/946; algo=0.0s | YES; algo=0.0s |
| TurynAnnealing | 5 | 56 | OK; e=1792; rms=1.08; orth=1428/1540; algo=0.0s | OK; e=1792; rms=1.08; orth=1428/1540; algo=0.0s |
| TurynAnnealing | 6 | 68 | OK; e=3264; rms=1.20; orth=2074/2278; algo=0.0s | OK; e=5440; rms=1.55; orth=1938/2278; algo=0.0s |
| TurynAnnealing | 7 | 80 | OK; e=14080; rms=2.11; orth=2920/3160; algo=0.0s | SIEVED OUT |
| TurynAnnealing | 8 | 92 | OK; e=14720; rms=1.88; orth=3818/4186; algo=0.0s | OK; e=26496; rms=2.52; orth=3910/4186; algo=0.0s |
| TurynAnnealing | 9 | 104 | OK; e=33280; rms=2.49; orth=4732/5356; algo=0.0s | OK; e=6656; rms=1.11; orth=4940/5356; algo=0.0s |
| TurynAnnealing | 36 | 428 | OK; e=4697728; rms=7.17; orth=78538/91378; algo=0.0s | OK; e=5848192; rms=8.00; orth=78538/91378; algo=0.1s |
| TurynAnnealing | 56 | 668 | OK; e=22006592; rms=9.94; orth=188042/222778; algo=0.0s | OK; e=19665920; rms=9.40; orth=188042/222778; algo=0.1s |
| TurynPOCS | 2 | 20 | YES; algo=0.1s | YES; algo=0.1s |
| TurynPOCS | 3 | 32 | OK; e=512; rms=1.02; orth=464/496; algo=0.0s | SIEVED OUT |
| TurynPOCS | 4 | 44 | YES; algo=0.0s | YES; algo=0.0s |
| TurynPOCS | 5 | 56 | OK; e=1792; rms=1.08; orth=1428/1540; algo=0.0s | OK; e=1792; rms=1.08; orth=1428/1540; algo=0.0s |
| TurynPOCS | 6 | 68 | YES; algo=0.1s | YES; algo=0.0s |
| TurynPOCS | 7 | 80 | OK; e=3840; rms=1.10; orth=2920/3160; algo=0.0s | SIEVED OUT |
| TurynPOCS | 8 | 92 | OK; e=1472; rms=0.59; orth=4094/4186; algo=0.1s | OK; e=1472; rms=0.59; orth=4094/4186; algo=0.0s |
| TurynPOCS | 9 | 104 | OK; e=6656; rms=1.11; orth=4940/5356; algo=0.1s | OK; e=6656; rms=1.11; orth=4940/5356; algo=0.0s |
| TurynPOCS | 36 | 428 | OK; e=1540800; rms=4.11; orth=81106/91378; algo=0.0s | OK; e=1349056; rms=3.84; orth=80250/91378; algo=0.1s |
| TurynPOCS | 56 | 668 | OK; e=9234432; rms=6.44; orth=193386/222778; algo=0.0s | OK; e=7011328; rms=5.61; orth=192050/222778; algo=0.3s |
| TurynSteepest | 2 | 20 | YES; algo=0.0s | YES; algo=0.0s |
| TurynSteepest | 3 | 32 | OK; e=512; rms=1.02; orth=464/496; algo=0.0s | SIEVED OUT |
| TurynSteepest | 4 | 44 | OK; e=1408; rms=1.22; orth=858/946; algo=0.0s | YES; algo=0.0s |
| TurynSteepest | 5 | 56 | OK; e=1792; rms=1.08; orth=1428/1540; algo=0.0s | OK; e=1792; rms=1.08; orth=1428/1540; algo=0.0s |
| TurynSteepest | 6 | 68 | OK; e=3264; rms=1.20; orth=2074/2278; algo=0.0s | OK; e=4352; rms=1.38; orth=2006/2278; algo=0.0s |
| TurynSteepest | 7 | 80 | OK; e=8960; rms=1.68; orth=2840/3160; algo=0.0s | SIEVED OUT |
| TurynSteepest | 8 | 92 | OK; e=7360; rms=1.33; orth=3726/4186; algo=0.0s | OK; e=4416; rms=1.03; orth=3910/4186; algo=0.0s |
| TurynSteepest | 9 | 104 | OK; e=13312; rms=1.58; orth=4836/5356; algo=0.0s | OK; e=6656; rms=1.11; orth=4940/5356; algo=0.0s |
| TurynSteepest | 36 | 428 | OK; e=1013504; rms=3.33; orth=78966/91378; algo=0.0s | OK; e=1136768; rms=3.53; orth=78538/91378; algo=0.1s |
| TurynSteepest | 56 | 668 | OK; e=5921152; rms=5.16; orth=191382/222778; algo=0.0s | OK; e=4040064; rms=4.26; orth=192718/222778; algo=0.1s |
| MonteCarlo | 2 | 20 | YES; algo=0.0s | YES; algo=0.0s |
| MonteCarlo | 3 | 32 | OK; e=512; rms=1.02; orth=464/496; algo=0.0s | SIEVED OUT |
| MonteCarlo | 4 | 44 | YES; algo=0.0s | YES; algo=0.0s |
| MonteCarlo | 5 | 56 | OK; e=1792; rms=1.08; orth=1428/1540; algo=0.0s | OK; e=1792; rms=1.08; orth=1428/1540; algo=0.0s |
| MonteCarlo | 6 | 68 | YES; algo=0.0s | YES; algo=0.0s |
| MonteCarlo | 7 | 80 | OK; e=3840; rms=1.10; orth=2920/3160; algo=0.0s | SIEVED OUT |
| MonteCarlo | 8 | 92 | OK; e=1472; rms=0.59; orth=4094/4186; algo=0.0s | OK; e=2944; rms=0.84; orth=4002/4186; algo=0.0s |
| MonteCarlo | 9 | 104 | OK; e=6656; rms=1.11; orth=4940/5356; algo=0.0s | OK; e=6656; rms=1.11; orth=4940/5356; algo=0.1s |
| MonteCarlo | 36 | 428 | OK; e=2342016; rms=5.06; orth=77682/91378; algo=0.0s | OK; e=2369408; rms=5.09; orth=78538/91378; algo=0.1s |
| MonteCarlo | 56 | 668 | OK; e=12227072; rms=7.41; orth=189378/222778; algo=0.0s | OK; e=10057408; rms=6.72; orth=191382/222778; algo=0.3s |
| Spectral | 2 | 20 | YES; algo=0.2s | YES; algo=0.1s |
| Spectral | 3 | 32 | OK; e=512; rms=1.02; orth=464/496; algo=0.1s | SIEVED OUT |
| Spectral | 4 | 44 | YES; algo=0.1s | YES; algo=0.1s |
| Spectral | 5 | 56 | OK; e=1792; rms=1.08; orth=1428/1540; algo=0.1s | OK; e=1792; rms=1.08; orth=1428/1540; algo=0.1s |
| Spectral | 6 | 68 | YES; algo=0.1s | YES; algo=0.1s |
| Spectral | 7 | 80 | OK; e=3840; rms=1.10; orth=2920/3160; algo=0.1s | SIEVED OUT |
| Spectral | 8 | 92 | YES; algo=0.1s | OK; e=2944; rms=0.84; orth=4002/4186; algo=0.1s |
| Spectral | 9 | 104 | OK; e=6656; rms=1.11; orth=4940/5356; algo=0.2s | OK; e=6656; rms=1.11; orth=4940/5356; algo=0.2s |
| Spectral | 36 | 428 | OK; e=561536; rms=2.48; orth=80250/91378; algo=0.1s | OK; e=472512; rms=2.27; orth=80250/91378; algo=0.2s |
| Spectral | 56 | 668 | OK; e=2148288; rms=3.11; orth=195390/222778; algo=0.1s | OK; e=2372736; rms=3.26; orth=196058/222778; algo=0.4s |
| Genetic | 2 | 20 | YES; algo=0.1s | YES; algo=0.0s |
| Genetic | 3 | 32 | OK; e=512; rms=1.02; orth=464/496; algo=0.0s | SIEVED OUT |
| Genetic | 4 | 44 | OK; e=704; rms=0.86; orth=902/946; algo=0.1s | YES; algo=0.0s |
| Genetic | 5 | 56 | OK; e=1792; rms=1.08; orth=1428/1540; algo=0.0s | OK; e=1792; rms=1.08; orth=1428/1540; algo=0.0s |
| Genetic | 6 | 68 | OK; e=2176; rms=0.98; orth=2142/2278; algo=0.1s | OK; e=2176; rms=0.98; orth=2142/2278; algo=0.0s |
| Genetic | 7 | 80 | OK; e=8960; rms=1.68; orth=2840/3160; algo=0.1s | SIEVED OUT |
| Genetic | 8 | 92 | OK; e=5888; rms=1.19; orth=4094/4186; algo=0.0s | OK; e=4416; rms=1.03; orth=3910/4186; algo=0.1s |
| Genetic | 9 | 104 | OK; e=13312; rms=1.58; orth=4836/5356; algo=0.1s | OK; e=6656; rms=1.11; orth=4940/5356; algo=0.1s |
| Genetic | 36 | 428 | OK; e=1568192; rms=4.14; orth=78110/91378; algo=0.1s | OK; e=1643520; rms=4.24; orth=79394/91378; algo=0.1s |
| Genetic | 56 | 668 | OK; e=8005312; rms=5.99; orth=190714/222778; algo=0.0s | OK; e=9223744; rms=6.43; orth=191382/222778; algo=0.0s |
| Ising | 2 | 20 | YES; algo=0.0s | YES; algo=0.0s |
| Ising | 3 | 32 | OK; e=512; rms=1.02; orth=464/496; algo=0.0s | SIEVED OUT |
| Ising | 4 | 44 | YES; algo=0.1s | YES; algo=0.0s |
| Ising | 5 | 56 | OK; e=1792; rms=1.08; orth=1428/1540; algo=0.0s | OK; e=1792; rms=1.08; orth=1428/1540; algo=0.0s |
| Ising | 6 | 68 | OK; e=1088; rms=0.69; orth=2210/2278; algo=0.1s | YES; algo=0.0s |
| Ising | 7 | 80 | OK; e=3840; rms=1.10; orth=2920/3160; algo=0.1s | SIEVED OUT |
| Ising | 8 | 92 | OK; e=4416; rms=1.03; orth=3910/4186; algo=0.1s | OK; e=5888; rms=1.19; orth=3818/4186; algo=0.0s |
| Ising | 9 | 104 | OK; e=13312; rms=1.58; orth=4836/5356; algo=0.1s | OK; e=6656; rms=1.11; orth=4940/5356; algo=0.1s |
| Ising | 36 | 428 | OK; e=1855808; rms=4.51; orth=78966/91378; algo=0.1s | OK; e=1527104; rms=4.09; orth=77682/91378; algo=0.1s |
| Ising | 56 | 668 | OK; e=9405440; rms=6.50; orth=188710/222778; algo=0.0s | OK; e=6199040; rms=5.28; orth=189378/222778; algo=0.3s |
