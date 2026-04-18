| Game | Stoch Score | Det Score | %Ammo | %Stam | Kills A/K | Attacks A/K | Failures | EpLen |
|---|---|---|---|---|---|---|---|---|
| g0 | 8.70 ± 9.65 | — | 0.00 | 0.00 | 7.5 / 4.2 | 306.8 / 12.9 | 3.0 | 186 |
| g1a | 1.70 ± 3.00 | — | 0.60 | 0.55 | 1.0 / 3.6 | 36.3 / 11.9 | 2.9 | 158 |
| g1b | 1.60 ± 2.76 | — | 0.60 | 0.55 | 0.9 / 3.6 | 36.0 / 12.0 | 2.9 | 158 |
| g2 | 1.80 ± 1.08 | −1.90 ± 1.14 | 0.97 | 0.86 | 4.5 / 0.1 | 70.9 / 49.2 | 2.8 | 171 |
| g3 | 4.10 ± 2.17 | 0.60 ± 1.43 | 0.96 | 0.85 | 5.5 / 1.0 | 71.0 / 64.9 | 2.4 | 176 |
| g4 | 7.40 ± 3.32 | 0.20 ± 2.71 | 0.90 | 0.93 | 10.0 / 0.1 | 62.3 / 28.5 | 2.7 | 190 |
| **g5** | **11.20 ± 3.12** | **2.90 ± 2.62** | 0.88 | 0.81 | 12.3 / 0.7 | 56.7 / 23.8 | 1.8 | 183 |

Score = kills − failures. Deterministic scores use greedy argmax actions; stochastic scores sample from the policy. Seed convention: episode *i* uses seed `42 + i` identically across all games, so zombie spawn patterns are directly comparable.
