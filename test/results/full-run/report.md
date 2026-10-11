# Experiment matrix

✅ ran, valid, modularity ≥ 90% of greedy reference · ⚠️ ran but poor/invalid · ❌ error, timeout or too many qubits

| Experiments Done | Tri-2 | Tri-3 | Tri-4 | Tri-8 | Tri-16 | Les Miserables | Karate Club | Davis Southern Women |
|---|---|---|---|---|---|---|---|---|
| Classical Binary Runs | ✅ | ⚠️ | ⚠️ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Quantum Binary Runs | ✅ | ⚠️ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cl 1-hot 2 | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| Q 1-hot 2 | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cl 1-hot 4 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| Q 1-hot 4 | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cl 1-hot 8 | ✅ | ✅ | ⚠️ | ⚠️ | ❌ | ❌ | ❌ | ❌ |
| Q 1-hot 8 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cl 1-hot 16 | ⚠️ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Q 1-hot 16 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cl H74 | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| Q H74 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cl H84 | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| Q H84 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

## Details

| Experiment | Graph | K | Status | Time (s) | Reason |
|---|---|---|---|---|---|
| Classical Binary Runs | Tri-2 | 2 | ✅ | 2.3 | Q=0.250 vs greedy ref 0.250 |
| Classical Binary Runs | Tri-3 | 3 | ⚠️ | 3.2 | low modularity: Q=0.278 vs greedy ref 0.417 |
| Classical Binary Runs | Tri-4 | 4 | ⚠️ | 2.9 | low modularity: Q=0.375 vs greedy ref 0.500 |
| Classical Binary Runs | Tri-8 | 8 | ✅ | 22.7 | Q=0.562 vs greedy ref 0.625 |
| Classical Binary Runs | Tri-16 | 16 | ❌ | 600.0 | timeout after 600s |
| Classical Binary Runs | Les Miserables | 2 | ✅ | 35.1 | Q=0.401 vs greedy ref 0.381 |
| Classical Binary Runs | Karate Club | 2 | ✅ | 3.3 | Q=0.404 vs greedy ref 0.404 |
| Classical Binary Runs | Davis Southern Women | 2 | ✅ | 2.5 | Q=0.320 vs greedy ref 0.309 |
| Quantum Binary Runs | Tri-2 | 2 | ✅ | 2.9 | Q=0.250 vs greedy ref 0.250 |
| Quantum Binary Runs | Tri-3 | 3 | ⚠️ | 9.6 | low modularity: Q=0.059 vs greedy ref 0.417 |
| Quantum Binary Runs | Tri-4 | 4 | ⚠️ | 69.0 | low modularity: Q=0.115 vs greedy ref 0.500 |
| Quantum Binary Runs | Tri-8 | 8 | ❌ | 0.0 | needs 72 qubits > simulator limit 29 |
| Quantum Binary Runs | Tri-16 | 16 | ❌ | 0.0 | needs 192 qubits > simulator limit 29 |
| Quantum Binary Runs | Les Miserables | 2 | ❌ | 0.0 | needs 77 qubits > simulator limit 29 |
| Quantum Binary Runs | Karate Club | 2 | ❌ | 0.0 | needs 34 qubits > simulator limit 29 |
| Quantum Binary Runs | Davis Southern Women | 2 | ❌ | 0.0 | needs 32 qubits > simulator limit 29 |
| Cl 1-hot 2 | Tri-2 | 2 | ✅ | 3.5 | Q=0.250 vs greedy ref 0.250 |
| Cl 1-hot 2 | Tri-3 | 2 | ✅ | 3.3 | Q=0.278 vs greedy ref 0.278 |
| Cl 1-hot 2 | Tri-4 | 2 | ✅ | 3.1 | Q=0.375 vs greedy ref 0.375 |
| Cl 1-hot 2 | Tri-8 | 2 | ✅ | 16.6 | Q=0.438 vs greedy ref 0.438 |
| Cl 1-hot 2 | Tri-16 | 2 | ✅ | 101.5 | Q=0.438 vs greedy ref 0.469 |
| Cl 1-hot 2 | Les Miserables | 2 | ⚠️ | 431.0 | low modularity: Q=0.343 vs greedy ref 0.381 |
| Cl 1-hot 2 | Karate Club | 2 | ✅ | 26.5 | Q=0.404 vs greedy ref 0.404 |
| Cl 1-hot 2 | Davis Southern Women | 2 | ✅ | 24.1 | Q=0.320 vs greedy ref 0.309 |
| Q 1-hot 2 | Tri-2 | 2 | ✅ | 4.6 | Q=0.250 vs greedy ref 0.250 |
| Q 1-hot 2 | Tri-3 | 2 | ✅ | 8.3 | Q=0.278 vs greedy ref 0.278 |
| Q 1-hot 2 | Tri-4 | 2 | ✅ | 54.6 | Q=0.375 vs greedy ref 0.375 |
| Q 1-hot 2 | Tri-8 | 2 | ❌ | 0.0 | needs 48 qubits > simulator limit 29 |
| Q 1-hot 2 | Tri-16 | 2 | ❌ | 0.0 | needs 96 qubits > simulator limit 29 |
| Q 1-hot 2 | Les Miserables | 2 | ❌ | 0.0 | needs 154 qubits > simulator limit 29 |
| Q 1-hot 2 | Karate Club | 2 | ❌ | 0.0 | needs 68 qubits > simulator limit 29 |
| Q 1-hot 2 | Davis Southern Women | 2 | ❌ | 0.0 | needs 64 qubits > simulator limit 29 |
| Cl 1-hot 4 | Tri-2 | 4 | ✅ | 2.6 | Q=0.250 vs greedy ref 0.250 |
| Cl 1-hot 4 | Tri-3 | 4 | ✅ | 8.0 | Q=0.417 vs greedy ref 0.417 |
| Cl 1-hot 4 | Tri-4 | 4 | ✅ | 17.9 | Q=0.500 vs greedy ref 0.500 |
| Cl 1-hot 4 | Tri-8 | 4 | ✅ | 88.7 | Q=0.594 vs greedy ref 0.625 |
| Cl 1-hot 4 | Tri-16 | 4 | ❌ | 600.0 | timeout after 600s |
| Cl 1-hot 4 | Les Miserables | 4 | ❌ | 600.0 | timeout after 600s |
| Cl 1-hot 4 | Karate Club | 4 | ✅ | 252.6 | Q=0.444 vs greedy ref 0.435 |
| Cl 1-hot 4 | Davis Southern Women | 4 | ✅ | 215.2 | Q=0.311 vs greedy ref 0.315 |
| Q 1-hot 4 | Tri-2 | 4 | ⚠️ | 63.0 | low modularity: Q=0.000 vs greedy ref 0.250 |
| Q 1-hot 4 | Tri-3 | 4 | ❌ | 0.0 | needs 36 qubits > simulator limit 29 |
| Q 1-hot 4 | Tri-4 | 4 | ❌ | 0.0 | needs 48 qubits > simulator limit 29 |
| Q 1-hot 4 | Tri-8 | 4 | ❌ | 0.0 | needs 96 qubits > simulator limit 29 |
| Q 1-hot 4 | Tri-16 | 4 | ❌ | 0.0 | needs 192 qubits > simulator limit 29 |
| Q 1-hot 4 | Les Miserables | 4 | ❌ | 0.0 | needs 308 qubits > simulator limit 29 |
| Q 1-hot 4 | Karate Club | 4 | ❌ | 0.0 | needs 136 qubits > simulator limit 29 |
| Q 1-hot 4 | Davis Southern Women | 4 | ❌ | 0.0 | needs 128 qubits > simulator limit 29 |
| Cl 1-hot 8 | Tri-2 | 8 | ✅ | 12.2 | Q=0.250 vs greedy ref 0.250 |
| Cl 1-hot 8 | Tri-3 | 8 | ✅ | 34.5 | Q=0.417 vs greedy ref 0.417 |
| Cl 1-hot 8 | Tri-4 | 8 | ⚠️ | 78.7 | low modularity: Q=0.404 vs greedy ref 0.500 |
| Cl 1-hot 8 | Tri-8 | 8 | ⚠️ | 596.0 | low modularity: Q=0.392 vs greedy ref 0.625 |
| Cl 1-hot 8 | Tri-16 | 8 | ❌ | 600.0 | timeout after 600s |
| Cl 1-hot 8 | Les Miserables | 8 | ❌ | 600.0 | timeout after 600s |
| Cl 1-hot 8 | Karate Club | 8 | ❌ | 600.0 | timeout after 600s |
| Cl 1-hot 8 | Davis Southern Women | 8 | ❌ | 600.0 | timeout after 600s |
| Q 1-hot 8 | Tri-2 | 8 | ❌ | 0.0 | needs 36 qubits > simulator limit 29 |
| Q 1-hot 8 | Tri-3 | 8 | ❌ | 0.0 | needs 72 qubits > simulator limit 29 |
| Q 1-hot 8 | Tri-4 | 8 | ❌ | 0.0 | needs 96 qubits > simulator limit 29 |
| Q 1-hot 8 | Tri-8 | 8 | ❌ | 0.0 | needs 192 qubits > simulator limit 29 |
| Q 1-hot 8 | Tri-16 | 8 | ❌ | 0.0 | needs 384 qubits > simulator limit 29 |
| Q 1-hot 8 | Les Miserables | 8 | ❌ | 0.0 | needs 616 qubits > simulator limit 29 |
| Q 1-hot 8 | Karate Club | 8 | ❌ | 0.0 | needs 272 qubits > simulator limit 29 |
| Q 1-hot 8 | Davis Southern Women | 8 | ❌ | 0.0 | needs 256 qubits > simulator limit 29 |
| Cl 1-hot 16 | Tri-2 | 16 | ⚠️ | 100.1 | low modularity: Q=0.117 vs greedy ref 0.250 |
| Cl 1-hot 16 | Tri-3 | 16 | ⚠️ | 277.1 | low modularity: Q=0.125 vs greedy ref 0.417 |
| Cl 1-hot 16 | Tri-4 | 16 | ❌ | 600.0 | timeout after 600s |
| Cl 1-hot 16 | Tri-8 | 16 | ❌ | 600.0 | timeout after 600s |
| Cl 1-hot 16 | Tri-16 | 16 | ❌ | 600.0 | timeout after 600s |
| Cl 1-hot 16 | Les Miserables | 16 | ❌ | 600.0 | timeout after 600s |
| Cl 1-hot 16 | Karate Club | 16 | ❌ | 600.0 | timeout after 600s |
| Cl 1-hot 16 | Davis Southern Women | 16 | ❌ | 600.0 | timeout after 600s |
| Q 1-hot 16 | Tri-2 | 16 | ❌ | 0.0 | needs 36 qubits > simulator limit 29 |
| Q 1-hot 16 | Tri-3 | 16 | ❌ | 0.0 | needs 81 qubits > simulator limit 29 |
| Q 1-hot 16 | Tri-4 | 16 | ❌ | 0.0 | needs 144 qubits > simulator limit 29 |
| Q 1-hot 16 | Tri-8 | 16 | ❌ | 0.0 | needs 384 qubits > simulator limit 29 |
| Q 1-hot 16 | Tri-16 | 16 | ❌ | 0.0 | needs 768 qubits > simulator limit 29 |
| Q 1-hot 16 | Les Miserables | 16 | ❌ | 0.0 | needs 1232 qubits > simulator limit 29 |
| Q 1-hot 16 | Karate Club | 16 | ❌ | 0.0 | needs 544 qubits > simulator limit 29 |
| Q 1-hot 16 | Davis Southern Women | 16 | ❌ | 0.0 | needs 512 qubits > simulator limit 29 |
| Cl H74 | Tri-2 | 2 | ✅ | 8.9 | Q=0.250 vs greedy ref 0.250 |
| Cl H74 | Tri-3 | 3 | ⚠️ | 11.0 | low modularity: Q=0.278 vs greedy ref 0.417 |
| Cl H74 | Tri-4 | 4 | ⚠️ | 13.1 | low modularity: Q=0.375 vs greedy ref 0.500 |
| Cl H74 | Tri-8 | 8 | ⚠️ | 22.4 | low modularity: Q=0.072 vs greedy ref 0.625 |
| Cl H74 | Tri-16 | 16 | ⚠️ | 41.2 | low modularity: Q=0.019 vs greedy ref 0.750 |
| Cl H74 | Les Miserables | 2 | ⚠️ | 63.8 | low modularity: Q=-0.049 vs greedy ref 0.381 |
| Cl H74 | Karate Club | 2 | ⚠️ | 30.6 | low modularity: Q=0.013 vs greedy ref 0.404 |
| Cl H74 | Davis Southern Women | 2 | ⚠️ | 28.6 | low modularity: Q=0.017 vs greedy ref 0.309 |
| Q H74 | Tri-2 | 2 | ❌ | 600.0 | timeout after 600s |
| Q H74 | Tri-3 | 3 | ❌ | 600.0 | timeout after 600s |
| Q H74 | Tri-4 | 4 | ❌ | 0.0 | needs 84 qubits > simulator limit 63 |
| Q H74 | Tri-8 | 8 | ❌ | 0.0 | needs 168 qubits > simulator limit 63 |
| Q H74 | Tri-16 | 16 | ❌ | 0.0 | needs 336 qubits > simulator limit 63 |
| Q H74 | Les Miserables | 2 | ❌ | 0.0 | needs 539 qubits > simulator limit 63 |
| Q H74 | Karate Club | 2 | ❌ | 0.0 | needs 238 qubits > simulator limit 63 |
| Q H74 | Davis Southern Women | 2 | ❌ | 0.0 | needs 224 qubits > simulator limit 63 |
| Cl H84 | Tri-2 | 2 | ✅ | 11.7 | Q=0.250 vs greedy ref 0.250 |
| Cl H84 | Tri-3 | 3 | ⚠️ | 14.8 | low modularity: Q=0.278 vs greedy ref 0.417 |
| Cl H84 | Tri-4 | 4 | ⚠️ | 18.4 | low modularity: Q=0.273 vs greedy ref 0.500 |
| Cl H84 | Tri-8 | 8 | ⚠️ | 32.3 | low modularity: Q=0.043 vs greedy ref 0.625 |
| Cl H84 | Tri-16 | 16 | ⚠️ | 60.4 | low modularity: Q=0.013 vs greedy ref 0.750 |
| Cl H84 | Les Miserables | 2 | ⚠️ | 95.9 | low modularity: Q=-0.064 vs greedy ref 0.381 |
| Cl H84 | Karate Club | 2 | ⚠️ | 44.7 | low modularity: Q=-0.083 vs greedy ref 0.404 |
| Cl H84 | Davis Southern Women | 2 | ⚠️ | 43.5 | low modularity: Q=-0.012 vs greedy ref 0.309 |
| Q H84 | Tri-2 | 2 | ❌ | 600.0 | timeout after 600s |
| Q H84 | Tri-3 | 3 | ❌ | 0.0 | needs 72 qubits > simulator limit 63 |
| Q H84 | Tri-4 | 4 | ❌ | 0.0 | needs 96 qubits > simulator limit 63 |
| Q H84 | Tri-8 | 8 | ❌ | 0.0 | needs 192 qubits > simulator limit 63 |
| Q H84 | Tri-16 | 16 | ❌ | 0.0 | needs 384 qubits > simulator limit 63 |
| Q H84 | Les Miserables | 2 | ❌ | 0.0 | needs 616 qubits > simulator limit 63 |
| Q H84 | Karate Club | 2 | ❌ | 0.0 | needs 272 qubits > simulator limit 63 |
| Q H84 | Davis Southern Women | 2 | ❌ | 0.0 | needs 256 qubits > simulator limit 63 |
