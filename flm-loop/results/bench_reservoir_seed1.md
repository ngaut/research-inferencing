| variant | graph | online NLL | tail NLL | probe NLL | probe acc | memory cap. | eff. rank | iters | residual | state RMS | s/token |
|---|---|---|---|---|---|---|---|---|---|---|---|
| direct | — | 3.382 | 3.261 | 3.120 | 0.221 | 1.11 | 22.8 | 0.00 | — | — | 0.000 |
| flm | malecns_v1.0 | 3.311 | 3.148 | 2.959 | 0.302 | 2.89 | 23.4 | 1.00 | 0.1263 | 0.098 | 0.191 |
| flm-rewired | malecns_v1.0-rewired1 | 3.402 | 3.257 | 3.069 | 0.259 | 2.34 | 21.5 | 1.00 | 0.1281 | 0.095 | 0.358 |
| loop-signed | malecns_v1.0 | 3.366 | 3.227 | 3.053 | 0.241 | 2.72 | 21.2 | 4.82 | 0.0007 | 0.020 | 1.533 |
| loop-signed-rewired | malecns_v1.0-rewired1 | 3.368 | 3.217 | 3.017 | 0.276 | 2.59 | 22.3 | 7.82 | 0.0019 | 0.092 | 1.576 |
