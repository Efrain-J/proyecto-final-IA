# Q-learning tuning, preset 8h, year 2025

Fit Jan-Jun, validation Jul-Sep (patterns 2000..2004), test Oct-Dec only in Stage 3 (retrained on Jan-Sep). Score = Q-learning mean savings / tuned PriceThreshold mean savings on the same patterns.

## Stage 1 (one at a time, validation)

| config | mean score | worst seed | delta vs default | epochs run (mean) |
|---|---|---|---|---|
| default | 1.0855 | 1.0604 | +0.00% | 300 |
| alpha=0.05 | 1.0919 | 1.0515 | +0.59% | 300 |
| alpha=0.2 | 1.1041 | 1.0671 | +1.71% | 300 |
| alpha_end=0.005 | 1.1069 | 1.0767 | +1.97% | 300 |
| alpha_end=0.05 | 1.0655 | 1.0231 | -1.84% | 300 |
| gamma=0.95 | 0.6370 | 0.5616 | -41.32% | 300 |
| gamma=0.995 | 1.1359 | 1.1156 | +4.64% | 300 |
| epsilon_end=0.01 | 1.0653 | 1.0340 | -1.87% | 300 |
| epsilon_end=0.15 | 1.0560 | 1.0095 | -2.72% | 300 |
| max_epochs=600 | 1.0753 | 1.0242 | -0.95% | 600 |
| shift_draws=10 | 1.1133 | 1.1003 | +2.56% | 300 |

## Stage 2 (default/best grid, validation)

| config | mean score | worst seed | delta vs default | epochs run (mean) |
|---|---|---|---|---|
| default | 1.0855 | 1.0604 | +0.00% | 300 |
| alpha_end=0.005 | 1.1069 | 1.0767 | +1.97% | 300 |
| shift_draws=10 | 1.1133 | 1.1003 | +2.56% | 300 |
| alpha_end=0.005, shift_draws=10 | 1.1131 | 1.0735 | +2.54% | 300 |
| gamma=0.995 | 1.1359 | 1.1156 | +4.64% | 300 |
| alpha_end=0.005, gamma=0.995 | 1.1432 | 1.0981 | +5.31% | 300 |
| gamma=0.995, shift_draws=10 | 1.1455 | 1.1292 | +5.53% | 300 |
| alpha_end=0.005, gamma=0.995, shift_draws=10 | 1.1513 | 1.1103 | +6.06% | 300 |

## Chosen configuration

`alpha_end=0.005, gamma=0.995, shift_draws=10`: mean validation score 1.1513, +6.06% vs the default.

## Stage 3 (test, Oct-Dec)

| role | config | Q / best rule (mean) | Q / best rule per seed | Q / perfect | Q savings COP | shift schedule, threshold, night/day COP | perfect COP |
|---|---|---|---|---|---|---|---|
| default | default | 1.0288 | 1.074, 0.960, 1.056, 1.025 | 67.1% | 3,060,406 | 2,466,769, 2,974,674, 2,561,016 | 4,562,446 |
| tuned | alpha_end=0.005, gamma=0.995, shift_draws=10 | 1.0692 | 1.053, 1.047, 1.084, 1.093 | 69.7% | 3,180,602 | 2,466,769, 2,974,674, 2,561,016 | 4,562,446 |

## Recommendation

Validation gain +6.06% (needs >= +1%); test Q / best rule: tuned 1.0692 vs default 1.0288 (not worse). ADOPT the tuned configuration.
