# PFD Simulation — Privileged Fee Discount Value Leakage

Simulation of value leakage in Uniswap V3 concentrated liquidity pools when a privileged arbitrageur receives a 0% fee discount on one pool.

## Overview

Two identical-TVL pools trade the same ETH/USDC pair:
- **Pool A** — variable price range (the pool with the fee discount)
- **Pool B** — fixed ±10% range (baseline)

Each trade goes through a **Triple-Step** process:

1. **Aggregator Routing** — Routes trade volume between Pool A and Pool B, optimizing for best execution. Precision N means the aggregator evaluates N+1 split options (0%, 1/N, 2/N, ..., 100% to Pool A).
2. **Standard Arbitrage** — A competitive arbitrageur equalizes prices between pools, paying the standard 0.3% fee on both pools. This represents normal MEV activity.
3. **Privileged Arbitrage** — A special actor arbitrages any remaining price spread, paying **0% fee on Pool A** and 0.3% on Pool B. This captures value that would otherwise remain with LPs.

The simulation compares LP fee revenue **with and without** the privileged arbitrage step to quantify the value leak.

## Key Results

### Realistic Routing (Random Precision 1-5, 10 seeds average)

$10M volume on $3M TVL pools, ±10% range:

| Metric | Value |
|---|---|
| LP Fees (no privileged arb) | $15,314 |
| LP Fees (with privileged arb) | $15,322 |
| Privileged Arb Profit | **$112** |
| Leak / LP Fees | **0.73%** |

### Worst-Case Routing (Precision 1 only)

Same parameters, single seed:

| Metric | Value |
|---|---|
| LP Fees (no privileged arb) | $16,118 |
| LP Fees (with privileged arb) | $16,776 |
| Privileged Arb Profit | **$720** |
| Leak / LP Fees | **4.29%** |

The leak is much larger with precision 1 (binary all-or-nothing routing) because dumb routing creates bigger price spreads between pools.

## Trade Distribution

Trades are sampled from empirical on-chain Uniswap V3 data:

| Size Bucket | Buy Weight | Sell Weight |
|---|---|---|
| $50 | 560,747 | 510,766 |
| $300 | 339,048 | 363,267 |
| $750 | 202,018 | 211,302 |
| $3,000 | 322,268 | 327,371 |
| $7,500 | 72,468 | 71,829 |
| $30,000 | 70,076 | 70,218 |
| $75,000 | 8,706 | 8,975 |
| $150,000 | 6,620 | 6,760 |

## Files

| File | Description |
|---|---|
| `simulation.py` | Core engine: pool math, aggregator, arb search, trade generation, scenario runner |
| `run_random_prec.py` | 10-seed test with random precision 1-5 per trade (realistic routing) |

## Running

```bash
python3 run_random_prec.py
```

## Technical Details

- Uses Python `Decimal` with 50-digit precision for swap math
- Uniswap V3 concentrated liquidity formulas: `sqrtPrice`, liquidity `L`, bounded ranges
- Brute-force arbitrage search in $1 USDC increments with early stopping
- Standard arb threshold: ~0.6% spread (two 0.3% fees), Privileged: ~0.3% (one 0.3% fee)
