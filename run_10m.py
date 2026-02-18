#!/usr/bin/env python3
"""Run the 12-scenario simulation with $10M trade volume on $10M TVL pools."""

from simulation import *
import sys

# Generate ONE trade sequence, shared across all scenarios
print("Generating trade sequence ($10M volume)...")
trades = generate_trades(total_volume=10_000_000, seed=42)
print(f"  {len(trades)} trades")
buy_count = sum(1 for s, _ in trades if s == 'buy')
sell_count = sum(1 for s, _ in trades if s == 'sell')
buy_vol = sum(float(v) for s, v in trades if s == 'buy')
sell_vol = sum(float(v) for s, v in trades if s == 'sell')
print(f"  Buy: {buy_count} trades, ${buy_vol:,.0f}")
print(f"  Sell: {sell_count} trades, ${sell_vol:,.0f}")
print()

# Scenario matrix
ranges = [
    ('±5%',  D('1900'), D('2100')),
    ('±10%', D('1800'), D('2200')),
    ('±15%', D('1700'), D('2300')),
]
precisions = [1, 10]
tvl = D('10000000')

# Run all 12 scenarios
results = {}  # key: (range, precision, mode) -> result dict

for range_name, p_low, p_high in ranges:
    for prec in precisions:
        for mode in ['standard', 'privileged']:
            with_priv = (mode == 'privileged')
            label = f"{range_name} / Prec {prec:>2} / {mode}"
            print(f"Running: {label} ...", end='', flush=True)

            res = run_scenario(
                pool_a_range=(p_low, p_high),
                precision=prec,
                with_privileged=with_priv,
                trades=trades,
                tvl=tvl,
            )
            results[(range_name, prec, mode)] = res
            print(f" done (Pool A vol: {res['volume_a_pct']:.1f}%)")

# Build the comparison table
print()
print("=" * 120)
print(f"RESULTS: $10M Volume on $10M TVL Pools  |  {len(trades)} trades  |  Seed=42")
print("=" * 120)
print(f"{'Scenario':<22} "
      f"{'Fees(Std)':>12} {'Fees(Priv)':>12} {'Fee Delta':>10} "
      f"{'Priv Profit':>12} {'Leak/Fees%':>10} "
      f"{'Vol A%(S)':>9} {'Vol A%(P)':>9}")
print("-" * 120)

for range_name, _, _ in ranges:
    for prec in precisions:
        std = results[(range_name, prec, 'standard')]
        priv = results[(range_name, prec, 'privileged')]

        fees_std = std['agg_fees_a'] + std['arb_fees_a']
        fees_priv = priv['agg_fees_a'] + priv['arb_fees_a']
        delta = fees_priv - fees_std
        leak = priv['priv_profit']
        leak_pct = (leak / fees_priv * 100) if fees_priv > 0 else 0

        label = f"{range_name} / Prec {prec}"
        print(f"{label:<22} "
              f"${fees_std:>10,.2f} ${fees_priv:>10,.2f} ${delta:>8,.2f} "
              f"${leak:>10,.2f} {leak_pct:>9.2f}% "
              f"{std['volume_a_pct']:>8.1f}% {priv['volume_a_pct']:>8.1f}%")

# Breakdown table
print()
print("=" * 120)
print("DETAILED BREAKDOWN")
print("=" * 120)
print(f"{'Scenario':<22} {'Mode':<12} "
      f"{'Agg Fees A':>12} {'Arb Fees A':>12} {'Total Fees':>12} "
      f"{'Priv Profit':>12} {'Vol A %':>8} {'Final P_A':>10} {'Final P_B':>10}")
print("-" * 120)

for range_name, _, _ in ranges:
    for prec in precisions:
        for mode in ['standard', 'privileged']:
            r = results[(range_name, prec, mode)]
            total = r['agg_fees_a'] + r['arb_fees_a']
            label = f"{range_name} / Prec {prec}"
            print(f"{label:<22} {mode:<12} "
                  f"${r['agg_fees_a']:>10,.2f} ${r['arb_fees_a']:>10,.2f} ${total:>10,.2f} "
                  f"${r['priv_profit']:>10,.2f} "
                  f"{r['volume_a_pct']:>7.1f}% "
                  f"${r['final_price_a']:>9,.2f} ${r['final_price_b']:>9,.2f}")
        print()
