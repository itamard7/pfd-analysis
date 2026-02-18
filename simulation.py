#!/usr/bin/env python3
"""
High-Resolution Atomic Arbitrage Simulation (Privileged Path)

Simulates value leakage in Uniswap V3 pools by comparing:
- Standard market activity (0.3% fee) with competitive arbitrage
- Privileged arbitrage sweep (0% fee on Pool A)

Scenario Matrix: 3 ranges x 2 precisions x 2 modes = 12 scenarios
"""

from decimal import Decimal, getcontext
import random
import sys

getcontext().prec = 50

D = Decimal
ZERO = D('0')
ONE = D('1')

# ============================================================
# Constants
# ============================================================
INITIAL_PRICE = D('2000')
INITIAL_SQRT_PRICE = INITIAL_PRICE.sqrt()
STANDARD_FEE = D('0.003')   # 0.3%
ZERO_FEE = D('0')
USDC_STEP = D('1')           # $1 brute-force increment
MAX_ARB_SEARCH = 500_000     # safety cap on brute-force iterations

# ============================================================
# Trade Distribution (from on-chain data)
# ============================================================
TRADE_BUCKETS = [
    # (side, midpoint_value, weight)
    ('buy',  D('50'),     560747),
    ('buy',  D('300'),    339048),
    ('buy',  D('750'),    202018),
    ('buy',  D('3000'),   322268),
    ('buy',  D('7500'),   72468),
    ('buy',  D('30000'),  70076),
    ('buy',  D('75000'),  8706),
    ('buy',  D('150000'), 6620),
    ('sell', D('50'),     510766),
    ('sell', D('300'),    363267),
    ('sell', D('750'),    211302),
    ('sell', D('3000'),   327371),
    ('sell', D('7500'),   71829),
    ('sell', D('30000'),  70218),
    ('sell', D('75000'),  8975),
    ('sell', D('150000'), 6760),
]


# ============================================================
# Pure swap simulation functions (no side effects)
# ============================================================
def sim_buy_eth(sqrt_p, L, fee, sqrt_p_upper, usdc_in):
    """
    Simulate buying ETH with USDC (price goes UP).
    Returns (eth_out, new_sqrt_p, fee_usdc).
    """
    if usdc_in <= ZERO:
        return ZERO, sqrt_p, ZERO

    fee_amt = usdc_in * fee
    effective = usdc_in - fee_amt

    new_sqrt_p = sqrt_p + effective / L

    # Cap at upper bound
    if new_sqrt_p > sqrt_p_upper:
        max_effective = (sqrt_p_upper - sqrt_p) * L
        if max_effective <= ZERO:
            return ZERO, sqrt_p, ZERO
        actual_in = max_effective / (ONE - fee)
        fee_amt = actual_in * fee
        effective = max_effective
        new_sqrt_p = sqrt_p_upper

    eth_out = L * (ONE / sqrt_p - ONE / new_sqrt_p)
    return eth_out, new_sqrt_p, fee_amt


def sim_sell_eth(sqrt_p, L, fee, sqrt_p_lower, eth_in):
    """
    Simulate selling ETH for USDC (price goes DOWN).
    Returns (usdc_out, new_sqrt_p, fee_eth).
    """
    if eth_in <= ZERO:
        return ZERO, sqrt_p, ZERO

    fee_amt = eth_in * fee
    effective = eth_in - fee_amt

    inv_new = ONE / sqrt_p + effective / L
    new_sqrt_p = ONE / inv_new

    # Cap at lower bound
    if new_sqrt_p < sqrt_p_lower:
        max_effective = (ONE / sqrt_p_lower - ONE / sqrt_p) * L
        if max_effective <= ZERO:
            return ZERO, sqrt_p, ZERO
        actual_in = max_effective / (ONE - fee)
        fee_amt = actual_in * fee
        effective = max_effective
        new_sqrt_p = sqrt_p_lower

    usdc_out = L * (sqrt_p - new_sqrt_p)
    return usdc_out, new_sqrt_p, fee_amt


# ============================================================
# Pool class (tracks mutable state)
# ============================================================
class Pool:
    def __init__(self, name, sqrt_p_lower, sqrt_p_upper, liquidity, fee_rate):
        self.name = name
        self.sqrt_p = INITIAL_SQRT_PRICE  # current sqrt(price)
        self.sqrt_p_lower = sqrt_p_lower
        self.sqrt_p_upper = sqrt_p_upper
        self.L = liquidity
        self.fee_rate = fee_rate
        self.fees_usdc = ZERO   # cumulative USDC fees
        self.fees_eth = ZERO    # cumulative ETH fees

    def price(self):
        return self.sqrt_p ** 2

    def copy(self):
        p = Pool.__new__(Pool)
        p.name = self.name
        p.sqrt_p = self.sqrt_p
        p.sqrt_p_lower = self.sqrt_p_lower
        p.sqrt_p_upper = self.sqrt_p_upper
        p.L = self.L
        p.fee_rate = self.fee_rate
        p.fees_usdc = self.fees_usdc
        p.fees_eth = self.fees_eth
        return p

    def buy_eth(self, usdc_in, fee_override=None):
        """Execute buy ETH swap. Returns eth_out."""
        fee = self.fee_rate if fee_override is None else fee_override
        eth_out, new_sqrt_p, fee_amt = sim_buy_eth(
            self.sqrt_p, self.L, fee, self.sqrt_p_upper, usdc_in
        )
        self.sqrt_p = new_sqrt_p
        self.fees_usdc += fee_amt
        return eth_out

    def sell_eth(self, eth_in, fee_override=None):
        """Execute sell ETH swap. Returns usdc_out."""
        fee = self.fee_rate if fee_override is None else fee_override
        usdc_out, new_sqrt_p, fee_amt = sim_sell_eth(
            self.sqrt_p, self.L, fee, self.sqrt_p_lower, eth_in
        )
        self.sqrt_p = new_sqrt_p
        self.fees_eth += fee_amt
        return usdc_out

    def __repr__(self):
        return (f"Pool({self.name}: price={float(self.price()):.4f}, "
                f"sqrtP={float(self.sqrt_p):.8f}, "
                f"fees_usdc={float(self.fees_usdc):.4f}, "
                f"fees_eth={float(self.fees_eth):.8f})")


# ============================================================
# Liquidity calculation
# ============================================================
def compute_liquidity(price, price_lower, price_upper, tvl):
    """
    Compute L for a concentrated position given TVL at current price.

    Real reserves:
      USDC = L * (sqrt(P) - sqrt(P_lower))
      ETH  = L * (1/sqrt(P) - 1/sqrt(P_upper))
      TVL  = USDC + ETH * P
    """
    sqrt_p = price.sqrt()
    sqrt_pl = price_lower.sqrt()
    sqrt_pu = price_upper.sqrt()

    # TVL = L * [sqrt(P) - sqrt(Pl) + P * (1/sqrt(P) - 1/sqrt(Pu))]
    #      = L * [sqrt(P) - sqrt(Pl) + sqrt(P) - P/sqrt(Pu)]
    #      = L * [2*sqrt(P) - sqrt(Pl) - P/sqrt(Pu)]
    factor = 2 * sqrt_p - sqrt_pl - price / sqrt_pu
    return tvl / factor


def make_pool(name, price_lower, price_upper, tvl, fee_rate):
    """Create a Pool with computed liquidity."""
    sqrt_pl = price_lower.sqrt()
    sqrt_pu = price_upper.sqrt()
    L = compute_liquidity(INITIAL_PRICE, price_lower, price_upper, tvl)
    return Pool(name, sqrt_pl, sqrt_pu, L, fee_rate)


# ============================================================
# Aggregator: route trade between two pools
# ============================================================
def aggregator_execute(side, value_usdc, pool_a, pool_b, precision):
    """
    Route a trade between Pool A and Pool B.

    precision=N:  N+1 fractions from 0 to 1 (e.g., N=1 -> [0,1], N=5 -> [0,0.2,...,1])

    Modifies pool_a and pool_b in place.
    Returns (volume_to_a_usdc, volume_to_b_usdc).
    """
    fractions = [D(i) / D(precision) for i in range(precision + 1)]

    best_output = D('-1')
    best_frac = ZERO

    for frac_a in fractions:
        frac_b = ONE - frac_a
        val_a = value_usdc * frac_a
        val_b = value_usdc * frac_b

        if side == 'buy':
            # Input is USDC, output is ETH
            eth_a, _, _ = sim_buy_eth(
                pool_a.sqrt_p, pool_a.L, pool_a.fee_rate,
                pool_a.sqrt_p_upper, val_a
            )
            eth_b, _, _ = sim_buy_eth(
                pool_b.sqrt_p, pool_b.L, pool_b.fee_rate,
                pool_b.sqrt_p_upper, val_b
            )
            total = eth_a + eth_b
        else:
            # Input is ETH (convert value_usdc to ETH at current mid-price)
            mid_price = (pool_a.price() + pool_b.price()) / 2
            eth_total = value_usdc / mid_price
            eth_a = eth_total * frac_a
            eth_b = eth_total * frac_b
            usdc_a, _, _ = sim_sell_eth(
                pool_a.sqrt_p, pool_a.L, pool_a.fee_rate,
                pool_a.sqrt_p_lower, eth_a
            )
            usdc_b, _, _ = sim_sell_eth(
                pool_b.sqrt_p, pool_b.L, pool_b.fee_rate,
                pool_b.sqrt_p_lower, eth_b
            )
            total = usdc_a + usdc_b

        if total > best_output:
            best_output = total
            best_frac = frac_a

    # Execute the best split
    frac_a = best_frac
    frac_b = ONE - frac_a

    if side == 'buy':
        val_a = value_usdc * frac_a
        val_b = value_usdc * frac_b
        if val_a > ZERO:
            pool_a.buy_eth(val_a)
        if val_b > ZERO:
            pool_b.buy_eth(val_b)
        return float(val_a), float(val_b)
    else:
        mid_price = (pool_a.price() + pool_b.price()) / 2
        eth_total = value_usdc / mid_price
        eth_a = eth_total * frac_a
        eth_b = eth_total * frac_b
        if eth_a > ZERO:
            pool_a.sell_eth(eth_a)
        if eth_b > ZERO:
            pool_b.sell_eth(eth_b)
        # Return USDC-equivalent volume
        return float(value_usdc * frac_a), float(value_usdc * frac_b)


# ============================================================
# Arbitrage engine (brute-force $1 USDC steps)
# ============================================================
def find_and_execute_arb(pool_a, pool_b, fee_a_override=None, fee_b_override=None):
    """
    Find and execute optimal arbitrage between two pools.

    fee_a_override / fee_b_override: if set, use this fee for simulation
    and execution (for privileged arb).

    Returns: (profit_usdc, direction_str)
    """
    price_a = pool_a.price()
    price_b = pool_b.price()

    fee_a = fee_a_override if fee_a_override is not None else pool_a.fee_rate
    fee_b = fee_b_override if fee_b_override is not None else pool_b.fee_rate

    # Determine direction
    if price_a > price_b:
        # Buy ETH in B (cheap), sell in A (expensive)
        cheap_sqrt_p = pool_b.sqrt_p
        cheap_L = pool_b.L
        cheap_fee = fee_b
        cheap_upper = pool_b.sqrt_p_upper
        exp_sqrt_p = pool_a.sqrt_p
        exp_L = pool_a.L
        exp_fee = fee_a
        exp_lower = pool_a.sqrt_p_lower
        direction = 'B->A'
    elif price_b > price_a:
        # Buy ETH in A (cheap), sell in B (expensive)
        cheap_sqrt_p = pool_a.sqrt_p
        cheap_L = pool_a.L
        cheap_fee = fee_a
        cheap_upper = pool_a.sqrt_p_upper
        exp_sqrt_p = pool_b.sqrt_p
        exp_L = pool_b.L
        exp_fee = fee_b
        exp_lower = pool_b.sqrt_p_lower
        direction = 'A->B'
    else:
        return ZERO, 'none'

    # Quick check: is even $1 profitable?
    eth_got, _, _ = sim_buy_eth(cheap_sqrt_p, cheap_L, cheap_fee, cheap_upper, USDC_STEP)
    usdc_got, _, _ = sim_sell_eth(exp_sqrt_p, exp_L, exp_fee, exp_lower, eth_got)
    if usdc_got <= USDC_STEP:
        return ZERO, 'none'

    # Brute-force search for optimal dX
    best_profit = ZERO
    best_dx = ZERO
    was_positive = False

    for dx_int in range(1, MAX_ARB_SEARCH + 1):
        dx = D(dx_int)

        eth_got, new_cheap_sqrt_p, _ = sim_buy_eth(
            cheap_sqrt_p, cheap_L, cheap_fee, cheap_upper, dx
        )
        usdc_got, new_exp_sqrt_p, _ = sim_sell_eth(
            exp_sqrt_p, exp_L, exp_fee, exp_lower, eth_got
        )
        profit = usdc_got - dx

        if profit > best_profit:
            best_profit = profit
            best_dx = dx
            was_positive = True
        elif was_positive and profit <= ZERO:
            break

    if best_profit <= ZERO:
        return ZERO, 'none'

    # Execute the optimal trade
    if direction == 'B->A':
        eth_mid = pool_b.buy_eth(best_dx, fee_override=fee_b_override)
        usdc_out = pool_a.sell_eth(eth_mid, fee_override=fee_a_override)
        actual_profit = usdc_out - best_dx
    else:  # A->B
        eth_mid = pool_a.buy_eth(best_dx, fee_override=fee_a_override)
        usdc_out = pool_b.sell_eth(eth_mid, fee_override=fee_b_override)
        actual_profit = usdc_out - best_dx

    return actual_profit, direction


# ============================================================
# Trade generation
# ============================================================
def generate_trades(total_volume=1_000_000, seed=42):
    """Generate random trade sequence from the empirical distribution."""
    rng = random.Random(seed)

    sides = [b[0] for b in TRADE_BUCKETS]
    midpoints = [b[1] for b in TRADE_BUCKETS]
    weights = [b[2] for b in TRADE_BUCKETS]

    trades = []
    cumulative = ZERO
    target = D(str(total_volume))

    while cumulative < target:
        idx = rng.choices(range(len(TRADE_BUCKETS)), weights=weights, k=1)[0]
        side = sides[idx]
        value = midpoints[idx]

        remaining = target - cumulative
        if value > remaining:
            value = remaining
        if value < ONE:
            break

        trades.append((side, value))
        cumulative += value

    return trades


# ============================================================
# Single-trade test (verification)
# ============================================================
def run_test_trade():
    """Run one $50,000 buy trade and print pool states after each step.

    Uses $500K TVL to create visible arb dynamics (with $10M TVL
    a $50K trade only creates ~0.05% spread, below arb thresholds).
    """
    test_tvl = D('500000')
    print("=" * 70)
    print("TEST: Single $50,000 Buy-ETH Trade")
    print(f"Pool A: ±5% range, Pool B: ±10% range, Precision 1, TVL=${int(test_tvl):,}")
    print("=" * 70)

    pool_a = make_pool('A', D('1900'), D('2100'), test_tvl, STANDARD_FEE)
    pool_b = make_pool('B', D('1800'), D('2200'), test_tvl, STANDARD_FEE)

    print(f"\n--- Initial State ---")
    print(f"  {pool_a}")
    print(f"  {pool_b}")
    print(f"  Pool A liquidity L = {float(pool_a.L):,.2f}")
    print(f"  Pool B liquidity L = {float(pool_b.L):,.2f}")

    # Step 1: Aggregator
    trade_value = D('50000')
    usdc_f0 = pool_a.fees_usdc; eth_f0 = pool_a.fees_eth

    vol_a, vol_b = aggregator_execute('buy', trade_value, pool_a, pool_b, precision=1)
    spread = float((pool_a.price() - pool_b.price()) / pool_b.price() * 100)

    agg_usdc = pool_a.fees_usdc - usdc_f0
    agg_eth = pool_a.fees_eth - eth_f0

    print(f"\n--- After Step 1: Aggregator (Buy $50,000 ETH) ---")
    print(f"  Routed: ${vol_a:,.2f} to A, ${vol_b:,.2f} to B")
    print(f"  {pool_a}")
    print(f"  {pool_b}")
    print(f"  Price spread: {spread:.4f}%")
    print(f"  Pool A fees this step: {float(agg_usdc):.4f} USDC + {float(agg_eth):.8f} ETH")

    # Step 2: Standard Arbitrage
    usdc_f1 = pool_a.fees_usdc; eth_f1 = pool_a.fees_eth

    profit_std, dir_std = find_and_execute_arb(pool_a, pool_b)
    spread2 = float((pool_a.price() - pool_b.price()) / pool_b.price() * 100)

    arb_usdc = pool_a.fees_usdc - usdc_f1
    arb_eth = pool_a.fees_eth - eth_f1

    print(f"\n--- After Step 2: Standard Arbitrage ---")
    print(f"  Arb direction: {dir_std}, profit: ${float(profit_std):.4f}")
    print(f"  {pool_a}")
    print(f"  {pool_b}")
    print(f"  Residual spread: {spread2:.4f}%")
    print(f"  Pool A fees this step: {float(arb_usdc):.4f} USDC + {float(arb_eth):.8f} ETH")

    # Step 3: Privileged Arbitrage (0% fee on Pool A)
    profit_priv, dir_priv = find_and_execute_arb(
        pool_a, pool_b, fee_a_override=ZERO_FEE
    )
    spread3 = float((pool_a.price() - pool_b.price()) / pool_b.price() * 100)

    print(f"\n--- After Step 3: Privileged Arbitrage (Pool A 0% fee) ---")
    print(f"  Arb direction: {dir_priv}, profit: ${float(profit_priv):.4f}")
    print(f"  {pool_a}")
    print(f"  {pool_b}")
    print(f"  Residual spread: {spread3:.4f}%")
    print(f"  Privileged Value Captured: ${float(profit_priv):.4f}")

    # Convert ETH fees at final price for summary
    fp = pool_a.price()
    print(f"\n--- Fee Summary (ETH fees converted at final price {float(fp):.2f}) ---")
    print(f"  Pool A aggregator fees:  ${float(agg_usdc + agg_eth * fp):.4f}")
    print(f"  Pool A std-arb fees:     ${float(arb_usdc + arb_eth * fp):.4f}")
    print(f"  Privileged profit:       ${float(profit_priv):.4f}")
    print()


# ============================================================
# Full scenario runner
# ============================================================
def run_scenario(pool_a_range, precision, with_privileged, trades, tvl=D('10000000')):
    """
    Run a full simulation.

    pool_a_range: (price_lower, price_upper) for Pool A
    precision: int (single value for all trades) or list of ints (per-trade)
    with_privileged: if True, run Step 3 after Step 2
    trades: list of (side, value_usdc) tuples

    Returns dict with results.
    """
    # Support per-trade precisions
    if isinstance(precision, list):
        per_trade_prec = precision
    else:
        per_trade_prec = None
        fixed_prec = precision
    price_lower_a, price_upper_a = pool_a_range
    pool_a = make_pool('A', price_lower_a, price_upper_a, tvl, STANDARD_FEE)
    pool_b = make_pool('B', D('1800'), D('2200'), tvl, STANDARD_FEE)

    total_volume_a = 0.0
    total_volume_b = 0.0
    # Track raw fee deltas (USDC and ETH separately) to avoid
    # price-revaluation contamination
    agg_fees_usdc = ZERO
    agg_fees_eth = ZERO
    arb_fees_usdc = ZERO
    arb_fees_eth = ZERO
    total_priv_profit = ZERO

    num_trades = len(trades)

    for i, (side, value) in enumerate(trades):
        # Snapshot raw fees before aggregator
        usdc_before = pool_a.fees_usdc
        eth_before = pool_a.fees_eth

        # Step 1: Aggregator
        prec_i = per_trade_prec[i] if per_trade_prec is not None else fixed_prec
        vol_a, vol_b = aggregator_execute(side, value, pool_a, pool_b, prec_i)
        total_volume_a += vol_a
        total_volume_b += vol_b

        agg_fees_usdc += (pool_a.fees_usdc - usdc_before)
        agg_fees_eth += (pool_a.fees_eth - eth_before)

        # Step 2: Standard Arbitrage
        usdc_before = pool_a.fees_usdc
        eth_before = pool_a.fees_eth

        profit_std, _ = find_and_execute_arb(pool_a, pool_b)

        arb_fees_usdc += (pool_a.fees_usdc - usdc_before)
        arb_fees_eth += (pool_a.fees_eth - eth_before)

        # Step 3: Privileged Arbitrage (optional)
        if with_privileged:
            profit_priv, _ = find_and_execute_arb(
                pool_a, pool_b, fee_a_override=ZERO_FEE
            )
            total_priv_profit += profit_priv

        # Progress indicator for long runs
        if (i + 1) % 50 == 0:
            print(f"    ... processed {i+1}/{num_trades} trades", file=sys.stderr)

    total_vol = total_volume_a + total_volume_b
    pct_a = (total_volume_a / total_vol * 100) if total_vol > 0 else 0

    # Convert ETH fees to USDC at final pool price
    final_price = pool_a.price()
    total_agg_fees_a = float(agg_fees_usdc + agg_fees_eth * final_price)
    total_arb_fees_a = float(arb_fees_usdc + arb_fees_eth * final_price)

    return {
        'agg_fees_a': total_agg_fees_a,
        'agg_fees_a_usdc': float(agg_fees_usdc),
        'agg_fees_a_eth': float(agg_fees_eth),
        'arb_fees_a': total_arb_fees_a,
        'arb_fees_a_usdc': float(arb_fees_usdc),
        'arb_fees_a_eth': float(arb_fees_eth),
        'priv_profit': float(total_priv_profit),
        'volume_a_pct': pct_a,
        'volume_a': total_volume_a,
        'volume_b': total_volume_b,
        'final_price_a': float(pool_a.price()),
        'final_price_b': float(pool_b.price()),
    }


# ============================================================
# Main
# ============================================================
def print_results_table(results, title="RESULTS SUMMARY"):
    """Print formatted results table."""
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    header = (f"{'Scenario':<52} {'Agg Fees A':>12} {'Arb Fees A':>12} "
              f"{'Priv Profit':>12} {'Vol A %':>8} {'Final P_A':>10}")
    print(header)
    print("-" * 110)

    for r in results:
        print(f"{r['label']:<52} "
              f"${r['agg_fees_a']:>10,.2f} "
              f"${r['arb_fees_a']:>10,.2f} "
              f"${r['priv_profit']:>10,.2f} "
              f"{r['volume_a_pct']:>7.1f}% "
              f"${r['final_price_a']:>9,.2f}")


def print_leak_analysis(results, ranges, precisions):
    """Print comparative leak analysis."""
    print("\n" + "=" * 110)
    print("VALUE LEAK ANALYSIS (Privileged Profit by Scenario)")
    print("=" * 110)

    for range_name, _, _ in ranges:
        for prec in precisions:
            std = next((r for r in results
                       if r['range'] == range_name and r['precision'] == prec
                       and r['mode'] == 'standard'), None)
            priv = next((r for r in results
                        if r['range'] == range_name and r['precision'] == prec
                        and r['mode'] == 'privileged'), None)

            if std is None or priv is None:
                continue

            total_fees_std = std['agg_fees_a'] + std['arb_fees_a']
            total_fees_priv = priv['agg_fees_a'] + priv['arb_fees_a']
            leak = priv['priv_profit']
            leak_pct = (leak / total_fees_priv * 100) if total_fees_priv > 0 else 0

            print(f"  {range_name} / Prec {prec:>2}: "
                  f"LP Fees(std)=${total_fees_std:,.2f}  "
                  f"LP Fees(priv)=${total_fees_priv:,.2f}  "
                  f"Value Leak=${leak:,.2f}  "
                  f"Leak/Fees={leak_pct:.2f}%")


def main():
    # --- Test first ---
    run_test_trade()

    # --- Generate trades ---
    print("Generating trade sequence...")
    trades = generate_trades(total_volume=1_000_000, seed=42)
    print(f"Generated {len(trades)} trades totaling $1,000,000")

    buy_count = sum(1 for s, _ in trades if s == 'buy')
    sell_count = sum(1 for s, _ in trades if s == 'sell')
    print(f"  Buy trades: {buy_count}, Sell trades: {sell_count}")

    # Show trade distribution summary
    buy_vol = sum(float(v) for s, v in trades if s == 'buy')
    sell_vol = sum(float(v) for s, v in trades if s == 'sell')
    print(f"  Buy volume: ${buy_vol:,.0f}, Sell volume: ${sell_vol:,.0f}")
    print()

    # --- Scenario Matrix: 3 ranges × 2 precisions × 2 modes = 12 ---
    ranges = [
        ('±5%',  D('1900'), D('2100')),
        ('±10%', D('1800'), D('2200')),
        ('±15%', D('1700'), D('2300')),
    ]
    precisions = [1, 10]
    tvl = D('10000000')

    results = []

    for range_name, p_low, p_high in ranges:
        for prec in precisions:
            for mode in ['standard', 'privileged']:
                with_priv = (mode == 'privileged')
                label = f"TVL $10M | {range_name} | Prec {prec:>2} | {mode.capitalize()}"
                print(f"Running: {label} ...")

                res = run_scenario(
                    pool_a_range=(p_low, p_high),
                    precision=prec,
                    with_privileged=with_priv,
                    trades=trades,
                    tvl=tvl,
                )
                res['label'] = label
                res['range'] = range_name
                res['precision'] = prec
                res['mode'] = mode
                res['tvl'] = '$10M'
                results.append(res)

                print(f"  Done. Pool A vol: {res['volume_a_pct']:.1f}%, "
                      f"Priv profit: ${res['priv_profit']:.2f}")

    print_results_table(results, "RESULTS: $10M TVL (Original Spec)")
    print_leak_analysis(results, ranges, precisions)

    # --- TVL Sensitivity: Run with smaller TVLs to show scaling ---
    print("\n\n" + "#" * 110)
    print("# TVL SENSITIVITY ANALYSIS")
    print("# (Same trade sequence, varying TVL to show arb dynamics at different depths)")
    print("#" * 110)

    tvl_levels = [D('1000000'), D('500000')]
    sensitivity_results = {}

    for tvl_val in tvl_levels:
        tvl_label = f"${int(tvl_val/1000)}K"
        sensitivity_results[tvl_label] = []

        for range_name, p_low, p_high in ranges:
            for prec in precisions:
                for mode in ['standard', 'privileged']:
                    with_priv = (mode == 'privileged')
                    label = f"TVL {tvl_label} | {range_name} | Prec {prec:>2} | {mode.capitalize()}"
                    print(f"Running: {label} ...")

                    res = run_scenario(
                        pool_a_range=(p_low, p_high),
                        precision=prec,
                        with_privileged=with_priv,
                        trades=trades,
                        tvl=tvl_val,
                    )
                    res['label'] = label
                    res['range'] = range_name
                    res['precision'] = prec
                    res['mode'] = mode
                    res['tvl'] = tvl_label
                    sensitivity_results[tvl_label].append(res)

                    print(f"  Done. Priv profit: ${res['priv_profit']:.2f}")

        print_results_table(sensitivity_results[tvl_label],
                            f"RESULTS: TVL {tvl_label}")
        print_leak_analysis(sensitivity_results[tvl_label], ranges, precisions)

    # --- Cross-TVL comparison: Standard vs Privileged ---
    all_tvl_results = [('$10M', results)]
    for tvl_label, res_list in sensitivity_results.items():
        all_tvl_results.append((tvl_label, res_list))

    # Table 1: Standard (Step 1 + Step 2 only)
    print("\n\n" + "=" * 110)
    print("CROSS-TVL COMPARISON TABLE 1: STANDARD (Aggregator + Competitive Arb only)")
    print("=" * 110)
    print(f"{'TVL':<10} {'Range':<8} {'Prec':>5} {'Agg Fees A':>12} {'Arb Fees A':>12} "
          f"{'Total Fees A':>13} {'Vol A %':>8} {'Final P_A':>10}")
    print("-" * 82)

    for tvl_label, res_list in all_tvl_results:
        for r in res_list:
            if r['mode'] != 'standard':
                continue
            total_fees = r['agg_fees_a'] + r['arb_fees_a']
            print(f"{tvl_label:<10} {r['range']:<8} {r['precision']:>5} "
                  f"${r['agg_fees_a']:>10,.2f} "
                  f"${r['arb_fees_a']:>10,.2f} "
                  f"${total_fees:>11,.2f} "
                  f"{r['volume_a_pct']:>7.1f}% "
                  f"${r['final_price_a']:>9,.2f}")
        if tvl_label != all_tvl_results[-1][0]:
            print()

    # Table 2: Privileged (Step 1 + Step 2 + Step 3)
    print("\n" + "=" * 110)
    print("CROSS-TVL COMPARISON TABLE 2: PRIVILEGED (Aggregator + Competitive Arb + Privileged Arb)")
    print("=" * 110)
    print(f"{'TVL':<10} {'Range':<8} {'Prec':>5} {'Agg Fees A':>12} {'Arb Fees A':>12} "
          f"{'Priv Profit':>12} {'Leak/Fees':>10} {'Vol A %':>8} {'Final P_A':>10}")
    print("-" * 96)

    for tvl_label, res_list in all_tvl_results:
        for r in res_list:
            if r['mode'] != 'privileged':
                continue
            total_fees = r['agg_fees_a'] + r['arb_fees_a']
            leak_pct = (r['priv_profit'] / total_fees * 100) if total_fees > 0 else 0
            print(f"{tvl_label:<10} {r['range']:<8} {r['precision']:>5} "
                  f"${r['agg_fees_a']:>10,.2f} "
                  f"${r['arb_fees_a']:>10,.2f} "
                  f"${r['priv_profit']:>10,.2f} "
                  f"{leak_pct:>9.2f}% "
                  f"{r['volume_a_pct']:>7.1f}% "
                  f"${r['final_price_a']:>9,.2f}")
        if tvl_label != all_tvl_results[-1][0]:
            print()

    # Table 3: Side-by-side delta
    print("\n" + "=" * 110)
    print("CROSS-TVL COMPARISON TABLE 3: STANDARD vs PRIVILEGED (Delta)")
    print("=" * 110)
    print(f"{'TVL':<10} {'Range':<8} {'Prec':>5} {'Fees(Std)':>12} {'Fees(Priv)':>12} "
          f"{'Fee Delta':>10} {'Priv Profit':>12} {'Leak/Fees':>10}")
    print("-" * 85)

    for tvl_label, res_list in all_tvl_results:
        for range_name in ['±5%', '±10%', '±15%']:
            for prec in precisions:
                std = next((r for r in res_list
                            if r['range'] == range_name and r['precision'] == prec
                            and r['mode'] == 'standard'), None)
                priv = next((r for r in res_list
                             if r['range'] == range_name and r['precision'] == prec
                             and r['mode'] == 'privileged'), None)
                if std is None or priv is None:
                    continue

                fees_std = std['agg_fees_a'] + std['arb_fees_a']
                fees_priv = priv['agg_fees_a'] + priv['arb_fees_a']
                fee_delta = fees_priv - fees_std
                leak = priv['priv_profit']
                leak_pct = (leak / fees_priv * 100) if fees_priv > 0 else 0

                print(f"{tvl_label:<10} {range_name:<8} {prec:>5} "
                      f"${fees_std:>10,.2f} "
                      f"${fees_priv:>10,.2f} "
                      f"${fee_delta:>8,.2f} "
                      f"${leak:>10,.2f} "
                      f"{leak_pct:>9.2f}%")
        if tvl_label != all_tvl_results[-1][0]:
            print()

    # --- Export to CSV ---
    export_csv(all_tvl_results, 'results.csv')
    export_comparison_csv(all_tvl_results, precisions, 'comparison.csv')


def export_csv(all_tvl_results, filename='results.csv'):
    """Export all results to a CSV file."""
    import csv
    import os

    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'TVL', 'Range', 'Precision', 'Mode',
            'Agg_Fees_A', 'Arb_Fees_A', 'Total_LP_Fees_A',
            'Priv_Profit', 'Leak_Over_Fees_Pct',
            'Volume_A_Pct', 'Volume_A', 'Volume_B',
            'Final_Price_A', 'Final_Price_B',
        ])

        for tvl_label, res_list in all_tvl_results:
            for r in res_list:
                total_fees = r['agg_fees_a'] + r['arb_fees_a']
                leak_pct = (r['priv_profit'] / total_fees * 100) if total_fees > 0 else 0
                writer.writerow([
                    tvl_label,
                    r['range'],
                    r['precision'],
                    r['mode'],
                    round(r['agg_fees_a'], 4),
                    round(r['arb_fees_a'], 4),
                    round(total_fees, 4),
                    round(r['priv_profit'], 4),
                    round(leak_pct, 4),
                    round(r['volume_a_pct'], 2),
                    round(r['volume_a'], 2),
                    round(r['volume_b'], 2),
                    round(r['final_price_a'], 4),
                    round(r['final_price_b'], 4),
                ])

    print(f"\nCSV exported to: {filepath}")
    return filepath


def export_comparison_csv(all_tvl_results, precisions, filename='comparison.csv'):
    """Export Standard vs Privileged comparison to CSV."""
    import csv
    import os

    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'TVL', 'Range', 'Precision',
            'Std_Agg_Fees_A', 'Std_Arb_Fees_A', 'Std_Total_Fees_A', 'Std_Vol_A_Pct', 'Std_Final_Price_A',
            'Priv_Agg_Fees_A', 'Priv_Arb_Fees_A', 'Priv_Total_Fees_A', 'Priv_Vol_A_Pct', 'Priv_Final_Price_A',
            'Priv_Profit', 'Leak_Over_Fees_Pct', 'Fee_Delta',
        ])

        for tvl_label, res_list in all_tvl_results:
            for range_name in ['±5%', '±10%', '±15%']:
                for prec in precisions:
                    std = next((r for r in res_list
                                if r['range'] == range_name and r['precision'] == prec
                                and r['mode'] == 'standard'), None)
                    priv = next((r for r in res_list
                                 if r['range'] == range_name and r['precision'] == prec
                                 and r['mode'] == 'privileged'), None)
                    if std is None or priv is None:
                        continue

                    fees_std = std['agg_fees_a'] + std['arb_fees_a']
                    fees_priv = priv['agg_fees_a'] + priv['arb_fees_a']
                    leak = priv['priv_profit']
                    leak_pct = (leak / fees_priv * 100) if fees_priv > 0 else 0
                    fee_delta = fees_priv - fees_std

                    writer.writerow([
                        tvl_label, range_name, prec,
                        round(std['agg_fees_a'], 4),
                        round(std['arb_fees_a'], 4),
                        round(fees_std, 4),
                        round(std['volume_a_pct'], 2),
                        round(std['final_price_a'], 4),
                        round(priv['agg_fees_a'], 4),
                        round(priv['arb_fees_a'], 4),
                        round(fees_priv, 4),
                        round(priv['volume_a_pct'], 2),
                        round(priv['final_price_a'], 4),
                        round(leak, 4),
                        round(leak_pct, 4),
                        round(fee_delta, 4),
                    ])

    print(f"Comparison CSV exported to: {filepath}")
    return filepath


if __name__ == '__main__':
    main()
