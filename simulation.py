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



if __name__ == '__main__':
    pass
