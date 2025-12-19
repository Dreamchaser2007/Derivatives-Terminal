# This file explains core_fx.py line-by-line with WHY / WHAT / WHY NOT notes.
# Each code line from core_fx.py is followed by three short comments:
#   WHY: why this line exists (motivation)
#   WHAT: what it does (effect)
#   WHY NOT: brief rationale for the chosen approach vs obvious alternatives

import math
# WHY: Need basic math functions used throughout the module.
# WHAT: Exposes log, exp, sqrt, erf, pi, etc.
# WHY NOT: numpy/scipy could be used, but math is lighter-weight and has required functions.

from typing import List, Tuple
# WHY: Help readers and tools understand function interfaces.
# WHAT: Brings in type names for annotations used below.
# WHY NOT: Type hints are optional at runtime, but improve clarity and linting.

# ---------- Normal distribution helpers ----------
def norm_cdf(x: float) -> float:
# WHY: Central building block for Black–Scholes style formulas.
# WHAT: Declares a function that returns the standard normal CDF at x.
# WHY NOT: Could inline the expression where used, but a function improves reuse and readability.
    # WHY: Use math.erf to compute the normal CDF reliably.
    # WHAT: Computes 0.5*(1+erf(x/sqrt(2))).
    # WHY NOT: Adding scipy.stats.norm would be heavier dependency for a simple formula.
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
# WHY: Probability density is needed for Greeks and intermediate terms.
# WHAT: Declares a function returning the standard normal PDF at x.
# WHY NOT: Same reasoning as CDF—keeps implementation self-contained and dependency-light.
    # WHY: Use direct formula for standard normal pdf.
    # WHAT: Returns (1/sqrt(2π)) * exp(-0.5*x^2).
    # WHY NOT: Using numpy.exp would be fine but math.exp suffices and avoids numpy overhead.
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


# =================================================
# FX OPTIONS (Garman–Kohlhagen + Binomial)
# =================================================

def garman_kohlhagen(option_type: str, S: float, K: float,
                     rd: float, rf: float, sigma: float, T: float) -> float:
# WHY: Provide a closed-form FX European option pricer.
# WHAT: Function signature for Garman–Kohlhagen (BS adapted for FX).
# WHY NOT: Keeping signature explicit makes unit testing and reuse straightforward.
    """
    Garman–Kohlhagen formula for FX European options.
    rd = domestic rate, rf = foreign rate
    """
    # WHY: Normalize option type to avoid case-sensitivity bugs.
    # WHAT: Convert the option_type string to lowercase.
    # WHY NOT: Could accept enums, but string is simple and sufficient here.
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        # WHY: Defensive programming to catch invalid inputs early.
        # WHAT: Raises if an unsupported option type is supplied.
        # WHY NOT: Returning None hides errors; raising makes issues obvious.
        raise ValueError("option_type must be 'call' or 'put'.")

    # WHY: Handle instant-expiry as a simple special case.
    # WHAT: If time to expiry is zero or negative, return intrinsic value.
    # WHY NOT: Avoids numerical issues dividing by sqrt(T) when T==0.
    if T <= 0.0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    # WHY: Treat zero volatility analytically to avoid division by zero.
    # WHAT: Compute forward price and discounted intrinsic for sigma == 0.
    # WHY NOT: A numerical limit or tiny sigma could be used, but exact branch is clearer.
    if sigma <= 0.0:
        forward = S * math.exp((rd - rf) * T)
        disc = math.exp(-rd * T)
        if option_type == "call":
            return max(forward - K, 0.0) * disc
        else:
            return max(K - forward, 0.0) * disc

    # WHY: Compute standard d1 and d2 intermediates used in closed-form formulas.
    # WHAT: d1 and d2 required by the GK/BS formula.
    # WHY NOT: These are standard; re-deriving elsewhere would be redundant.
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (rd - rf + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    # WHY: Precompute discount factors to keep formulas compact.
    # WHAT: disc_d discounts domestic cashflows, disc_f discounts foreign/underlying.
    # WHY NOT: Inline exp calls could be used, but naming clarifies purpose.
    disc_d = math.exp(-rd * T)
    disc_f = math.exp(-rf * T)

    # WHAT: Use GK closed-form; foreign rate acts like continuous dividend yield.
    # WHY: This is the accepted model for spot FX European options.
    # WHY NOT: Local volatility or stochastic rates would be more complex and unnecessary here.
    if option_type == "call":
        price = S * disc_f * norm_cdf(d1) - K * disc_d * norm_cdf(d2)
    else:
        price = K * disc_d * norm_cdf(-d2) - S * disc_f * norm_cdf(-d1)

    # WHY: Return a plain float to ease JSON serialization and display.
    # WHAT: Ensure result is a built-in float type.
    # WHY NOT: Returning numpy types can cause surprises in templates/JSON.
    return float(price)


def gk_greeks(option_type: str, S: float, K: float,
              rd: float, rf: float, sigma: float, T: float):
# WHY: Provide Greeks for GK so UI and callers can display sensitivities.
# WHAT: Function signature that returns a dict of Greek values.
# WHY NOT: Separating Greeks from price keeps single-responsibility and easier testing.
    """
    Greeks for Garman–Kohlhagen FX options.
    Returns dict: delta, gamma, vega, theta, rho_domestic, rho_foreign
    """
    # WHY: Validate input early like in the pricer.
    # WHAT: Normalize and validate option_type value.
    # WHY NOT: Reuse of validation logic avoids subtle downstream errors.
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    # WHY: For expired or zero-vol options the sensitivities are effectively zero.
    # WHAT: Return zeros to avoid divide-by-zero and meaningless values.
    # WHY NOT: Returning None would complicate callers expecting numeric Greeks.
    if T <= 0.0 or sigma <= 0.0:
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho_domestic": 0.0,
            "rho_foreign": 0.0,
        }

    # WHY: Compute d1 and d2 once for reuse in formulas below.
    # WHAT: Standard BS intermediates with domestic/foreign adjustments.
    # WHY NOT: Recomputing inline per Greek would duplicate code.
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (rd - rf + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    # WHY: Prepare discount factors and pdf used by multiple Greeks.
    # WHAT: disc_d and disc_f used for domestic/foreign discounting; pdf_d1 used for density.
    # WHY NOT: Inline recomputation would be less readable.
    disc_d = math.exp(-rd * T)
    disc_f = math.exp(-rf * T)
    pdf_d1 = norm_pdf(d1)

    # WHY: For FX, delta discounts by foreign rate (continuous dividend analog).
    # WHAT: Compute call and put deltas using the GK convention.
    # WHY NOT: Using undiscounted delta would be incorrect for FX instruments.
    if option_type == "call":
        delta = disc_f * norm_cdf(d1)
    else:
        delta = -disc_f * norm_cdf(-d1)

    # WHY: Gamma and vega formulas follow directly from differentiation of BS/GK.
    # WHAT: Compute gamma and vega (not scaled).
    # WHY NOT: Some code scales vega to per-1% units; caller can rescale if desired.
    gamma = disc_f * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_f * pdf_d1 * sqrtT

    # WHY: Theta and rho have different signs and terms for call vs put.
    # WHAT: Compute theta and rhos for both option types.
    # WHY NOT: Alternative conventions exist (per-day vs per-year theta); this choice is explicit.
    if option_type == "call":
        theta = (
            -S * disc_f * pdf_d1 * sigma / (2.0 * sqrtT)
            - rd * K * disc_d * norm_cdf(d2)
            + rf * S * disc_f * norm_cdf(d1)
        )
        rho_dom = T * K * disc_d * norm_cdf(d2)
        rho_for = -T * S * disc_f * norm_cdf(d1)
    else:
        theta = (
            -S * disc_f * pdf_d1 * sigma / (2.0 * sqrtT)
            + rd * K * disc_d * norm_cdf(-d2)
            - rf * S * disc_f * norm_cdf(-d1)
        )
        rho_dom = -T * K * disc_d * norm_cdf(-d2)
        rho_for = T * S * disc_f * norm_cdf(-d1)

    # WHY: Return plain floats for predictable serialization and display.
    # WHAT: Pack Greeks into a dict with clear keys.
    # WHY NOT: Returning custom objects would complicate template rendering and JSON export.
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho_domestic": float(rho_dom),
        "rho_foreign": float(rho_for),
    }


def fx_binomial(option_type: str, exercise_type: str,
                S: float, K: float, rd: float, rf: float,
                sigma: float, T: float, N: int) -> float:
# WHY: Provide a discrete CRR tree alternative to closed-form GK for FX.
# WHAT: Function signature for binomial pricing that supports European/American.
# WHY NOT: Binomial adds flexibility (early exercise) and is simple to implement.
    """
    CRR binomial tree for FX options (European or American).
    """
    # WHY: Normalize inputs to canonical form.
    # WHAT: Lowercase strings to avoid case-sensitivity bugs.
    # WHY NOT: Could accept enums, but strings are simpler in web forms.
    option_type = option_type.lower()
    exercise_type = exercise_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    if exercise_type not in ("european", "american"):
        raise ValueError("exercise_type must be 'european' or 'american'.")
    if N < 1:
        raise ValueError("N must be >= 1")

    # WHY: Compute CRR tree parameters from inputs.
    # WHAT: u and d are up/down multipliers; dt is step length; disc for discounting.
    # WHY NOT: Other lattice schemes exist, but CRR is standard and stable.
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-rd * dt)
    # WHY: Risk-neutral probability must account for domestic/foreign differential.
    # WHAT: p computed to make expected growth consistent with rd-rf.
    # WHY NOT: Using (r - q) analog would be fine; here rf plays q-like role.
    p = (math.exp((rd - rf) * dt) - d) / (u - d)
    # WHY NOT: If p falls outside (0,1) inputs are inconsistent — raise to surface problem.
    if not (0.0 < p < 1.0):
        raise ValueError("Risk-neutral probability out of bounds; check inputs (rd, rf, sigma, N)")

    # WHY: Build terminal node spot prices for all up/down combinations.
    # WHAT: List comprehension building S * u^j * d^(N-j) for j=0..N.
    # WHY NOT: Storing full tree nodes earlier wastes memory; terminal approach is compact.
    prices = [S * (u ** j) * (d ** (N - j)) for j in range(N + 1)]
    # WHY: Compute terminal payoffs depending on call/put.
    # WHAT: Use max(0, S-K) or max(0, K-S).
    # WHY NOT: Alternative payoff forms aren't needed here.
    if option_type == "call":
        vals = [max(0.0, p0 - K) for p0 in prices]
    else:
        vals = [max(0.0, K - p0) for p0 in prices]

    # WHY: Backward induction to collapse tree to current price.
    # WHAT: Iterate from time N-1 down to 0, recomputing node values.
    # WHY NOT: Using full two-dimensional arrays is possible but this rolling approach saves memory.
    for i in range(N - 1, -1, -1):
        new_vals = []
        for j in range(i + 1):
            # WHY: Expected discounted continuation value at this node.
            # WHAT: cont = discount * (p*up + (1-p)*down).
            # WHY NOT: This is the standard CRR recursion; no simpler correct alternative.
            cont = disc * (p * vals[j + 1] + (1 - p) * vals[j])
            if exercise_type == "american":
                # WHY: For American options, compare continuation with immediate exercise.
                # WHAT: Compute spot at this node and the exercise payoff.
                # WHY NOT: European style skips early exercise check.
                spot = S * (u ** (j + 1)) * (d ** (i - j))
                exercise = max(0.0, spot - K) if option_type == "call" else max(0.0, K - spot)
                newv = max(cont, exercise)
            else:
                newv = cont
            new_vals.append(newv)
        vals = new_vals
    # WHY: Return a float for easy consumption by callers.
    # WHAT: Today's option value equals the lone remaining element.
    # WHY NOT: Keeping as float avoids downstream type complications.
    return float(vals[0])


# =================================================
# STOCK OPTIONS (Black–Scholes + Binomial)
# =================================================

def black_scholes(option_type: str, S: float, K: float,
                  r: float, sigma: float, T: float, q: float = 0.0) -> float:
# WHY: Standard closed-form equity option pricer with continuous dividend yield q.
# WHAT: Function signature mirrors typical BS implementations.
# WHY NOT: Keeps parity with GK function for FX so callers can swap models easily.
    """
    Black–Scholes for equity options with continuous dividend yield q.
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    # WHY: Immediate expiry returns intrinsic value to avoid numeric issues.
    # WHAT: Intrinsic if T<=0.
    # WHY NOT: Alternative would be a tiny epsilon, but this explicit branch is clearer.
    if T <= 0.0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)

    # WHY: Compute d1/d2 needed by BS formula.
    # WHAT: d1 uses (r-q) adjustment for continuous dividend.
    # WHY NOT: No need to re-explain standard BS algebra here.
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    # WHY: Discount factors for interest and dividend.
    # WHAT: disc_r discounts strike cashflow; disc_q discounts underlying dividend effect.
    # WHY NOT: Explicit names make formulas more readable.
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    # WHAT: Closed-form stock option price with continuous dividend yield.
    # WHY: Mirrors textbook Black–Scholes.
    # WHY NOT: For discrete dividends a different model would be needed.
    if option_type == "call":
        price = S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2)
    else:
        price = K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1)

    return float(price)


def bs_greeks(option_type: str, S: float, K: float,
              r: float, sigma: float, T: float, q: float = 0.0):
# WHY: Provide Greeks for equity options consistent with BS formula.
# WHAT: Function returns common sensitivities.
# WHY NOT: Keeps analytics centralized and consistent with pricing function.
    """
    Greeks for Black–Scholes equity option with dividend yield q.
    Returns dict: delta, gamma, vega, theta, rho
    """
    option_type = option_type.lower()
    # WHY: Expired or zero-vol cases return zeros to avoid invalid math.
    # WHAT: Defensive check for degenerate cases.
    # WHY NOT: Returning None would force callers to check for None each time.
    if T <= 0.0 or sigma <= 0.0:
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
        }

    # WHY: Compute d1/d2 and pdf only once.
    # WHAT: Reuse in multiple Greek calculations.
    # WHY NOT: Recomputing per Greek would be wasteful and error-prone.
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    pdf_d1 = norm_pdf(d1)

    # WHY: Delta depends on option type and continuous dividend.
    # WHAT: Compute call/put delta with discount by q.
    # WHY NOT: Using undiscounted delta would misrepresent sensitivity for dividend-paying stocks.
    if option_type == "call":
        delta = disc_q * norm_cdf(d1)
    else:
        delta = -disc_q * norm_cdf(-d1)

    # WHY: Gamma and vega derived from derivatives of d1/d2.
    # WHAT: Compute gamma and vega in natural units.
    # WHY NOT: Many callers scale vega (/100) when presenting; keep raw value here.
    gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_q * pdf_d1 * sqrtT

    # WHY: Theta and rho have distinct formulas for calls and puts.
    # WHAT: Compute them following standard BS derivatives (theta here annualized).
    # WHY NOT: Could present theta per-day; choose per-year for consistency with formula.
    if option_type == "call":
        theta = (-S * sigma * disc_q * pdf_d1 / (2.0 * sqrtT)
                 - r * K * disc_r * norm_cdf(d2)
                 + q * S * disc_q * norm_cdf(d1))
        rho = K * T * disc_r * norm_cdf(d2)
    else:
        theta = (-S * sigma * disc_q * pdf_d1 / (2.0 * sqrtT)
                 + r * K * disc_r * norm_cdf(-d2)
                 - q * S * disc_q * norm_cdf(-d1))
        rho = -K * T * disc_r * norm_cdf(-d2)

    # WHY: Return floats in a dict to make output easy to consume by UI/tests.
    # WHAT: Pack and return computed Greeks.
    # WHY NOT: Avoid returning numpy scalars to keep results JSON-friendly.
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def equity_binomial(option_type: str, exercise_type: str,
                    S: float, K: float, r: float,
                    sigma: float, T: float, N: int, q: float = 0.0) -> float:
# WHY: Mirror fx_binomial but for equity (dividend yield q used).
# WHAT: Function signature for CRR tree with continuous dividend q.
# WHY NOT: Keeps symmetry with GK/BS implementations, easing maintenance.
    """
    CRR binomial tree for equity options with dividend yield q.
    """
    option_type = option_type.lower()
    exercise_type = exercise_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    if exercise_type not in ("european", "american"):
        raise ValueError("exercise_type must be 'european' or 'american'.")
    if N < 1:
        raise ValueError("N must be >= 1")

    # WHY: CRR parameters take q into account via (r - q) in risk-neutral drift.
    # WHAT: Compute step size dt, up/down multipliers, discount, and p.
    # WHY NOT: This keeps behavior consistent with continuous-dividend BS.
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError("Risk-neutral probability out of bounds; check inputs (r, q, sigma, N)")

    # WHY: Terminal node spot prices then terminal payoffs as in FX tree.
    # WHAT: Build prices and payoffs lists, then backward-induct.
    # WHY NOT: More advanced lattices exist but CRR is standard and reliable.
    prices = [S * (u ** j) * (d ** (N - j)) for j in range(N + 1)]
    if option_type == "call":
        vals = [max(0.0, p0 - K) for p0 in prices]
    else:
        vals = [max(0.0, K - p0) for p0 in prices]

    for i in range(N - 1, -1, -1):
        new_vals = []
        for j in range(i + 1):
            cont = disc * (p * vals[j + 1] + (1 - p) * vals[j])
            if exercise_type == "american":
                spot = S * (u ** (j + 1)) * (d ** (i - j))
                exercise = max(0.0, spot - K) if option_type == "call" else max(0.0, K - spot)
                newv = max(cont, exercise)
            else:
                newv = cont
            new_vals.append(newv)
        vals = new_vals
    # WHY: Return current time option value as float.
    # WHAT: Caller receives a simple numeric price.
    # WHY NOT: Keeping it minimal avoids caller-side conversions.
    return float(vals[0])
