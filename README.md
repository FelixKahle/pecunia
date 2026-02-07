# Pecunia

This repository contains a Julia-based financial modeling engine that was designed to solve the **Liability-Driven Investing (LDI)** problem for personal portfolios.

It was built as an experiment to move beyond simple "Mean-Variance" optimization by introducing real-world constraints—such as minimum trade sizes, transaction costs, and hard safety floors—using Mixed-Integer Quadratic Programming (MIQP).

It **might** be useful to you if you are looking for a rigorous mathematical framework to balance absolute profit targets against minimum risk, rather than just blindly maximizing returns.

---

## 1. The Core Philosophy

The tool was built on the assumption that **volatility minimization** is more important than return maximization for a liability-driven investor.

Instead of asking *"What is the highest return I can get?"*, this model asked:
> *"What is the **safest possible portfolio** that still has a mathematical probability of hitting my fixed profit target (e.g., €15,000 real profit in 2 years)?"*

### Key Assumptions Made
* **Market Efficiency:** The model assumed that past returns were noisy. Therefore, it used **Ledoit-Wolf Shrinkage** on the covariance matrix rather than raw sample covariance to prevent the optimizer from "chasing noise."
* **Rational Constraints:** It assumed that holding 50 stocks with €5 allocations was impractical. A **Cardinality Constraint** (via binary integer variables) was enforced to ensure every position was at least `min_trade_amount` (e.g., €1,000).
* **Inflation Adjustment:** All optimization was performed in **Real Terms** (Nominal Return - Expected Inflation).

---

## 2. Technical Features

* **Solver Agnostic (mostly):** While originally configured for **Gurobi** (due to its speed with MIQP problems), the code was written in `JuMP`, meaning it could be adapted to open-source solvers like **HiGHS** with minimal changes.
* **Robust Data Ingestion:**
    * Automatically fetched historical data via `YFinance`.
    * Filtered out assets with insufficient history to prevent "Survivorship Bias" in the covariance matrix.
    * Applied a hard cap (`maximum_stock_return_cap`) to expected returns to prevent the model from overfitting to recent bubbles.
* **Visualization:** Generated a donut chart of the optimal allocation using `CairoMakie`.

---

## 3. Prerequisites

To run this tool as it was originally designed, the following environment was required:

1.  **Julia 1.10+**
2.  **Gurobi Optimizer** (and a valid license).
    * *Note: If you do not have a license, you must modify the `Model(Gurobi.Optimizer)` line to use `HiGHS` or `Ipopt`.*
3.  **Packages:**
    ```julia
    import Pkg
    Pkg.add(["JuMP", "Gurobi", "YFinance", "DataFrames", "Statistics", "LinearAlgebra", "Dates", "ProgressMeter", "Printf", "CSV", "CairoMakie", "ColorSchemes"])
    ```

---

## 4. Input Data Format

The system expected two CSV files in the `data/` directory:

### `shares.csv`
Used for assets where historical data needed to be fetched via API.

| Ticker | Name | Min | Max |
| :--- | :--- | :--- | :--- |
| MSFT | Microsoft | 0.0 | 0.15 |
| LIN | Linde PLC | 0.05 | |
| AAPL | Apple Inc | | |

* **Min/Max:** Optional hard constraints (0.0 to 1.0). Leave empty if no constraint.

### `bonds.csv`
Used for "Manual" assets where yield and risk were known/fixed (e.g., bonds, private equity).

| Name | Yield | Risk |
| :--- | :--- | :--- |
| Apple Corp Bond | 0.045 | 0.08 |
| HY ETF Manual | 0.065 | 0.12 |

---

## 5. Configuration

The `PortfolioConfiguration` struct was the control center. Important parameters included:

```julia
PortfolioConfiguration(
    "EUR", "€",          # Currency
    80_000.0, 100_000.0, # Min/Max Investable Capital
    1000.0,              # Minimum Trade Size (MIQP Constraint)
    15_000.0,            # Target REAL Profit
    2.0,                 # Time Horizon (Years)
    0.025,               # Inflation Assumption
    0.035,               # Risk Free Rate
    0.25,                # Global Max Weight per Asset
    0.10,                # Minimum Safety Bucket (Cash/Bonds)
    ...
)
```

---

## 6. Usage

1. Place your CSVs in `src/data/`.
2. Adjust the `PortfolioConfiguration` in the `run_pipeline()` function.
3. Execute the script:

```bash
julia src/WealthOptimizer.jl
```

If the optimization was feasible, it produced a CLI report and a `portfolio_strategy.png` chart.

---

## 7. Disclaimer

**This software was created for educational and research purposes.**

* It is **not** financial advice.
* It relied on historical data, which is not a predictor of future results.
* The `shrinkage_intensity` and `max_stock_return_cap` were heuristic parameters that required tuning based on the economic regime of the time (2026).

Use at your own risk.