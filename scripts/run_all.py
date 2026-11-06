"""Main pipeline entry point."""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loaders import load_yields, build_forward_rates, build_excess_returns
from src.cp_factor import construct_cp_factor, replication_checks
from src.oos_evaluation import expanding_window_oos, summarize_oos
from src.strategies import strategy_performance
from src.stress import run_stress_comparison, named_shocks
import pandas as pd


def main():
    print("Loading Fama-Bliss yields...")
    yields = load_yields(source="fama_bliss")
    print(f"Yields shape: {yields.shape}")

    print("Building forwards and excess returns...")
    forwards = build_forward_rates(yields)
    rx = build_excess_returns(yields)
    print(f"Excess returns shape: {rx.shape}")

    print("Constructing CP factor (in-sample 1964-2003)...")
    factor, model = construct_cp_factor(forwards, rx)
    replication_checks(model)

    print("\nRunning expanding-window OOS (including Timestral)...")
    oos = expanding_window_oos(forwards, rx, include_timestral=True)
    print(f"OOS periods: {len(oos)}")
    summary = summarize_oos(oos)
    print("OOS summary:", summary)
    oos.to_csv(Path(__file__).parent.parent / "results" / "oos_forecasts_fb.csv")
    print("Saved OOS forecasts to results/oos_forecasts_fb.csv")

    print("\nDuration-timing strategy evaluation...")
    strat_results = []
    for col in ["cp", "timestral", "hybrid"]:
        if col in oos.columns and oos[col].notna().sum() > 20:
            res = strategy_performance(oos, col)
            strat_results.append(res)
            print(f"  {col}: gross Sharpe {res['gross_sharpe']}, 2bp Sharpe {res.get('sharpe_2bp', 'n/a')}")
    if strat_results:
        strat_df = pd.DataFrame(strat_results)
        print(strat_df.to_string(index=False))
        strat_df.to_csv(Path(__file__).parent.parent / "results" / "strategy_performance_fb.csv", index=False)
        print("Saved strategy performance to results/strategy_performance_fb.csv")

    print("\nStress testing...")
    stress = run_stress_comparison(oos, alpha=0.99)
    print("VaR/ES comparison:", stress)
    shocks = named_shocks(None, {"+200bp parallel": 200, "bear steepener": 150})
    print("Named shocks (60/40 2y/10y approx):", shocks)
    pd.DataFrame([stress]).to_csv(Path(__file__).parent.parent / "results" / "stress_results.csv", index=False)
    print("Saved stress results to results/stress_results.csv")


if __name__ == "__main__":
    main()
