import numpy as np
import pandas as pd
from typing import Dict, Optional, Iterable, Tuple

# ---- Helpers ----

def _reset_simfin(df: pd.DataFrame, index_names: Tuple[str, str] = ("Ticker","Report Date")) -> pd.DataFrame:

    if isinstance(df.index, pd.MultiIndex) and list(df.index.names)[:2] == list(index_names):
        out = df.reset_index()
        out.rename(columns={index_names[0]: "ticker", index_names[1]: "report_date"}, inplace=True)
    elif "Ticker" in df.columns and "Report Date" in df.columns:
        out = df.rename(columns={"Ticker":"ticker","Report Date":"report_date"}).copy()
    else:
        # Fallback: try existing index to columns
        out = df.reset_index().copy()
        if "Report Date" in out.columns and "Ticker" in out.columns:
            out.rename(columns={"Ticker":"ticker","Report Date":"report_date"}, inplace=True)
        elif "SimFinId" in out.columns and "report_date" not in out.columns:
            # Some variants return SimFinId without explicit Report Date column
            # In that case we can't recover 'report_date' reliably; keep as-is.
            pass
    # Standardize case
    out.columns = [str(c) for c in out.columns]
    if "report_date" in out.columns:
        out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce")
    if "Publish Date" in out.columns:
        out["Publish Date"] = pd.to_datetime(out["Publish Date"], errors="coerce")
    return out

def _trim_cols(df: pd.DataFrame, keep: Iterable[str]) -> pd.DataFrame:
    present = [c for c in keep if c in df.columns]
    return df[present].copy()

def _safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: Iterable[str]) -> pd.DataFrame:
    return left.merge(right, on=list(on), how="outer")

def _restrict_timeline(df: pd.DataFrame, start: str = "2019-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    # Use Publish Date when available, else Report Date
    if "Publish Date" in df.columns:
        key = "Publish Date"
    else:
        key = "report_date" if "report_date" in df.columns else None
    if key:
        m = (df[key] >= pd.Timestamp(start)) & (df[key] <= pd.Timestamp(end))
        df = df.loc[m].copy()
    return df

# ---- Core loader (expects caller to invoke SimFin) ----

def build_quarterly_panel(
    income_q: pd.DataFrame,
    balance_q: pd.DataFrame,
    cashflow_q: pd.DataFrame,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:

    inc = _reset_simfin(income_q)
    bal = _reset_simfin(balance_q)
    cfs = _reset_simfin(cashflow_q)

    # Select allowed columns per user specification
    inc_keep = [
        "ticker","report_date","Publish Date",
        "Revenue","Gross Profit","Operating Expenses","Operating Income (Loss)",
        "Net Income","Shares (Basic)","Shares (Diluted)"
    ]
    bal_keep = [
        "ticker","report_date","Publish Date",
        "Total Assets","Total Liabilities","Total Equity",
        "Short Term Debt","Long Term Debt","Accounts & Notes Receivable",
        "Inventories","Payables & Accruals","Total Current Assets","Total Current Liabilities"
    ]
    cfs_keep = [
        "ticker","report_date","Publish Date",
        "Net Cash from Operating Activities","Dividends Paid","Net Change in Cash",
        "Change in Working Capital","Change in Accounts Receivable","Change in Inventories","Change in Accounts Payable"
    ]

    inc = _trim_cols(inc, inc_keep)
    bal = _trim_cols(bal, bal_keep)
    cfs = _trim_cols(cfs, cfs_keep)

    # Merge on (ticker, report_date) with Publish Date preference
    panel = _safe_merge(inc, bal, on=("ticker","report_date","Publish Date"))
    panel = _safe_merge(panel, cfs, on=("ticker","report_date","Publish Date"))

    # Restrict to timeline
    panel = _restrict_timeline(panel, start, end)
    panel = panel.sort_values(["ticker","report_date","Publish Date"]).reset_index(drop=True)
    return panel

# ---- Feature engineering ----

def quarterly_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    # Ratios
    df["gross_margin"] = df["Gross Profit"] / df["Revenue"]
    df["oper_margin"]  = df["Operating Income (Loss)"] / df["Revenue"]
    # Leverage
    debt = df["Short Term Debt"].fillna(0) + df["Long Term Debt"].fillna(0)
    df["leverage"] = debt / df["Total Assets"]
    # Working-capital pressure (receivables + inventories - payables) scaled by revenue
    wc = df["Accounts & Notes Receivable"].fillna(0) + df["Inventories"].fillna(0) - df["Payables & Accruals"].fillna(0)
    df["wc_over_rev"] = wc / df["Revenue"]
    # Cash-flow quality
    df["accruals"] = (df["Net Income"] - df["Net Cash from Operating Activities"]) / df["Total Assets"]

    # Group by ticker for YoY and accelerations (4-quarter diff/perc)
    def _by_tkr(g):
        g = g.sort_values("report_date")
        # YoY growth
        g["rev_yoy"]  = g["Revenue"].pct_change(4)
        g["gmom_yoy"] = g["gross_margin"] - g["gross_margin"].shift(4)    # margin change (pp)
        g["opmom_yoy"]= g["oper_margin"]  - g["oper_margin"].shift(4)
        g["cfo_yoy"]  = g["Net Cash from Operating Activities"].pct_change(4)
        # Deteriorations (bad if rising)
        g["lev_delta"] = g["leverage"] - g["leverage"].shift(4)
        g["wc_delta"]  = g["wc_over_rev"] - g["wc_over_rev"].shift(4)
        g["accr_delta"]= g["accruals"] - g["accruals"].shift(4)
        # Publish Date carried through
        return g

    out = df.groupby("ticker", group_keys=False).apply(_by_tkr).reset_index(drop=True)
    return out

def expand_to_daily(features_q: pd.DataFrame, close: pd.DataFrame, lag_bdays: int = 2) -> Dict[str, pd.DataFrame]:

    f = features_q.dropna(subset=["ticker"]).copy()
    # Choose go-live: Publish Date if available; fallback to report_date
    use_date = "Publish Date" if "Publish Date" in f.columns else "report_date"
    f[use_date] = pd.to_datetime(f[use_date], errors="coerce")
    f["go_live"] = (f[use_date] + pd.tseries.offsets.BDay(lag_bdays)).dt.tz_localize(None)

    # Select feature columns to expand
    ignore = {"ticker","report_date","Publish Date","go_live"}
    feats = [c for c in f.columns if c not in ignore]

    calendar = pd.DataFrame(index=close.index)
    out = {}
    for feat in feats:
        tmp = f[["ticker","go_live",feat]].dropna(subset=["go_live"])
        wide = tmp.pivot_table(index="go_live", columns="ticker", values=feat, aggfunc="last").sort_index()
        wide = wide.reindex(calendar.index).ffill()
        out[feat] = wide
    return out