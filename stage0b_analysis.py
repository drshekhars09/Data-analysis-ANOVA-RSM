"""
Stage 0b (slide 9): stage-2 milling DoE (Taguchi L9) run on two stage-1 feeds, A and B.

From the completed run table this script produces, for each response
(O pick-up, yield in 20-106 um, flake thickness, micro-hardness):
  - split-plot ANOVA     feed (A, B) = whole-plot factor, tested against batch(feed);
                         speed, time and BPR = sub-plot factors
  - regression model     reduced quadratic model in coded factors (backward elimination,
                         p > 0.10, hierarchy kept), ANOVA with lack of fit and pure error,
                         model summary (Std. Dev., Mean, C.V. %, PRESS, R2, Adj R2, Pred R2,
                         Adeq Precision), equations in coded and actual factors,
                         predicted vs actual plots and response-surface plots
  - optimisation         Derringer-Suich desirability: constraints table, ranked solutions,
                         predicted values with 95 % prediction intervals for the confirmation run

Usage:   python stage0b_analysis.py stage0b_runs.csv
Outputs: stage0b_results.xlsx and PNG figures in 'stage0b_out/'

Run table (CSV), one row per milling run:
  run, feed, batch, speed, time, bpr, o_pickup, yield, flake_thickness, microhardness
  feed  : A or B (stage-1 setting)            batch : stage-1 batch, e.g. A1, A2, B1, B2
  speed : rpm   time : h   bpr : number before ':1'

Note: a Taguchi L9 cannot estimate speed x time x BPR two-factor interactions, so the model
holds main effects, curvature (squared terms) and the feed x factor interactions.
"""
import sys, os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f as fdist, t as tdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3-D projection)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------ settings
FACTORS = {  # name: (low, centre, high, unit, letter)
    "speed": (250, 300, 350, "rpm", "A"),
    "time": (12, 18, 24, "h", "B"),
    "bpr": (5, 10, 15, ":1", "C"),
}
RESPONSES = {  # column: (label, unit, goal, importance)
    "o_pickup":        ("O pick-up", "wt%", "minimize", 5),
    "yield":           ("Yield 20-106 µm", "%", "maximize", 5),
    "flake_thickness": ("Flake thickness", "µm", "maximize", 3),
    "microhardness":   ("Micro-hardness", "HV0.05", "minimize", 2),
}
FACTOR_GOALS = {"time": ("minimize", 2)}   # fewer mill-hours per kg; others 'is in range'
P_REMOVE = 0.10
GRID_STEP = 0.05
OUT = "stage0b_out"
LET = {f: v[4] for f, v in FACTORS.items()}


# ------------------------------------------------------------------------------ design
def prepare(df):
    df = df.copy()
    df["feed"] = df["feed"].astype(str).str.strip()
    df["batch"] = df["batch"].astype(str).str.strip()
    for f, (lo, c, hi, _, L) in FACTORS.items():
        df[L] = (df[f] - c) / ((hi - lo) / 2)          # coded -1 / 0 / +1
    feeds = sorted(df["feed"].unique())
    df["D"] = np.where(df["feed"] == feeds[0], -1.0, 1.0)  # feed coded: A = -1, B = +1
    return df, feeds


def term_columns(df, terms, actual=False):
    X = pd.DataFrame(index=df.index)
    for t in terms:
        parts = t.split("*")
        col = np.ones(len(df))
        for p in parts:
            if actual and p in LET.values():
                p = [f for f, L in LET.items() if L == p][0]
            col = col * df[p].values
        X[t] = col
    return X


FULL = ["D", "A", "B", "C", "A*A", "B*B", "C*C", "D*A", "D*B", "D*C"]


def hierarchy_ok(term, terms):
    others = [t for t in terms if t != term]
    if term in ("A", "B", "C"):
        return not any(term in t.split("*") for t in others)
    if term == "D":
        return not any("D" in t.split("*") for t in others)
    return True


def fit(df, y, terms, actual=False):
    X = sm.add_constant(term_columns(df, terms, actual), has_constant="add")
    return sm.OLS(df[y].values, X).fit()


def reduce_model(df, y):
    terms = FULL.copy()
    while True:
        m = fit(df, y, terms)
        cand = [(m.pvalues[t], t) for t in terms if hierarchy_ok(t, terms) and m.pvalues[t] > P_REMOVE]
        if not cand:
            return terms, m
        terms.remove(max(cand)[1])


# ------------------------------------------------------------------------------ statistics
def pure_error(df, y):
    g = df.groupby(["feed", "speed", "time", "bpr"])[y]
    ss = g.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum()
    dfe = int((g.size() - 1).clip(lower=0).sum())
    return ss, dfe


def model_anova(df, y, terms, m):
    sst = ((df[y] - df[y].mean()) ** 2).sum()
    rss, dfr = m.ssr, int(m.df_resid)
    msr = rss / dfr
    rows = [("Model", sst - rss, len(terms))]
    for t in terms:
        rows.append((t, fit(df, y, [u for u in terms if u != t]).ssr - rss, 1))
    tab = pd.DataFrame(rows, columns=["Source", "Sum of Squares", "df"]).set_index("Source")
    tab["Mean Square"] = tab["Sum of Squares"] / tab["df"]
    tab["F-value"] = tab["Mean Square"] / msr
    tab["p-value"] = fdist.sf(tab["F-value"], tab["df"], dfr)
    sspe, dfpe = pure_error(df, y)
    extra = [("Residual", rss, dfr, msr, np.nan, np.nan)]
    if dfpe > 0 and dfr - dfpe > 0:
        lof, dflof = rss - sspe, dfr - dfpe
        flof = (lof / dflof) / (sspe / dfpe)
        extra += [("Lack of Fit", lof, dflof, lof / dflof, flof, fdist.sf(flof, dflof, dfpe)),
                  ("Pure Error", sspe, dfpe, sspe / dfpe, np.nan, np.nan)]
    extra.append(("Cor Total", sst, len(df) - 1, np.nan, np.nan, np.nan))
    tab = pd.concat([tab, pd.DataFrame(extra, columns=["Source"] + list(tab.columns)).set_index("Source")])
    sig = []
    for s, p in tab["p-value"].items():
        if s == "Lack of Fit":
            sig.append("not significant" if p >= 0.05 else "significant")
        else:
            sig.append("" if np.isnan(p) else ("significant" if p < 0.05 else "not significant"))
    tab["Remarks"] = sig
    tab.index = pd.Index([pretty(i) for i in tab.index], name="Source")
    return tab


def pretty(t):
    names = {"D": "D-Feed", "A": "A-Speed", "B": "B-Time", "C": "C-BPR"}
    if t in names:
        return names[t]
    if "*" in t:
        a, b = t.split("*")
        return f"{a}²" if a == b else f"{a}{b}"
    return t


def summary(df, y, m, p):
    X = m.model.exog
    h = np.diag(X @ np.linalg.pinv(X.T @ X) @ X.T)
    press = float(np.sum((m.resid / (1 - h)) ** 2))
    sst = ((df[y] - df[y].mean()) ** 2).sum()
    sd = np.sqrt(m.mse_resid)
    yhat = m.fittedvalues
    adeq = (yhat.max() - yhat.min()) / np.sqrt(p * m.mse_resid / len(df))
    return pd.Series({"Std. Dev.": sd, "Mean": df[y].mean(), "C.V. %": 100 * sd / df[y].mean(),
                      "PRESS": press, "R-Squared": m.rsquared, "Adj R-Squared": m.rsquared_adj,
                      "Pred R-Squared": 1 - press / sst, "Adeq Precision": adeq})


def equation(m, terms, coded=True):
    s = f"{m.params['const']:.5g}"
    for t in terms:
        c = m.params[t]
        if coded:
            name = pretty(t).replace("-Feed", "").replace("-Speed", "").replace("-Time", "").replace("-BPR", "")
        else:
            name = "·".join(("Feed" if p == "D" else [f for f, L in LET.items() if L == p][0]) for p in t.split("*"))
        s += f" {'+' if c >= 0 else '-'} {abs(c):.5g}·{name}"
    return s


def split_plot(df, y):
    """Sequential (Type I) split-plot ANOVA built from explicit column blocks, so every df is a true rank.
    Feed is the whole-plot factor and is tested against batch(feed); everything else against the residual."""
    dummies = lambda col: pd.get_dummies(df[col].astype(str), drop_first=True).astype(float).values
    blocks = [("Feed (whole plot)", df[["D"]].values),
              ("Batch(feed) = whole-plot error", dummies("batch")),
              ("Speed", dummies("speed")), ("Time", dummies("time")), ("BPR", dummies("bpr")),
              ("Feed × Speed", (df["D"] * df["A"]).values[:, None]),
              ("Feed × Time", (df["D"] * df["B"]).values[:, None]),
              ("Feed × BPR", (df["D"] * df["C"]).values[:, None])]
    yv = df[y].values.astype(float)
    X = np.ones((len(df), 1))
    rss_prev = ((yv - yv.mean()) ** 2).sum(); rank_prev = 1; rows = []
    for name, B in blocks:
        X = np.column_stack([X, B])
        rank = np.linalg.matrix_rank(X)
        beta = np.linalg.lstsq(X, yv, rcond=None)[0]
        rss = ((yv - X @ beta) ** 2).sum()
        rows.append([name, rank - rank_prev, rss_prev - rss]); rss_prev, rank_prev = rss, rank
    df_res = len(df) - rank_prev
    rows.append(["Residual (sub-plot error)", df_res, rss_prev])
    tab = pd.DataFrame(rows, columns=["Source", "df", "Sum of Squares"]).set_index("Source")
    tab = tab[tab["df"] > 0].copy()
    tab["Mean Square"] = tab["Sum of Squares"] / tab["df"]
    ms_res = tab.loc["Residual (sub-plot error)", "Mean Square"]
    tab["F-value"] = tab["Mean Square"] / ms_res
    tab["p-value"] = fdist.sf(tab["F-value"], tab["df"], df_res)
    tab.loc["Residual (sub-plot error)", ["F-value", "p-value"]] = np.nan
    if "Batch(feed) = whole-plot error" in tab.index:
        wp = tab.loc["Batch(feed) = whole-plot error"]
        F = tab.loc["Feed (whole plot)", "Mean Square"] / wp["Mean Square"]
        tab.loc["Feed (whole plot)", ["F-value", "p-value"]] = [F, fdist.sf(F, 1, wp["df"])]
        tab.loc["Batch(feed) = whole-plot error", ["F-value", "p-value"]] = np.nan
    tab["% contribution"] = 100 * tab["Sum of Squares"] / tab["Sum of Squares"].sum()
    return tab


# ------------------------------------------------------------------------------ desirability
def d_value(v, goal, lo, hi):
    if goal == "maximize":
        d = (v - lo) / (hi - lo)
    elif goal == "minimize":
        d = (hi - v) / (hi - lo)
    else:
        d = np.ones_like(v)
    return np.clip(d, 0, 1)


def predict_terms(terms, m, A, B, C, D):
    g = pd.DataFrame({"A": A, "B": B, "C": C, "D": D})
    X = sm.add_constant(term_columns(g, terms), has_constant="add")
    return X.values @ m.params.values, X.values


# ------------------------------------------------------------------------------ main
def main(path):
    os.makedirs(OUT, exist_ok=True)
    raw = pd.read_csv(path)
    df, feeds = prepare(raw)
    resp = [r for r in RESPONSES if r in df.columns and df[r].notna().all()]
    if not resp:
        sys.exit("No complete response columns: fill in the run table first.")
    W = pd.ExcelWriter("stage0b_results.xlsx")

    lv = pd.DataFrame([(f.capitalize(), u, L, lo, c, hi) for f, (lo, c, hi, u, L) in FACTORS.items()],
                      columns=["Factor", "Unit", "Code", "Level -1", "Level 0", "Level +1"])
    lv.loc[len(lv)] = ["Feed (stage-1 setting)", "-", "D", feeds[0], "", feeds[1]]
    lv.to_excel(W, sheet_name="T1 Factors and levels", index=False)
    raw.to_excel(W, sheet_name="T2 Design with responses", index=False)

    models, sums, eqs, sp_all = {}, [], [], []
    for y in resp:
        label, unit, *_ = RESPONSES[y]
        terms, m = reduce_model(df, y)
        ma = fit(df, y, terms, actual=True)
        models[y] = (terms, m)
        model_anova(df, y, terms, m).to_excel(W, sheet_name=f"ANOVA {label}"[:31])
        s = summary(df, y, m, len(terms) + 1); s.name = label; sums.append(s)
        eqs.append((label, equation(m, terms, True), equation(ma, terms, False)))
        sp = split_plot(df, y); sp.insert(0, "Response", label); sp_all.append(sp)

        # predicted vs actual
        fig, ax = plt.subplots(figsize=(3.8, 3.6))
        ax.scatter(df[y], m.fittedvalues, c=np.where(df["D"] < 0, "tab:blue", "tab:orange"), s=28, edgecolor="k", lw=0.4)
        lims = [min(df[y].min(), m.fittedvalues.min()), max(df[y].max(), m.fittedvalues.max())]
        ax.plot(lims, lims, "k-", lw=0.8)
        ax.set_xlabel(f"Actual ({unit})"); ax.set_ylabel(f"Predicted ({unit})"); ax.set_title(f"Predicted vs. Actual – {label}", fontsize=9)
        fig.tight_layout(); fig.savefig(f"{OUT}/pred_vs_actual_{y}.png", dpi=220); plt.close(fig)

        # response surfaces: speed x time at BPR centre, one per feed
        g = np.linspace(-1, 1, 41)
        AA, BB = np.meshgrid(g, g)
        fig = plt.figure(figsize=(8, 3.6))
        for k, (fd, Dv) in enumerate([(feeds[0], -1), (feeds[1], 1)]):
            Z, _ = predict_terms(terms, m, AA.ravel(), BB.ravel(), np.zeros(AA.size), np.full(AA.size, Dv))
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            ax.plot_surface(250 + 50 * (AA + 1), 12 + 6 * (BB + 1), Z.reshape(AA.shape), cmap="jet", linewidth=0, alpha=0.9)
            ax.set_xlabel("A: Speed (rpm)", fontsize=7); ax.set_ylabel("B: Time (h)", fontsize=7); ax.set_zlabel(f"{label} ({unit})", fontsize=7)
            ax.tick_params(labelsize=6); ax.set_title(f"Feed {fd}, C: BPR = 10:1", fontsize=8)
        fig.tight_layout(); fig.savefig(f"{OUT}/surface_{y}.png", dpi=220); plt.close(fig)

    pd.concat(sums, axis=1).rename_axis("Statistic").to_excel(W, sheet_name="Model summary statistics")
    pd.DataFrame(eqs, columns=["Response", "Coded factors (A, B, C = -1..+1; D: A = -1, B = +1)",
                               "Actual factors (speed rpm, time h, bpr; Feed = -1 / +1)"]).to_excel(W, sheet_name="Equations", index=False)
    pd.concat(sp_all).to_excel(W, sheet_name="Split-plot ANOVA")

    # ---- desirability: constraints (as Design-Expert) and solutions
    cons = []
    for f, (lo, c, hi, u, L) in FACTORS.items():
        goal, imp = FACTOR_GOALS.get(f, ("is in range", 3))
        cons.append((f"{L}: {f}", goal, lo, hi, 1, 1, imp))
    cons.append(("D: feed", "is in range", feeds[0], feeds[1], 1, 1, 3))
    lims = {}
    for y in resp:
        label, unit, goal, imp = RESPONSES[y]
        lo, hi = df[y].min(), df[y].max(); lims[y] = (lo, hi)
        cons.append((f"{label} ({unit})", goal, lo, hi, 1, 1, imp))
    pd.DataFrame(cons, columns=["Name", "Goal", "Lower Limit", "Upper Limit", "Lower Weight", "Upper Weight", "Importance"]
                 ).to_excel(W, sheet_name="Constraints", index=False)

    g = np.round(np.arange(-1, 1 + 1e-9, GRID_STEP), 4)
    A, B, C = [a.ravel() for a in np.meshgrid(g, g, g, indexing="ij")]
    sols = []
    for fd, Dv in [(feeds[0], -1.0), (feeds[1], 1.0)]:
        Dc = np.full(A.size, Dv)
        logD, wsum, preds = np.zeros(A.size), 0, {}
        for y in resp:
            terms, m = models[y]
            preds[y], _ = predict_terms(terms, m, A, B, C, Dc)
            _, _, goal, imp = RESPONSES[y]
            logD += imp * np.log(np.maximum(d_value(preds[y], goal, *lims[y]), 1e-12)); wsum += imp
        for f, (goal, imp) in FACTOR_GOALS.items():
            x = {"speed": A, "time": B, "bpr": C}[f]
            logD += imp * np.log(np.maximum(d_value(x, goal, -1, 1), 1e-12)); wsum += imp
        Dtot = np.exp(logD / wsum)
        for i in np.argsort(-Dtot)[:10]:
            r = {"Feed": fd, "Speed (rpm)": 300 + 50 * A[i], "Time (h)": 18 + 6 * B[i], "BPR": 10 + 5 * C[i]}
            r.update({f"{RESPONSES[y][0]} ({RESPONSES[y][1]})": preds[y][i] for y in resp})
            r["Desirability"] = Dtot[i]; r["_c"] = (A[i], B[i], C[i], Dv)
            sols.append(r)
    sol = pd.DataFrame(sols).sort_values("Desirability", ascending=False).reset_index(drop=True)
    sol.index = sol.index + 1
    best = sol.iloc[0]
    sol.drop(columns="_c").head(10).to_excel(W, sheet_name="Desirable solutions", index_label="Number")

    # ---- confirmation: predicted value and 95 % prediction interval at the selected solution
    a, b, c, d = best["_c"]
    conf = [("Feed", best["Feed"], "", "", "", ""), ("Speed (rpm)", round(best["Speed (rpm)"], 1), "", "", "", ""),
            ("Time (h)", round(best["Time (h)"], 1), "", "", "", ""), ("BPR", round(best["BPR"], 2), "", "", "", "")]
    for y in resp:
        terms, m = models[y]
        yp, X = predict_terms(terms, m, [a], [b], [c], [d])
        se = np.sqrt(m.mse_resid * (1 + X @ np.linalg.pinv(m.model.exog.T @ m.model.exog) @ X.T))[0, 0]
        tq = tdist.ppf(0.975, m.df_resid)
        conf.append((f"{RESPONSES[y][0]} ({RESPONSES[y][1]})", round(yp[0], 4), round(yp[0] - tq * se, 4), round(yp[0] + tq * se, 4), "", ""))
    pd.DataFrame(conf, columns=["Parameter", "Optimized / predicted value", "95 % PI low", "95 % PI high",
                                "Experimentally observed value", "% error"]).to_excel(W, sheet_name="Confirmation test", index=False)
    W.close()
    print(pd.concat(sums, axis=1).round(4))
    print(sol.drop(columns="_c").head(10).round(3).to_string())
    print("Saved stage0b_results.xlsx and figures in", OUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "stage0b_runs.csv")
