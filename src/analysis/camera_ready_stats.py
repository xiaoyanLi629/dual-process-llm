"""
Camera-ready statistics requested by the BIBM 2026 reviewers.

Reads the frozen trial-level results and prints/saves:
  R3-1  per-intuitive-benchmark ZS vs CoT accuracy + hedged-response counts
  R3-2  item-level bootstrap 95% CIs for the Table IV main effects
  R1-1  Prompt x Category interaction with benchmark / item as blocking factors
  R1-4  n per ANOVA cell
  R3-3  token accounting (total vs estimated completion tokens)
  R4-3  responses truncated at the completion cap
  R2-1/R2-2  MedQA table (accuracy, tokens, TER, calibration) + USMLE-step split

Usage:  python src/analysis/camera_ready_stats.py
Output: results/bibm_2026/camera_ready_stats.json
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "bibm_2026"
FACT = RES / "main_experiment_v2" / "ablation_factorial_final.json"
MULTI = RES / "multi_model" / "multi_model_final.json"
MEDQA = RES / "medqa" / "medqa_results.json"
OUT = RES / "camera_ready_stats.json"

RNG = np.random.default_rng(20261025)
N_BOOT = 10000

# A response is "hedged" when it declines to commit to one option.
HEDGE_RE = re.compile(
    r"insufficient|cannot (be )?determine|can't determine|cannot be determined|"
    r"not enough (information|context)|information is needed|unable to determine|"
    r"undetermin|indetermin|undecided|unclear|ambiguous|impossible to|"
    r"without (additional |more |further |specific )?(context|information|criteria)|"
    r"lack of context|needs context|difficult to determine|no (clear|definitive)|"
    r"not specified|depends on|both options|neither option",
    re.I,
)

CAT_NAME = {"system1": "intuitive", "system2": "analytical", "conflict": "conflict"}


def source_of(task_id: str) -> str:
    src = task_id.rsplit("_", 1)[0]
    return "novel" if src.startswith("novel") else src


def load_factorial() -> pd.DataFrame:
    d = json.load(open(FACT))
    rows = []
    for cid, c in d["conditions"].items():
        cond = c["condition"]
        for t in c["trials"]:
            rows.append(dict(
                cond=cid,
                model=int(cond["model"] == "gpt-4o"),
                lowT=int(cond["temperature"] < 0.5),
                cot=int(cond["prompt"] == "cot"),
                category=CAT_NAME[t["task_category"]],
                source=source_of(t["task_id"]),
                item=t["task_id"],
                correct=int(bool(t["is_correct"])),
                conf=float(t["confidence"]),
                tokens=int(t["tokens_used"]),
                parse_error=bool(t["parse_error"]),
                hedged=bool(HEDGE_RE.search(str(t["answer"]))),
            ))
    return pd.DataFrame(rows)


def interaction_F(df: pd.DataFrame, block: str = "none") -> dict:
    """F-test for Prompt x Category by nested-model comparison on explicit
    full-rank designs. Benchmarks and items are nested in categories, so a
    formula with both C(category) and C(source) is rank-deficient and gives
    version-dependent type-II ANOVA results; this construction does not."""
    y = df["correct"].to_numpy(float)
    cols = [np.ones(len(df)), df["cot"].to_numpy(float)]
    levels = {"none": ("category", ["analytical", "conflict"]),
              "source": ("source", sorted(df["source"].unique())[1:]),
              "item": ("item", sorted(df["item"].unique())[1:])}[block]
    col, lv = levels
    cols += [(df[col] == v).to_numpy(float) for v in lv]
    X0 = np.column_stack(cols)
    X1 = np.column_stack([X0] + [df["cot"].to_numpy(float) * (df["category"] == c).to_numpy(float)
                                 for c in ("analytical", "conflict")])
    ssr = lambda X: float(np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2))
    r0, r1 = np.linalg.matrix_rank(X0), np.linalg.matrix_rank(X1)
    s0, s1 = ssr(X0), ssr(X1)
    q, dfr = int(r1 - r0), int(len(y) - r1)
    F = ((s0 - s1) / q) / (s1 / dfr)
    from scipy import stats as sps
    return dict(F=F, df1=q, df2=dfr, p=float(sps.f.sf(F, q, dfr)), eta_p2=(s0 - s1) / (s0 - s1 + s1))


def main_effect(df: pd.DataFrame, factor: str) -> float:
    return 100 * (df.loc[df[factor] == 1, "correct"].mean() - df.loc[df[factor] == 0, "correct"].mean())


def boot_effects(df: pd.DataFrame) -> dict:
    """Item-level cluster bootstrap: resample items (each carries its 8 trials)."""
    out = {}
    wide = df.pivot_table(index="item", columns="cond", values="correct")
    cats = df.groupby("item")["category"].first()
    conds = {c: df[df.cond == c][["model", "lowT", "cot"]].iloc[0] for c in wide.columns}
    for factor in ("cot", "model", "lowT"):
        hi = [c for c in wide.columns if conds[c][factor] == 1]
        lo = [c for c in wide.columns if conds[c][factor] == 0]
        item_eff = 100 * (wide[hi].mean(axis=1) - wide[lo].mean(axis=1))
        res = {}
        groups = {cat: item_eff[cats == cat].values for cat in ("analytical", "intuitive", "conflict")}
        for cat, v in groups.items():
            bs = [RNG.choice(v, len(v)).mean() for _ in range(N_BOOT)]
            res[cat] = dict(est=float(v.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
        # overall = mean of the three category means (as in Table IV)
        bs = [np.mean([RNG.choice(v, len(v)).mean() for v in groups.values()]) for _ in range(N_BOOT)]
        est = float(np.mean([v.mean() for v in groups.values()]))
        res["overall"] = dict(est=est, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
        out[factor] = res
    return out


def calibration(correct, conf, bins=10):
    correct, conf = np.asarray(correct, float), np.asarray(conf, float)
    brier = float(np.mean((conf - correct) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece = sum(abs(conf[idx == b].mean() - correct[idx == b].mean()) * (idx == b).mean()
              for b in range(bins) if (idx == b).any())
    return brier, float(ece)


def main():
    stats = {}
    df = load_factorial()

    # ---- R1-4: cell sizes -------------------------------------------------
    stats["cell_n"] = dict(
        three_way_cells=int(df.groupby(["model", "lowT", "cot"]).size().iloc[0]),
        cond_x_category=int(df.groupby(["cond", "category"]).size().iloc[0]),
        prompt_x_category=int(df.groupby(["cot", "category"]).size().iloc[0]),
        n_total=len(df),
        n_no_wg=int((df.source != "winogrande").sum()),
    )

    # ---- R3-1: per intuitive benchmark ------------------------------------
    intu = df[df.category == "intuitive"]
    per_src = []
    for src, g in intu.groupby("source"):
        n_items = g["item"].nunique()
        zs, cot = g[g.cot == 0], g[g.cot == 1]
        per_src.append(dict(
            source=src, n_items=int(n_items),
            zs_acc=100 * zs.correct.mean(), cot_acc=100 * cot.correct.mean(),
            delta=100 * (cot.correct.mean() - zs.correct.mean()),
            hedged_zs=int(zs.hedged.sum()), hedged_cot=int(cot.hedged.sum()),
            hedged_cot_per_cond=[int(cot[cot.cond == c].hedged.sum()) for c in sorted(cot.cond.unique())],
        ))
    stats["intuitive_by_source"] = per_src
    # share of the -11.1 decrement attributable to WinoGrande (weighted by item share)
    tot = 100 * (intu[intu.cot == 1].correct.mean() - intu[intu.cot == 0].correct.mean())
    nowg = intu[intu.source != "winogrande"]
    tot_nowg = 100 * (nowg[nowg.cot == 1].correct.mean() - nowg[nowg.cot == 0].correct.mean())
    stats["intuitive_decrement"] = dict(all=tot, excluding_wg=tot_nowg)

    # ---- R1-1: clustering / blocking sensitivity ---------------------------
    stats["interaction"] = dict(
        base=interaction_F(df),
        benchmark_block=interaction_F(df, "source"),
        item_block=interaction_F(df, "item"),
        base_no_wg=interaction_F(df[df.source != "winogrande"]),
        benchmark_block_no_wg=interaction_F(df[df.source != "winogrande"], "source"),
    )
    # per-benchmark CoT effect (leave-one-benchmark-out sign check)
    loo = {}
    for src in df.source.unique():
        sub = df[df.source != src]
        loo[src] = interaction_F(sub, "source")
    worst = min(loo, key=lambda k: loo[k]["F"])
    stats["interaction"]["leave_one_benchmark_out_min"] = dict(dropped=worst, **loo[worst])
    stats["interaction"]["leave_one_benchmark_out_F"] = loo

    # 3-way factorial with item as a blocking factor (items repeat across all 8 cells)
    fit = smf.ols("correct ~ C(model) * C(lowT) * C(cot) + C(item)", data=df).fit()
    tab = anova_lm(fit, typ=2)
    ss_res = tab.loc["Residual", "sum_sq"]
    stats["three_way_item_block"] = {
        k: dict(F=float(tab.loc[k, "F"]), p=float(tab.loc[k, "PR(>F)"]),
                eta_p2=float(tab.loc[k, "sum_sq"] / (tab.loc[k, "sum_sq"] + ss_res)))
        for k in ("C(model)", "C(lowT)", "C(cot)")
    } | {"df2": int(tab.loc["Residual", "df"])}

    # ---- multi-model: hedging + WG-excluded intuitive gaps -----------------
    stats["multi_model"] = multi_model_stats()

    # ---- R3-2: bootstrap CIs for main effects -----------------------------
    stats["main_effects_ci"] = boot_effects(df)

    # ---- R3-3 / R4-3: tokens ----------------------------------------------
    tok = {}
    for c in ("C5", "C8"):
        v = df[df.cond == c].tokens
        tok[c] = dict(mean=float(v.mean()), median=float(v.median()),
                      p90=float(v.quantile(.9)), max=int(v.max()))
    tok["ratio_total"] = tok["C8"]["mean"] / tok["C5"]["mean"]
    tok["parse_errors_by_cond"] = {c: int(g.parse_error.sum()) for c, g in df.groupby("cond")}
    tok["completion_estimate"] = completion_tokens(df)
    stats["tokens"] = tok

    # ---- R2-1 / R2-2: MedQA ------------------------------------------------
    stats["medqa"] = medqa_stats()

    # Keep a previously computed token estimate if this run could not recompute it
    if "skipped" in stats["tokens"]["completion_estimate"] and OUT.exists():
        prev = json.loads(OUT.read_text()).get("tokens", {}).get("completion_estimate", {})
        if "skipped" not in prev and prev:
            stats["tokens"]["completion_estimate"] = prev
    OUT.write_text(json.dumps(stats, indent=2, default=float))
    print(json.dumps(stats, indent=2, default=lambda x: round(float(x), 4)))


def completion_tokens(df: pd.DataFrame) -> dict:
    """The API logged usage.total_tokens (prompt + completion). Estimate the
    completion share by subtracting prompt tokens counted with the GPT-4o
    tokenizer (o200k_base; needs tiktoken>=0.7). Chat-format overhead ~11 tokens."""
    try:
        import sys
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
    except Exception as e:  # old tiktoken or not installed
        return {"skipped": str(e)}
    sys.path.insert(0, str(ROOT / "src"))
    from tasks.task_loader import TaskLoader
    src = (ROOT / "src" / "experiments" / "ablation_factorial.py").read_text()
    ns = {}
    exec(src[src.index("ZERO_SHOT_PROMPT ="):src.index("# ====", src.index("COT_PROMPT ="))], ns)
    loader = TaskLoader()
    loader.load()
    question = {t.id: t.question for cat in ("system1_tasks", "system2_tasks", "conflict_tasks")
                for t in loader.get_tasks(cat)}
    missing = set(df["item"]) - set(question)
    if missing:  # task pool not rebuilt (data/processed is not distributed)
        return {"skipped": f"{len(missing)} item texts missing; run data/prepare_bibm_dataset.py first"}
    out = {}
    for cid in sorted(df.cond.unique()):
        g = df[df.cond == cid]
        sysp = ns["COT_PROMPT"] if g.cot.iloc[0] else ns["ZERO_SHOT_PROMPT"]
        cap = 1000 if g.cot.iloc[0] else 150
        prompt = np.array([len(enc.encode(sysp)) + len(enc.encode(question[i])) + 11 for i in g["item"]])
        comp = g.tokens.values - prompt
        out[cid] = dict(prompt_mean=float(prompt.mean()), completion_mean=float(comp.mean()),
                        completion_median=float(np.median(comp)), completion_max=int(comp.max()),
                        cap=cap, n_within_5pct_of_cap=int((comp >= 0.95 * cap).sum()))
    out["ratio_C8_C5"] = out["C8"]["completion_mean"] / out["C5"]["completion_mean"]
    return out


def multi_model_stats() -> dict:
    d = json.load(open(MULTI))
    out = {}
    for fam, f in d["families"].items():
        res = {}
        for sysk in ("system1", "system2"):
            tr = f[sysk]["results_by_category"]["intuitive"]["trials"]
            wg = [t for t in tr if t["task_id"].startswith("winogrande")]
            nowg = [t for t in tr if not t["task_id"].startswith("winogrande")]
            res[sysk] = dict(
                n_wg=len(wg), hedged_wg=sum(bool(HEDGE_RE.search(str(t["answer"]))) for t in wg),
                acc_nowg=100 * np.mean([bool(t["is_correct"]) for t in nowg]),
                temperature=f[sysk].get("temperature"),
            )
        res["intuitive_gap_nowg"] = res["system2"]["acc_nowg"] - res["system1"]["acc_nowg"]
        out[fam] = res
    out["mean_intuitive_gap_nowg"] = float(np.mean([out[k]["intuitive_gap_nowg"] for k in d["families"]]))
    return out


def medqa_stats() -> dict:
    from datasets import load_dataset
    ds = load_dataset("GBaker/med_qa-usmle-4-options", split="test")
    m = json.load(open(MEDQA))
    out = {}
    for cid in ("C1", "C5", "C8"):
        tr = m[cid]["trials"]
        corr = [int(bool(t["is_correct"])) for t in tr]
        conf = [float(t["confidence"]) for t in tr]
        brier, ece = calibration(corr, conf)
        by_step = defaultdict(list)
        for t in tr:
            step = ds[int(t["task_id"].split("_")[1])]["meta_info"]
            by_step[step].append(int(bool(t["is_correct"])))
        out[cid] = dict(
            acc=100 * np.mean(corr), tokens=float(np.mean([t["tokens_used"] for t in tr])),
            conf=float(np.mean(conf)), overconf=100 * (np.mean(conf) - np.mean(corr)),
            brier=brier, ece=ece, parse_errors=int(sum(bool(t["parse_error"]) for t in tr)),
            by_step={k: dict(n=len(v), acc=100 * np.mean(v)) for k, v in by_step.items()},
        )
    for a, b in (("C1", "C8"), ("C5", "C8"), ("C1", "C5")):
        out[f"TER_{a}_{b}"] = (out[b]["acc"] - out[a]["acc"]) / (out[b]["tokens"] - out[a]["tokens"])
    return out


if __name__ == "__main__":
    main()
