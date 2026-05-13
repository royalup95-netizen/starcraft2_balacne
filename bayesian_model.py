"""
============================================================
SC2 动态贝叶斯平衡性分析（最终论文稳定版）
============================================================

核心创新：
1. Bayesian Matchup Balance Model
2. Dynamic Time Evolution
3. Population Effect
4. 去除玩家skill层（解决不收敛）

最终模型：

logit(p_i)
=
β_matchup
+
γ_time
+
φ * population_effect

============================================================
"""

import os

# =========================
# 防止 Windows + threadpoolctl 崩溃
# =========================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# PyTensor CPU模式
os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float64"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# =========================
# matplotlib
# =========================
plt.style.use("ggplot")

# 中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1. 数据读取
# ============================================================
def load_data(path):

    print("=" * 60)
    print("读取数据")
    print("=" * 60)

    df = pd.read_excel(path)

    print(f"原始数据量: {len(df)}")

    # ========================================================
    # 只保留 2020-2021
    # ========================================================
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df[
        (df["date"].dt.year >= 2020)
        &
        (df["date"].dt.year <= 2021)
    ]

    # ========================================================
    # 使用完整种族名
    # ========================================================
    df["race1"] = df["race1_full"]
    df["race2"] = df["race2_full"]

    # ========================================================
    # winner 清洗
    # ========================================================
    df["winner"] = pd.to_numeric(df["winner"], errors="coerce")

    df = df.dropna(subset=["winner"])

    df = df[df["winner"].isin([1, 2])]

    df["winner"] = df["winner"].astype(int)

    # ========================================================
    # winner_race
    # ========================================================
    df["winner_race"] = np.where(
        df["winner"] == 1,
        df["race1"],
        df["race2"]
    )

    # ========================================================
    # 删除缺失
    # ========================================================
    df = df.dropna(
        subset=[
            "race1",
            "race2",
            "winner_race",
            "date"
        ]
    )

    # ========================================================
    # matchup
    # ========================================================
    df["matchup"] = [
        "".join(sorted([r1[0], r2[0]]))
        for r1, r2 in zip(df["race1"], df["race2"])
    ]

    # ========================================================
    # month index
    # ========================================================
    df["month"] = df["date"].dt.to_period("M")

    month_order = sorted(df["month"].unique())

    month_map = {
        m: i
        for i, m in enumerate(month_order)
    }

    df["month_idx"] = df["month"].map(month_map)

    print(f"\n清洗后数据量: {len(df)}")

    print("\n对阵分布:")
    print(df["matchup"].value_counts())

    print("\n月份分布:")
    print(df["month"].value_counts().sort_index())

    return df, month_order


# ============================================================
# 2. 构建动态贝叶斯模型
# ============================================================
def build_model(df):

    print("\n" + "=" * 60)
    print("构建动态贝叶斯模型")
    print("=" * 60)

    # ========================================================
    # matchup
    # ========================================================
    matchup_order = ["PT", "PZ", "TZ"]

    df = df[df["matchup"].isin(matchup_order)].copy()

    matchup_map = {
        m: i
        for i, m in enumerate(matchup_order)
    }

    matchup_idx = df["matchup"].map(matchup_map).values

    # ========================================================
    # 时间
    # ========================================================
    month_idx = df["month_idx"].values

    n_months = df["month_idx"].nunique()

    # ========================================================
    # population effect
    # ========================================================
    race_dist = df["race1"].value_counts(normalize=True)

    def pop_effect(row):

        return (
            race_dist[row["race1"]]
            -
            race_dist[row["race2"]]
        )

    pop = df.apply(pop_effect, axis=1).values

    print("\nPopulation effect range:")
    print(f"{pop.min():.3f} ~ {pop.max():.3f}")

    # ========================================================
    # y
    # ========================================================
    y = []

    for _, row in df.iterrows():

        if row["matchup"] == "PT":

            y.append(
                1 if row["winner_race"] == "Protoss"
                else 0
            )

        elif row["matchup"] == "PZ":

            y.append(
                1 if row["winner_race"] == "Protoss"
                else 0
            )

        elif row["matchup"] == "TZ":

            y.append(
                1 if row["winner_race"] == "Terran"
                else 0
            )

    y = np.array(y)

    print(f"\n建模数据量: {len(y)}")
    print(f"月份数量: {n_months}")

    # ========================================================
    # 贝叶斯模型
    # ========================================================
    with pm.Model() as model:

        # ====================================================
        # matchup effect
        # ====================================================
        beta = pm.Normal(
            "beta",
            mu=0,
            sigma=0.5,
            shape=3
        )

        # ====================================================
        # 时间随机游走
        # ====================================================
        sigma_time = pm.Exponential(
            "sigma_time",
            10
        )

        rw = pm.GaussianRandomWalk(
            "rw",
            sigma=sigma_time,
            shape=n_months
        )

        # ====================================================
        # population effect
        # ====================================================
        phi = pm.Normal(
            "phi",
            mu=0,
            sigma=0.5
        )

        # ====================================================
        # logit
        # ====================================================
        logit = (
            beta[matchup_idx]
            +
            rw[month_idx]
            +
            phi * pop
        )

        p = pm.math.sigmoid(logit)

        pm.Bernoulli(
            "obs",
            p=p,
            observed=y
        )

    return (
        model,
        matchup_order,
        month_order
    )


# ============================================================
# 3. MCMC
# ============================================================
def run_mcmc(model):

    print("\n" + "=" * 60)
    print("开始 MCMC 采样")
    print("=" * 60)

    with model:

        trace = pm.sample(

            draws=1000,
            tune=1500,

            chains=4,

            cores=1,

            target_accept=0.95,

            random_seed=42,

            compute_convergence_checks=True,

            progressbar=True
        )

    print("\n采样完成")

    print(
        az.summary(
            trace,
            var_names=[
                "beta",
                "phi",
                "sigma_time"
            ]
        )
    )

    return trace


# ============================================================
# 4. 后验分析
# ============================================================
def posterior_analysis(trace):

    print("\n" + "=" * 60)
    print("后验分析")
    print("=" * 60)

    beta = trace.posterior["beta"].values.reshape(-1, 3)

    matchups = ["PT", "PZ", "TZ"]

    for i, mu in enumerate(matchups):

        samples = 1 / (1 + np.exp(-beta[:, i]))

        mean = samples.mean()

        hdi = az.hdi(samples, hdi_prob=0.94)

        print(f"\n{mu}")

        print(f"后验胜率: {mean:.2%}")

        print(
            f"94% HDI: "
            f"[{hdi[0]:.2%}, {hdi[1]:.2%}]"
        )

        if hdi[0] > 0.5:

            print("🔥 前者显著优势")

        elif hdi[1] < 0.5:

            print("🔥 后者显著优势")

        else:

            print("✅ 基本平衡")

    # ========================================================
    # population effect
    # ========================================================
    print("\nPopulation Effect:")

    print(
        az.summary(
            trace,
            var_names=["phi"]
        )
    )

    phi_mean = (
        trace.posterior["phi"]
        .values
        .mean()
    )

    if phi_mean > 0:

        print("人口更多的种族具有统计优势")

    else:

        print("人口更多并未带来统计优势")


# ============================================================
# 5. 时间动态图
# ============================================================
def plot_time_dynamics(trace, month_order):

    print("\n生成时间动态图...")

    rw = (
        trace.posterior["rw"]
        .values
        .reshape(-1, len(month_order))
    )

    mean = rw.mean(axis=0)

    lower = np.percentile(rw, 3, axis=0)

    upper = np.percentile(rw, 97, axis=0)

    plt.figure(figsize=(16, 8))

    x = np.arange(len(month_order))

    plt.plot(
        x,
        mean,
        linewidth=3,
        label="Latent Balance Trend"
    )

    plt.fill_between(
        x,
        lower,
        upper,
        alpha=0.3
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=2
    )

    plt.xticks(
        x,
        [str(m) for m in month_order],
        rotation=45
    )

    plt.ylabel("Latent Balance Effect")

    plt.xlabel("Month")

    plt.title(
        "SC2 Dynamic Balance Evolution (2020-2021)",
        fontsize=18
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "time_dynamic.png",
        dpi=300
    )

    print("时间动态图已保存：time_dynamic.png")


# ============================================================
# 6. 对阵图
# ============================================================
def plot_matchup(trace):

    beta = trace.posterior["beta"].values.reshape(-1, 3)

    plt.figure(figsize=(12, 7))

    names = ["PvT", "PvZ", "TvZ"]

    for i, name in enumerate(names):

        samples = 1 / (1 + np.exp(-beta[:, i]))

        az.plot_kde(
            samples,
            label=name
        )

    plt.axvline(
        0.5,
        linestyle="--",
        linewidth=2
    )

    plt.title(
        "Posterior Matchup Winrates",
        fontsize=18
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "posterior_matchup.png",
        dpi=300
    )

    print("对阵图已保存：posterior_matchup.png")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("SC2 动态贝叶斯平衡性分析")
    print("=" * 60)

    # ========================================================
    # 数据
    # ========================================================
    df, month_order = load_data(
        "sc2_data_2025_2026.xlsx"
    )

    # ========================================================
    # 模型
    # ========================================================
    (
        model,
        matchups,
        month_order
    ) = build_model(df)

    # ========================================================
    # MCMC
    # ========================================================
    trace = run_mcmc(model)

    # ========================================================
    # 分析
    # ========================================================
    posterior_analysis(trace)

    # ========================================================
    # 图
    # ========================================================
    plot_time_dynamics(
        trace,
        month_order
    )

    plot_matchup(trace)

    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)