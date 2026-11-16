"""2026 美国参议院回测模型（模拟时钟：2026-02-15）。

方法：每州给出民主党（或非共和党候选人）获胜概率 p，换算成预期得票差 mu；
每次模拟加一个全国共同冲击（各州同向）和一个州独立冲击，统计席位分布。
"""
import numpy as np
from statistics import NormalDist

_ppf = NormalDist().inv_cdf

SIGMA_NAT = 4.5    # 全国共同摆动（百分点），距选举约 9 个月
SIGMA_STATE = 6.5  # 州特有误差
SIGMA_TOTAL = (SIGMA_NAT**2 + SIGMA_STATE**2) ** 0.5
N = 200_000

D_NOT_UP = 34   # 民主党（含与其结盟的独立派）不改选席位
R_NOT_UP = 31   # 共和党不改选席位

# (州, 现任党派, 非共和党一方获胜概率, 是否独立候选人)
RACES = [
    # 共和党防守，竞争性
    ("NC", "R", 0.62, False), ("ME", "R", 0.50, False), ("OH*", "R", 0.40, False),
    ("AK", "R", 0.30, False), ("TX", "R", 0.22, False), ("IA", "R", 0.18, False),
    ("NE", "R", 0.15, True),  ("FL*", "R", 0.08, False),
    # 共和党防守，安全
    ("KS", "R", 0.05, False), ("MT", "R", 0.05, False), ("KY", "R", 0.04, False),
    ("MS", "R", 0.04, False), ("SC", "R", 0.04, False), ("LA", "R", 0.03, False),
    ("AL", "R", 0.02, False), ("AR", "R", 0.02, False), ("TN", "R", 0.02, False),
    ("ID", "R", 0.01, False), ("OK", "R", 0.01, False), ("SD", "R", 0.01, False),
    ("WV", "R", 0.01, False), ("WY", "R", 0.01, False),
    # 民主党防守
    ("GA", "D", 0.70, False), ("MI", "D", 0.55, False), ("NH", "D", 0.75, False),
    ("MN", "D", 0.88, False), ("NM", "D", 0.96, False), ("VA", "D", 0.96, False),
    ("CO", "D", 0.97, False), ("NJ", "D", 0.97, False), ("DE", "D", 0.99, False),
    ("IL", "D", 0.99, False), ("MA", "D", 0.99, False), ("OR", "D", 0.99, False),
    ("RI", "D", 0.99, False),
]
assert len(RACES) == 35

def simulate(races, sigma_nat=SIGMA_NAT, sigma_state=SIGMA_STATE, seed=2026):
    rng = np.random.default_rng(seed)
    total = (sigma_nat**2 + sigma_state**2) ** 0.5
    mu = np.array([_ppf(p) * total for _, _, p, _ in races])
    nat = rng.normal(0, sigma_nat, size=(N, 1))
    st = rng.normal(0, sigma_state, size=(N, len(races)))
    win = (mu + nat + st) > 0              # 非共和党一方获胜
    indep = np.array([i for *_, i in races])
    d = D_NOT_UP + win[:, ~indep].sum(1)
    i = win[:, indep].sum(1)
    r = 100 - d - i
    return d, i, r

if __name__ == "__main__":
    d, i, r = simulate(RACES)
    print(f"共和党保住多数（≥50 席，副总统万斯打破平局）: {np.mean(r >= 50):.1%}")
    print(f"民主党拿下多数（≥51 席）:                    {np.mean(d >= 51):.1%}")
    print(f"无党过半（奥斯本决定归属）:                  {np.mean((r < 50) & (d < 51)):.1%}")
    print(f"民主党席位中位数: {np.median(d):.0f}，80% 区间: {np.percentile(d,10):.0f}–{np.percentile(d,90):.0f}")
    vals, cnts = np.unique(d, return_counts=True)
    print("民主党席位分布:", {int(v): f"{c/len(d):.1%}" for v, c in zip(vals, cnts) if c/len(d) > 0.01})

    print("\n敏感性分析（州胜率不变，只改变各州之间的相关程度）:")
    for sn, ss in ((2.5, 7.5), (6.5, 4.5), (8.0, 0.1)):
        d, i, r = simulate(RACES, sn, ss)
        print(f"  全国冲击 {sn} / 州冲击 {ss}: 共和党保住 {np.mean(r>=50):.1%}，民主党过半 {np.mean(d>=51):.1%}")

    print("\n情景：缅因、北卡任一变化 ±10 个百分点胜率")
    for st_, dp in (("ME", 0.10), ("ME", -0.10), ("NC", 0.10), ("NC", -0.10), ("MI", 0.10), ("MI", -0.10)):
        races = [(a, b, min(max(p + dp, 0.01), 0.99) if a == st_ else p, c) for a, b, p, c in RACES]
        d, i, r = simulate(races)
        print(f"  {st_} {dp:+.0%}: 民主党过半 {np.mean(d>=51):.1%}")
