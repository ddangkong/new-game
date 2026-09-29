"""Top-down level-design map of stage 1 from stage1_layout.json -> renders/stage1_map.png"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(os.path.join(HERE, "stage1_layout.json")))

plt.rcParams["font.family"] = ["Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"]
ZC = ["#e8b04a", "#8cc152", "#4fc1e9", "#37bc9b", "#ffffff", "#ec87c0"]

fig, ax = plt.subplots(figsize=(18, 7.5), facecolor="#1b2320")
ax.set_facecolor("#24302a")


def xy(p):
    p = np.asarray(p)
    return p[..., 0], -p[..., 2]  # glTF (x, up, z) -> map (x, y)


inst = L["instances"]
for name, col, sz, a in [("pine_a", "#1f3b2c", 2, 0.5), ("pine_b", "#1f3b2c", 2, 0.5), ("pine_c", "#1f3b2c", 2, 0.5),
                         ("oak_a", "#4f7d3a", 6, 0.8), ("oak_b", "#4f7d3a", 6, 0.8), ("oak_c", "#4f7d3a", 6, 0.8),
                         ("birch_a", "#c9d77a", 5, 0.8), ("birch_b", "#c9d77a", 5, 0.8),
                         ("lily_pad", "#5aa05a", 1, 0.6), ("lotus", "#f3a6c8", 3, 0.9)]:
    if name in inst:
        x, y = xy(inst[name]["pos"])
        ax.scatter(x, y, s=sz, c=col, alpha=a, linewidths=0)
for fl, col in [("flower_red", "#e0413a"), ("flower_yellow", "#f5c542"), ("flower_white", "#f2f2f2"),
                ("flower_violet", "#a574e0"), ("flower_blue", "#5b7cf0")]:
    if fl in inst:
        x, y = xy(inst[fl]["pos"])
        ax.scatter(x[::5], y[::5], s=0.6, c=col, alpha=0.7, linewidths=0)
# lake outline (ellipse used by the generator)
t = np.linspace(0, 2 * np.pi, 200)
ax.fill(930 + 240 * np.cos(t) * 0.9, 15 + 165 * np.sin(t) * 0.9, color="#2f6f8f", alpha=0.55, zorder=1)

pos = np.array(L["path"]["pos"])
px, py = xy(pos)
zone = np.array(L["path"]["zone"])
for zi, z in enumerate(L["zones"]):
    m = zone == zi
    ax.plot(px[m], py[m], color=ZC[zi], lw=3.2, solid_capstyle="round", zorder=5,
            label=f'{z["name_ko"]}  {int(z["t"][0])//60}:{int(z["t"][0])%60:02d}–{int(z["t"][1])//60}:{int(z["t"][1])%60:02d}  ({z["speed_mps"]} m/s)')
tt = np.array(L["path"]["t"])
for sec in range(0, 301, 30):
    i = int(np.argmin(np.abs(tt - sec)))
    ax.plot(px[i], py[i], "o", ms=6, color="white", zorder=6)
    ax.annotate(f"{sec//60}:{sec%60:02d}", (px[i], py[i]), xytext=(0, 12), textcoords="offset points",
                ha="center", color="white", fontsize=9, zorder=7)
for w in L["wind"]:
    s0, s1 = w["s"]
    s = np.array(L["path"]["s"])
    m = (s >= s0) & (s <= s1)
    ax.plot(px[m], py[m], color="#ffffff", lw=10, alpha=0.18, zorder=4)
lm = {"mother_dandelion": "어미 민들레 (출발)", "stone_ring": "선돌 고리 (바람 관문)", "ruin_arch": "고대 아치 (바람 관문)",
      "waterfall_lip": "빛의 폭포 (상승 기류)", "sacred_tree": "성스러운 나무", "spring_pond": "샘"}
for k, label in lm.items():
    x, y = xy(L["landmarks"][k])
    ax.plot(x, y, "*", ms=16, color="#ffe28a", mec="#6b4f00", zorder=8)
    ax.annotate(label, (x, y), xytext=(6, -20), textcoords="offset points", color="#ffe28a", fontsize=10, zorder=8)
x, y = xy(L["landing"])
ax.plot(x, y, "o", ms=14, mfc="none", mec="#ff9ad5", mew=2.5, zorder=9)
ax.annotate("착지점", (x, y), xytext=(10, 8), textcoords="offset points", color="#ff9ad5", fontsize=11, weight="bold")

ax.set_xlim(-200, 2250)
ax.set_ylim(-420, 420)
ax.set_aspect("equal")
ax.tick_params(colors="#889")
for sp in ax.spines.values():
    sp.set_color("#445")
ax.set_title(f'Stage 1 — 첫 바람의 골짜기   |   비행 경로 {L["path_length_m"]:.0f} m · {L["duration_s"]//60}분',
             color="white", fontsize=15, loc="left")
leg = ax.legend(loc="lower left", facecolor="#1b2320", edgecolor="#445", labelcolor="white", fontsize=10)
ax.text(2240, -405, "흰 띠 = 바람 구간 · 흰 점 = 30초 간격 · 단위 m", color="#aab", ha="right", fontsize=9)
plt.tight_layout()
os.makedirs(os.path.join(HERE, "renders"), exist_ok=True)
plt.savefig(os.path.join(HERE, "renders", "stage1_map.png"), dpi=110, facecolor=fig.get_facecolor())
