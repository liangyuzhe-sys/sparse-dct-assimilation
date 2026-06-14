#!/usr/bin/env python3
"""
生成 PPT 用的两张结果图:
  figures/imp_bar.pdf     -- 各变量误差改进(DCT 零训练 + CNN 残差)
  figures/calibration.pdf -- 90% 可信区间覆盖率(校准)
配色与 ustcbeamer 蓝主题一致。默认英文标签(避免服务器无中文字体乱码);
想换中文标签见文件底部注释。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

os.makedirs("figures", exist_ok=True)

# ====================== 数据：核对 / 替换为你昨天的真实结果 ======================
# 每个变量: (方法, imp%, imp 的 std, cov90)
RESULTS = {
    "z500": ("DCT", 30.5, 1.0, 0.90),   # <- z500 的 std / cov90 请按你的结果核对
    "t500": ("CNN", 10.1, 1.1, 0.91),
    "t850": ("CNN",  6.4, 0.5, 0.91),
    "u700": ("CNN",  5.8, 0.3, 0.91),
    "v700": ("CNN",  6.0, 0.2, 0.90),
    "q500": ("CNN",  4.8, 0.5, 0.92),
    "q700": ("CNN",  4.3, 0.3, 0.90),
    "q850": ("CNN",  3.5, 0.3, 0.91),
    "q925": ("CNN",  3.0, 0.2, 0.92),
}
ORDER = ["z500", "t500", "t850", "u700", "v700", "q500", "q700", "q850", "q925"]
# =============================================================================

# ---- 全局风格 ----
plt.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.5,
    "savefig.dpi": 200,
})
C_DCT = "#1f5fa8"   # 蓝：零训练 DCT
C_CNN = "#e08a1e"   # 橙：CNN 残差
C_REF = "#c0392b"   # 红：参考线

methods = [RESULTS[v][0] for v in ORDER]
imps    = np.array([RESULTS[v][1] for v in ORDER])
stds    = np.array([RESULTS[v][2] for v in ORDER])
covs    = np.array([RESULTS[v][3] for v in ORDER])
colors  = [C_DCT if m == "DCT" else C_CNN for m in methods]
x = np.arange(len(ORDER))

# ============================ 图 1：误差改进 ============================
fig, ax = plt.subplots(figsize=(8.4, 4.3))
ax.bar(x, imps, yerr=stds, capsize=4, color=colors,
       edgecolor="white", linewidth=1.0,
       error_kw=dict(ecolor="#333333", lw=1.2))
ax.axhline(0, color="#888888", lw=0.8)
for xi, v, s in zip(x, imps, stds):
    ax.text(xi, v + s + 0.4, f"{v:.1f}", ha="center", va="bottom",
            fontsize=9.5, color="#222222")
ax.set_xticks(x)
ax.set_xticklabels(ORDER)
ax.set_ylabel("Error reduction  imp (%)")
ax.set_title("Assimilation skill: zero-training DCT (large-scale) + lightweight CNN",
             fontsize=12.5, pad=10)
ax.set_ylim(0, max(imps + stds) * 1.18)
ax.legend(handles=[Patch(color=C_DCT, label="DCT (zero-training)"),
                   Patch(color=C_CNN, label="CNN (residual)")],
          frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("figures/imp_bar.pdf", bbox_inches="tight")
print("saved figures/imp_bar.pdf")
plt.close(fig)

# ============================ 图 2：校准 ============================
fig, ax = plt.subplots(figsize=(8.4, 4.3))
ax.axhspan(0.88, 0.92, color=C_REF, alpha=0.08)          # 可接受带
ax.bar(x, covs, color=colors, edgecolor="white", linewidth=1.0)
ax.axhline(0.90, color=C_REF, ls="--", lw=1.5, label="Nominal 0.90")
for xi, c in zip(x, covs):
    ax.text(xi, c + 0.004, f"{c:.2f}", ha="center", va="bottom", fontsize=9.5)
ax.set_xticks(x)
ax.set_xticklabels(ORDER)
ax.set_ylim(0.80, 1.0)
ax.set_ylabel("Empirical 90% coverage")
ax.set_title("Uncertainty calibration: coverage close to nominal 0.90",
             fontsize=12.5, pad=10)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("figures/calibration.pdf", bbox_inches="tight")
print("saved figures/calibration.pdf")
plt.close(fig)

# ================================================================
# 想要中文标签? 在文件顶部 import 后加:
#   import matplotlib
#   matplotlib.rcParams["font.family"] = "Noto Sans CJK SC"   # 或 SimHei / WenQuanYi
#   matplotlib.rcParams["axes.unicode_minus"] = False
# 然后把上面的英文 set_ylabel / set_title / label 换成中文即可。
# (前提是系统装了对应中文字体,否则会显示方框)
# ================================================================