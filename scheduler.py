"""层-步调度：层范围/步区间解析、sigma 进度换算、过渡函数。

sigma 约定（M0 核查）：Anima 为 FLOW 采样（shift=3.0, multiplier=1.0，
supported_models.py:1104-1107 + model_sampling.py:315-319），sigma ∈ [0,1]
且随采样步递减，传入模型的 timestep == sigma。进度换算直接读
transformer_options["sigmas"]，有采样表（sample_sigmas）时按表端点精确换算。
"""

import logging
import math

logger = logging.getLogger("lxa_aat")

TRANSITION_FNS = ("hard", "linear", "cosine")


# ---------------------------------------------------------------------------
# layer_config 解析
# ---------------------------------------------------------------------------

def _parse_range(s, lo_lim, hi_lim, integer=False):
    """'0-8' / '5' / '0.2-0.6' → (lo, hi)；自动交换倒置并夹取到 [lo_lim, hi_lim]。"""
    if "-" in s:
        a_s, b_s = s.split("-", 1)
    else:
        a_s = b_s = s
    a, b = float(a_s.strip()), float(b_s.strip())
    if a > b:
        a, b = b, a
    a = min(max(a, lo_lim), hi_lim)
    b = min(max(b, lo_lim), hi_lim)
    if integer:
        a, b = int(round(a)), int(round(b))
    return (a, b)


def parse_layer_config(text, n_artists, num_blocks):
    """逐行解析 "artist_idx::weight@layer_range,step=lo-hi"（step 段可省略 = 全程）。

    返回 layer_map: {block_index: [(artist_idx, weight, (lo, hi)), ...]}；
    步区间为归一化进度 [0,1]（0=起始高噪，1=结束）。
    后声明优先覆盖（同 block 同 artist 后者替换前者）；非法行告警并跳过。
    """
    per_block = {}  # block -> {artist_idx: (weight, (lo, hi))}
    for lineno, raw in enumerate((text or "").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            artist_s, rest = line.split("::", 1)
            artist_idx = int(artist_s.strip())
            if not (0 <= artist_idx < n_artists):
                raise ValueError(f"artist_idx {artist_idx} 越界（共 {n_artists} 位画师，0 起）")
            if "@" not in rest:
                raise ValueError("缺少 '@layer_range'")
            weight_s, rest = rest.split("@", 1)
            weight = float(weight_s.strip())

            step_range = (0.0, 1.0)
            if "," in rest:
                range_s, *opts = rest.split(",")
                for opt in opts:
                    opt = opt.strip()
                    if not opt:
                        continue
                    if not opt.lower().startswith("step="):
                        raise ValueError(f"未知选项 {opt!r}（仅支持 step=lo-hi）")
                    step_range = _parse_range(opt[5:].strip(), 0.0, 1.0)
            else:
                range_s = rest
            lo_b, hi_b = _parse_range(range_s.strip(), 0, num_blocks - 1, integer=True)

            for b in range(lo_b, hi_b + 1):
                per_block.setdefault(b, {})[artist_idx] = (weight, step_range)
        except ValueError as e:
            logger.warning("[lxa_aat] 层-步调度 layer_config 第 %d 行跳过（%s）: %r",
                           lineno, e, line)
    return {b: [(a, w, sr) for a, (w, sr) in sorted(d.items())]
            for b, d in per_block.items()}


# ---------------------------------------------------------------------------
# sigma → 进度
# ---------------------------------------------------------------------------

def sigma_progress(current_sigma, sample_sigmas=None):
    """sigma → 归一化进度 [0,1]（0=起始高噪，1=结束）。

    有采样表时按表端点精确换算；无表时按 FLOW 端点 [1, 0] 近似（p = 1 - sigma）。
    """
    s = float(current_sigma)
    if sample_sigmas is not None:
        try:
            s0 = float(sample_sigmas[0])
            s1 = float(sample_sigmas[-1])
            if abs(s0 - s1) > 1e-12:
                return min(max((s0 - s) / (s0 - s1), 0.0), 1.0)
        except Exception:
            pass
    return min(max(1.0 - s, 0.0), 1.0)


# ---------------------------------------------------------------------------
# 过渡函数
# ---------------------------------------------------------------------------

def _ramp(t, fn):
    """t ∈ [0,1] → 过渡系数。cosine（默认）两端导数为 0。"""
    t = min(max(t, 0.0), 1.0)
    if fn == "linear":
        return t
    return 0.5 * (1.0 - math.cos(math.pi * t))  # cosine


def boundary_factor(p, lo, hi, width, fn):
    """进度 p 下单条步区间 [lo,hi] 的有效系数。

    过渡区在区间内侧（收缩式，画师绝不溢出声明区间）；触及 0/1 端点的
    边界完全放开（不做渐入渐出）；hard 或 width=0 时无过渡。
    """
    if hi < lo:
        lo, hi = hi, lo
    if fn == "hard" or width <= 1e-9:
        return 1.0 if lo <= p <= hi else 0.0
    f = 1.0
    if lo > 0.0:
        if p < lo:
            return 0.0
        f = min(f, _ramp((p - lo) / width, fn))
    if hi < 1.0:
        if p > hi:
            return 0.0
        f = min(f, _ramp((hi - p) / width, fn))
    return f
