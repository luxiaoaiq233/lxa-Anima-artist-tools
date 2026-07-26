"""AAT Step Alternator (Guider)：序数步画师轮换的自定义 guider。

实现要点（接口依据见源码行号注释）：
- 继承 comfy.samplers.CFGGuider，全链路复用其 CFG 行为（GUIDER 槽位取代内置
  CFGGuider 时采样不退化为 cfg=1）；
- set_conds 时把 base / negative / 各画师正条件一起塞进 inner_set_conds——
  outer_sample 的 prepare_sampling（samplers.py:1233）与 inner_sample 的
  process_conds（samplers.py:1218）对 dict 全部 key 做预处理，每份画师条件
  零额外编码地获得与正常流程一致的预处理结果；
- predict_noise（samplers.py:1211，每步调用）前按序数步索引把
  self.conds["positive"] 指到本步激活的条件键，再交回父类；
- 序数索引 = argmin(|sigmas_table − 当前 σ|)，对多评估采样器稳健；
- 无任何 model patch、无模块级全局状态（全部状态挂在 guider 实例上，
  采样结束随实例 GC）。
"""

import logging
import math

import torch

import comfy.samplers

logger = logging.getLogger("lxa_aat")

MODE_ALTERNATE_EVERY = "alternate_every"
MODE_ALTERNATE_N = "alternate_n"
MODE_CUSTOM_RANGES = "custom_ranges"
MODES = (MODE_ALTERNATE_EVERY, MODE_ALTERNATE_N, MODE_CUSTOM_RANGES)

FALLBACK_BASE = "base"
FALLBACK_LAST = "last"

BASE_KEY = "positive"
NEG_KEY = "negative"


def slice_conditioning(cond, token_length):
    """按 token_lengths 契约把 Pack 的零 pad conditioning 切回真长。

    防止零向量 token 进入 preprocess_text_embeds 后吸收注意力质量。
    cond: [[tensor, meta]]；token_length: {"qwen": int, "t5": int}。
    """
    tensor, meta = cond[0]
    q = int(token_length.get("qwen") or tensor.shape[1])
    t5 = int(token_length.get("t5") or 0)
    new_meta = dict(meta)
    if t5 and new_meta.get("t5xxl_ids") is not None:
        new_meta["t5xxl_ids"] = new_meta["t5xxl_ids"][:t5]
        if new_meta.get("t5xxl_weights") is not None:
            new_meta["t5xxl_weights"] = new_meta["t5xxl_weights"][:t5]
    return [[tensor[:, :q], new_meta]]


class StepAlternatorGuider(comfy.samplers.CFGGuider):
    """序数步画师轮换 guider：每步把 positive 指到本步激活的画师条件。"""

    def __init__(self, model_patcher, config):
        super().__init__(model_patcher)
        self._cfg_route = config          # dict，见 AATStepAlternator.build
        self._sigmas = None               # 采样完整 σ 表（sample 时捕获）
        self._last_artist_key = None      # fallback=last / final_artist=-1 用

    # ------------------------------------------------------------- 条件装载

    def set_conds(self, positive, negative, artist_conds):
        conds = {BASE_KEY: positive, NEG_KEY: negative}
        for i, c in enumerate(artist_conds):
            conds[self._artist_key(i)] = c
        self.inner_set_conds(conds)

    @staticmethod
    def _artist_key(i):
        return f"aat_artist_{i}"

    # ------------------------------------------------------------- 步索引

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None,
               callback=None, disable_pbar=False, seed=None):
        try:
            self._sigmas = [float(s) for s in sigmas.detach().float().cpu().reshape(-1)]
        except Exception:
            self._sigmas = None
        return super().sample(noise, latent_image, sampler, sigmas,
                              denoise_mask=denoise_mask, callback=callback,
                              disable_pbar=disable_pbar, seed=seed)

    def _step_index(self, timestep):
        if self._sigmas is None or len(self._sigmas) < 2:
            return None
        try:
            t = float(timestep.detach().float().cpu().reshape(-1)[0])
        except Exception:
            return None
        return min(range(len(self._sigmas)),
                   key=lambda i: abs(self._sigmas[i] - t))

    # ------------------------------------------------------------- 轮换决策

    def _route(self, idx, total_steps):
        """返回本步激活的条件键（BASE_KEY 或 _artist_key(i)）。"""
        cfg = self._cfg_route
        n_artists = cfg["n_artists"]
        if n_artists == 0:
            return BASE_KEY
        if idx is None:
            return self._artist_key(0)  # 拿不到步索引：恒定第一位画师（保守）

        # 最后 K 步固定画师收尾（K=0 关闭；final_artist=-1 = 上一位激活）
        k = int(cfg["final_k"])
        if k > 0 and idx >= total_steps - k:
            fa = int(cfg["final_artist"])
            if 0 <= fa < n_artists:
                return self._artist_key(fa)
            if self._last_artist_key is not None:
                return self._last_artist_key
            # 无上一位（K 覆盖全程）→ 第一位画师
            return self._artist_key(0)

        mode = cfg["mode"]
        if mode == MODE_ALTERNATE_EVERY:
            return self._artist_key(idx % n_artists)
        if mode == MODE_ALTERNATE_N:
            n = max(1, int(cfg["n_every"]))
            return self._artist_key((idx // n) % n_artists)

        # custom_ranges：后声明优先（取最后一个覆盖者）；end<0 = 到末步
        hit = None
        for artist, start, end in cfg["ranges"]:
            if start <= idx and (end < 0 or idx <= end):
                hit = artist
        if hit is not None:
            return self._artist_key(hit)
        # 区间未覆盖：base（默认）或 last（上一位）
        if cfg["fallback"] == FALLBACK_LAST and self._last_artist_key is not None:
            return self._last_artist_key
        return BASE_KEY

    # ------------------------------------------------------------- 每步换画师

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        total_steps = max(1, len(self._sigmas) - 1) if self._sigmas else 1
        idx = self._step_index(timestep)
        key = self._route(idx if idx is not None else 0, total_steps)
        if key != BASE_KEY and key not in self.conds:
            logger.warning("[lxa_aat] Alternator: 条件键 %s 缺失，回退 base", key)
            key = BASE_KEY
        self.conds[BASE_KEY] = self.conds[key]
        if key != BASE_KEY:
            self._last_artist_key = key
        if self._cfg_route["debug_logging"]:
            step_txt = "?" if idx is None else f"{idx}/{total_steps}"
            logger.info("[lxa_aat] Alternator 步 %s → %s", step_txt, key)
        return super().predict_noise(x, timestep, model_options, seed)


# ---------------------------------------------------------------------------
# Epsilon Multi-Guide
# ---------------------------------------------------------------------------

CFG_ORDER_STACK_THEN_CFG = "stack_then_cfg"
CFG_ORDER_CFG_THEN_STACK = "cfg_then_stack"
CFG_ORDERS = (CFG_ORDER_STACK_THEN_CFG, CFG_ORDER_CFG_THEN_STACK)


class EpsilonMultiGuideGuider(comfy.samplers.CFGGuider):
    """每步在 cond batch [uncond, base, base+@画师...] 一次前向后按公式合成 ε。

    公式: ε = ε_base + Σ sᵢ·(εᵢ − ε_base)（模型内部零接触）。
    与 CFG 的结合次序（线性 CFG 下代数恒等，推导见 README）：
      (a) stack_then_cfg（默认）: eps_c = ε_base + Σ sᵢΔᵢ，再 u + cfg·(eps_c − u)
      (b) cfg_then_stack: 各条件先各自 CFG，再 εb_cfg + Σ sᵢ(εᵢ_cfg − εb_cfg)
    全部系数为 0（或无画师）→ 走与 sampling_function 逐操作一致的标准 CFG
    路径（samplers.py:608-626 同构，含 cfg≈1.0 跳过 uncond 的优化），
    保证 s=0 时输出与纯 base 逐位一致。
    """

    def __init__(self, model_patcher, strengths, cfg_order):
        super().__init__(model_patcher)
        self._strengths = [float(s) for s in strengths]
        self._cfg_order = cfg_order

    def set_conds(self, positive, negative, artist_conds):
        conds = {"positive": positive, "negative": negative}
        for i, c in enumerate(artist_conds):
            conds[f"aat_artist_{i}"] = c
        self.inner_set_conds(conds)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        cfg = float(self.cfg)
        uncond = self.conds.get("negative")
        base = self.conds.get("positive")
        active = [(i, si) for i, si in enumerate(self._strengths)
                  if abs(si) > 1e-12 and f"aat_artist_{i}" in self.conds]

        if not active:
            # 标准 CFG 路径（与 samplers.py:608-626 逐操作一致，s=0 探针保证）
            if math.isclose(cfg, 1.0) and not model_options.get("disable_cfg1_optimization", False):
                uncond = None
            out = comfy.samplers.calc_cond_batch(
                self.inner_model, [base, uncond], x, timestep, model_options)
            return out[1] + (out[0] - out[1]) * cfg

        conds = [uncond, base] + [self.conds[f"aat_artist_{i}"] for i, _ in active]
        out = comfy.samplers.calc_cond_batch(self.inner_model, conds, x, timestep, model_options)
        u, eps_base = out[0], out[1]

        if self._cfg_order == CFG_ORDER_CFG_THEN_STACK:
            # (b) 各条件先各自 CFG，再按 Σ sᵢΔᵢ 堆叠
            eb = u + (eps_base - u) * cfg
            eps = eb
            for k, (_i, si) in enumerate(active):
                eps_i = u + (out[2 + k] - u) * cfg
                eps = eps + si * (eps_i - eb)
            return eps

        # (a) 默认：先 delta 堆叠，再做 CFG
        eps_c = eps_base
        for k, (_i, si) in enumerate(active):
            eps_c = eps_c + si * (out[2 + k] - eps_base)
        return u + (eps_c - u) * cfg
