"""lxa AAT 核心逻辑：wrapper、递归解包、融合、CFG mask。

并行混合（ParallelArtistCrossAttn）设计要点：
- 整模块包装 cross_attn：不触碰投影层内部（对 to_q / q_proj 等命名差异免疫），
  以「同一 q 输入、不同画师 conditioning 作 context」多次调用被包模块 forward；
- base 分支始终计算、始终保留（外部插件 wrapper 也因此始终生效）；
- 画师预处理严格复刻 comfy/model_base.py:1461-1480（Anima.extra_conds 推理分支），
  缓存挂在经 wrapper 实例可达的 ArtistState 上，随 clone GC；
- 数值敏感处（加权求和、EMA、投影分母）一律 fp32 计算再截回。

lowrank_delta 的 SVD 子空间投影移植自 Anima-Artist-Mixer（_fwd_lowrank_avg，
custom_nodes/Anima-Artist-Mixer/nodes.py:1000-1067），MIT License
（© 2026 An1X3R & 汐浮尘），完整许可文本见本包 NOTICE 文件。
"""

import logging

import torch
from torch import nn

import logging

import torch
from torch import nn

from .scheduler import boundary_factor, sigma_progress

logger = logging.getLogger("lxa_aat")

# transformer_options 下唯一命名空间键（架构约束 #4）
NS_KEY = "lxa_aat"

# interpolate = delta 堆叠（v0.1 新语义）；interpolate_legacy = 旧全量式 lerp（保留兼容）
# lowrank_delta = SVD top-k 子空间投影的 delta 堆叠（移植自 Anima-Artist-Mixer，见文件头声明）
FUSION_MODES = ("output_avg", "interpolate", "base_preserve", "interpolate_legacy", "lowrank_delta")


# ---------------------------------------------------------------------------
# 解包（架构约束 #3）
# ---------------------------------------------------------------------------

def recursive_unwrap(module, seen=None):
    """追踪 .original 链到最底层模块；seen 集合防循环引用。

    仅用于结构检查（验证底层模块形态），不用于替换——替换目标永远是
    get_model_object 返回的当前链顶模块，保证外部插件 wrapper 不被丢弃。
    """
    if seen is None:
        seen = set()
    cur = module
    while cur is not None and id(cur) not in seen and hasattr(cur, "original"):
        seen.add(id(cur))
        nxt = getattr(cur, "original")
        if nxt is None or not isinstance(nxt, nn.Module):
            break
        cur = nxt
    return cur


def unwrap_own(module, wrapper_cls, seen=None):
    """只解开本套件自己的 wrapper 类（同一节点重复应用时替换而非堆叠）。

    与 Anima-Artist-Mixer 的 _unwrap_cross_attn 同一约定：isinstance 限定
    自己的类，外部插件的 wrapper 一律保留在链中。
    """
    if seen is None:
        seen = set()
    cur = module
    while isinstance(cur, wrapper_cls) and id(cur) not in seen:
        seen.add(id(cur))
        cur = cur.original
    return cur


def _delegated_attr(self, name):
    """__getattr__ 共用实现：沿 .original 链查找自身没有的属性。

    使动态 VRAM 加载器的干净形态权重键在 wrapper 安装期间也能正确
    resolve/setattr；nn.Module.__getattr__ 仅在常规查找失败时触发。
    seen 集合防循环引用。
    """
    try:
        return super(self.__class__, self).__getattr__(name)
    except AttributeError:
        modules = self.__dict__.get("_modules")
        cur = modules.get("original") if modules else None
        seen = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            try:
                return getattr(cur, name)
            except AttributeError:
                nxt = getattr(cur, "original", None)
                cur = nxt if isinstance(nxt, nn.Module) else None
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


def build_cond_mask(t_opts, batch, device, apply_to_uncond, warn_once=None):
    """True = 注入画师语义的行。cond_or_uncond 缺失时全部按 cond 处理（安全回退）。"""
    if apply_to_uncond:
        return torch.ones(batch, dtype=torch.bool, device=device)
    cou = t_opts.get("cond_or_uncond")
    if cou is None:
        return torch.ones(batch, dtype=torch.bool, device=device)
    if torch.is_tensor(cou):
        cou = cou.tolist()
    cou = list(cou)
    if len(cou) == 0 or batch % len(cou) != 0:
        if warn_once is not None:
            warn_once(
                "cou",
                "[lxa_aat] cond_or_uncond 长度 %d 与 batch %d 不整除，"
                "全部按 cond 处理", len(cou), batch,
            )
        return torch.ones(batch, dtype=torch.bool, device=device)
    bpc = batch // len(cou)
    return torch.tensor([cou[r // bpc] == 0 for r in range(batch)],
                        dtype=torch.bool, device=device)


def _rms_match_rows(out, ref, eps=1e-8):
    """逐 batch 行把 out 的整体 RMS 缩放到 ref 的 RMS（fp32，分母 clamp）。

    用于 interpolate（delta 堆叠）后做幅度匹配：强度由 s 与原始权重控制
    "方向与刻度"，RMS 匹配负责把每层输出幅度钉回 base_out 的水平，
    避免非归一化权重导致的幅度漂移/过曝。
    """
    dims = tuple(range(1, ref.ndim))
    rms_ref = ref.pow(2).mean(dim=dims, keepdim=True).clamp_min(eps).sqrt()
    rms_out = out.pow(2).mean(dim=dims, keepdim=True).clamp_min(eps).sqrt()
    return out * (rms_ref / rms_out)


def _rms_clamp_rows(out, ref, ratio, eps=1e-8):
    """逐 batch 行 RMS 钳制：rms(out) 超出 ratio × rms(ref) 时按比例钳回（fp32）。

    与 interpolate 的精确 RMS 匹配同属一套幅度保护族；ratio ≥ 1 为合理区间
    （ratio < 1 会连 s=0 的 base 也被缩——仅作硬限使用时避免）。
    """
    dims = tuple(range(1, ref.ndim))
    rms_ref = ref.pow(2).mean(dim=dims, keepdim=True).clamp_min(eps).sqrt()
    rms_out = out.pow(2).mean(dim=dims, keepdim=True).clamp_min(eps).sqrt()
    scale = torch.clamp(ratio * rms_ref / rms_out, max=1.0)
    return out * scale


def _svd_topk_project(d_mat, k):
    """把 delta 矩阵 D (N, M) 投影到 top-k 右奇异子空间（去噪、只保留主风格方向）。

    逐行核对自 Anima-Artist-Mixer _fwd_lowrank_avg @1036-1052：
    torch.svd_lowrank(D, q=k, niter=2) → U(N,k) S(k) V(M,k)，D_lowrank = U·diag(S)·Vᵀ；
    k >= n 时不投影（与 output_avg 直和数学等价，AAM 原注 @1051-1052）。全程 fp32。
    """
    n = d_mat.shape[0]
    if k >= n:
        return d_mat
    u, s, v = torch.svd_lowrank(d_mat, q=k, niter=2)
    return u @ torch.diag(s) @ v.transpose(-1, -2)


# ---------------------------------------------------------------------------
# 权重备份键卫生（动态 VRAM 加载器兼容）
# ---------------------------------------------------------------------------
#
# 背景：ModelPatcherDynamic.load()（model_patcher.py:1896-1904）会按当前模块
# 路径把 CPU 原始权重写入（跨 clone 共享的）backup dict。wrapper 安装期间，
# 键形如 '….cross_attn.original.q_proj.weight'。之后未装 wrapper 的干净 run
# 在 restore_loaded_backups（model_patcher.py:1768-1776）按该键 resolve_attr
# 时会因 raw 模块没有 .original 而 AttributeError 崩溃。
# 修复为双向：
#   a) ParallelArtistCrossAttn.__getattr__ 委托 → 干净键在 wrapper 在场时也可达；
#   b) sanitize_weight_backups 把本套件产生的 '.original.' 脏键重映射回干净形态
#      → 干净模型上直接落在 raw 模块上。
# 触发点：节点 patch 时、ON_PRE_RUN、ON_CLEANUP（samplers.py:1251/1259，finally 中）。

def sanitize_weight_backups(patcher, prefixes):
    """把 patcher.backup / backup_buffers 中含本套件 wrapper 段的键重映射为干净形态。

    prefixes: 本套件 add_object_patch 使用过的完整前缀集合
    （如 {'diffusion_model.blocks.3.cross_attn', ...}）。
    只处理这些前缀，不触碰第三方插件的键。返回重映射条数。
    """
    remapped = 0
    for store_name in ("backup", "backup_buffers"):
        store = getattr(patcher, store_name, None)
        if not store:
            continue
        for key in list(store.keys()):
            if ".original." not in key:
                continue
            for prefix in prefixes:
                seg = prefix + ".original."
                if seg in key:
                    clean_key = key.replace(seg, prefix + ".")
                    if clean_key in store:
                        store.pop(key)  # 同一参数的重复备份，保留已有干净键
                    else:
                        store[clean_key] = store.pop(key)
                    remapped += 1
                    break
    if remapped:
        logger.info("[lxa_aat] 已重映射 %d 条含 wrapper 段的权重备份键", remapped)
    return remapped


def make_detach_strip_callback(patches):
    """生成 ON_DETACH 回调：本 clone 被 detach 时，从共享模型上剥离它注册的 wrapper。

    背景：load_models_gpu（model_management.py:902-910）加载新 patcher 时对同模型
    clone 调 detach(unpatch_all=False)，该路径不调用 unpatch_model——本套件的
    wrapper 会残留在共享模块上并被后续干净 patcher 的运行继承（实测干净 patch
    的出图与上一次节点出图逐位一致）。本回调在 detach 时按【身份一致】剥离：
    只移除仍挂在原位的本 clone wrapper（同名位置若已是他人 wrapper 则不动），
    后续无论干净运行还是新 clone 重新 patch，都从干净状态开始。
    """
    def _strip(patcher, unpatch_all):
        import comfy.utils
        model = patcher.model
        stripped = 0
        for name, wrapper in patches.items():
            try:
                cur = comfy.utils.get_attr(model, name)
            except Exception:
                continue
            if cur is wrapper:
                comfy.utils.set_attr(model, name, wrapper.original)
                stripped += 1
        if stripped:
            logger.info("[lxa_aat] detach 剥离 %d 个残留 wrapper（防跨运行泄漏）", stripped)
    return _strip


# ---------------------------------------------------------------------------
# 画师状态（原始 cond + 预处理缓存）
# ---------------------------------------------------------------------------

class ArtistState:
    """一次节点执行创建的共享画师状态，被各层 wrapper 引用。

    预处理结果按 (device, dtype) 缓存：同一采样中每位画师只算一次；
    模型换设备/精度时自动重算。采样结束随 wrapper 一起被 GC。
    """

    def __init__(self, diffusion_model, labels, weights, conditionings, token_lengths):
        self.diffusion_model = diffusion_model
        self.labels = list(labels)
        self.weights = [float(w) for w in weights]
        self.conditionings = conditionings
        self.token_lengths = token_lengths
        self.n_artists = len(self.labels)
        self._cache_key = None
        self._cache_ctxs = None

    def artist_contexts(self, ref_device, ref_dtype):
        key = (str(ref_device), ref_dtype)
        if self._cache_key == key and self._cache_ctxs is not None:
            return self._cache_ctxs
        ctxs = [
            self._preprocess_one(i, ref_device, ref_dtype)
            for i in range(self.n_artists)
        ]
        self._cache_key = key
        self._cache_ctxs = ctxs
        return ctxs

    def _preprocess_one(self, i, device, dtype):
        """先按 token_lengths[i] 把 cond 张量 / t5xxl_ids / t5xxl_weights 截回真实长度
        （qwen 维与 t5 维分别切），再严格复刻 comfy/model_base.py:1467-1474。
        截断后的数据与单画师天然编码路径逐位等价（pad 的是尾部零向量）。
        """
        tensor, meta = self.conditionings[i][0]
        tl = self.token_lengths[i] or {}
        q_len = tl.get("qwen") or tensor.shape[1]
        cond = tensor[:, :q_len].to(device=device, dtype=dtype)

        ids = meta.get("t5xxl_ids")
        if ids is None:
            return cond
        if not hasattr(self.diffusion_model, "preprocess_text_embeds"):
            logger.warning(
                "[lxa_aat] 画师 %s: 目标模型无 preprocess_text_embeds，"
                "t5xxl_ids 被忽略，直接使用原始 cond", self.labels[i],
            )
            return cond

        t5_len = tl.get("t5") or ids.shape[0]
        ids = ids[:t5_len]
        weights = meta.get("t5xxl_weights")
        weights = weights[:t5_len] if weights is not None else None

        # ↓ 与 comfy/model_base.py:1470-1474 逐行对应
        if weights is not None:
            weights = weights.unsqueeze(0).unsqueeze(-1).to(cond)
        ids = ids.unsqueeze(0)
        return self.diffusion_model.preprocess_text_embeds(
            cond, ids.to(device=device),
            t5xxl_weights=weights.to(device=device, dtype=dtype) if weights is not None else None,
        )


# ---------------------------------------------------------------------------
# 并行混合 wrapper
# ---------------------------------------------------------------------------

class ParallelArtistCrossAttn(nn.Module):
    """每位画师独立编码、独立 K/V 前向，输出在特征空间加权融合。"""

    def __init__(self, original, state, layer_idx, fusion_mode="output_avg",
                 strength=1.0, block_range=(0, 2 ** 31 - 1), apply_to_uncond=False,
                 ema_alpha=0.0, static_capture_k=0,
                 lowrank_k=1, lowrank_normalize=True, rms_clamp_ratio=1.0):
        super().__init__()
        self.original = original
        self._st = state
        self._idx = layer_idx
        self._fusion_mode = fusion_mode
        self._strength = float(strength)
        self._range = block_range
        self._apply_to_uncond = bool(apply_to_uncond)
        self._ema_alpha = float(ema_alpha)
        self._static_k = int(static_capture_k)
        self._lowrank_k = int(lowrank_k)
        self._lowrank_normalize = bool(lowrank_normalize)
        self._rms_clamp_ratio = float(rms_clamp_ratio)
        self._disabled = False
        self._warned = set()
        # 稳定器状态（全部 fp32 张量/标量，挂实例上，采样结束随 wrapper GC）
        self._last_sigma = None
        self._ema = None
        self._sc_seen = []
        self._sc_sum = None
        self._sc_count = 0
        self._sc_frozen = None

    def __getattr__(self, name):
        """把自身没有的属性（q_proj/k_proj/v_proj/output_proj 等）沿 .original 链委托。

        使动态 VRAM 加载器的干净形态权重键（'….cross_attn.q_proj.weight'）
        在 wrapper 安装期间也能正确 resolve/setattr。
        """
        return _delegated_attr(self, name)

    # ------------------------------------------------------------- 工具

    def _warn_once(self, key, msg, *args):
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg, *args)

    @staticmethod
    def _current_sigma(t_opts):
        sig = t_opts.get("sigmas")
        if sig is None:
            return None
        try:
            return float(sig.reshape(-1)[0])
        except Exception:
            return None

    # ------------------------------------------------------------- forward

    def forward(self, x, context=None, rope_emb=None, transformer_options=None, **kwargs):
        if (self._disabled
                or not (self._range[0] <= self._idx <= self._range[1])
                or self._st.n_artists == 0):
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options, **kwargs)
        try:
            return self._forward_isolated(x, context, rope_emb,
                                          transformer_options or {}, kwargs)
        except Exception:
            # 异常隔离：置永久短路标记，后续直接走 original
            self._disabled = True
            logger.exception(
                "[lxa_aat] 并行混合 第 %d 层 forward 异常，后续永久短路到 original",
                self._idx,
            )
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options, **kwargs)

    def _forward_isolated(self, x, context, rope_emb, t_opts, extra_kwargs):
        # base 分支：始终计算、始终保留
        base_out = self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=t_opts, **extra_kwargs)

        t_mask = self._target_mask(t_opts, x.shape[0], x.device)
        if not bool(t_mask.any()):
            return base_out
        active = self._active_entries(t_opts)
        if not active:
            return base_out

        x_t = x[t_mask]
        if self._fusion_mode == "lowrank_delta":
            fused = self._fuse_lowrank(x_t, base_out[t_mask], active,
                                       rope_emb, t_opts, extra_kwargs).to(base_out.dtype)
        else:
            artist_total, w_sum = self._artist_total(x_t, active, rope_emb, t_opts, extra_kwargs)  # fp32
            artist_total = self._stabilize(artist_total, t_opts)
            fused = self._fuse(base_out[t_mask].float(), artist_total, w_sum).to(base_out.dtype)
        out = base_out.clone()
        out[t_mask] = fused
        return out

    # ------------------------------------------------------------- CFG mask

    def _target_mask(self, t_opts, batch, device):
        """True = 应用画师注入的行。cond_or_uncond 缺失时全部按 cond 处理（安全回退）。"""
        return build_cond_mask(t_opts, batch, device, self._apply_to_uncond, self._warn_once)

    def _active_entries(self, t_opts):
        """[(artist_idx, factor)]：层-步调度 协同时读 active_set[self._idx]，缺省=全部 factor 1.0。

        协同语义：entry 存在但为空 → 本层全部画师关闭；entry 不存在 → 并行混合 默认全开。
        factor 与 Pack 权重相乘（层-步调度 作为门控/调制）。
        """
        ns = t_opts.get(NS_KEY)
        entry = None
        if isinstance(ns, dict):
            aset = ns.get("active_set")
            if isinstance(aset, dict):
                entry = aset.get(self._idx)
        if entry is None:
            return [(i, 1.0) for i in range(self._st.n_artists)]
        try:
            idxs, factors = entry
            return [(int(i), float(f)) for i, f in zip(idxs, factors)
                    if 0 <= int(i) < self._st.n_artists and float(f) > 0.0]
        except Exception:
            self._warn_once("aset", "[lxa_aat] 并行混合 第 %d 层: active_set 条目畸形，回退全开", self._idx)
            return [(i, 1.0) for i in range(self._st.n_artists)]

    # ------------------------------------------------------------- 分支计算

    def _weights(self, entries, device):
        """output_avg / base_preserve 用归一化权重；interpolate / interpolate_legacy /
        lowrank_delta 用原始权重（lowrank_delta 的 Σ|w| 归一化在 _fuse_lowrank 内）。
        entries 为 (artist_idx, factor)，factor 与 Pack 权重相乘。"""
        w = torch.tensor([self._st.weights[i] * f for i, f in entries],
                         dtype=torch.float32, device=device)
        if self._fusion_mode in ("interpolate", "interpolate_legacy", "lowrank_delta"):
            return w
        s = float(w.sum())
        if abs(s) < 1e-8:
            return torch.full_like(w, 1.0 / max(len(entries), 1))
        return w / s

    def _artist_outputs(self, x_t, entries, rope_emb, t_opts, extra_kwargs):
        """返回 (stacked_outputs (N, B, S, D) fp32, w)：每位画师一次独立前向。

        分支全部激活且序列长度一致 → batch 维拼接一次 attention 调用；
        长度不一致 → 串行并打印一次警告。与 _artist_total 同一收集策略。
        """
        ctxs_all = self._st.artist_contexts(x_t.device, x_t.dtype)
        ctxs = [ctxs_all[i] for i, _ in entries]
        w = self._weights(entries, x_t.device)
        b_t = x_t.shape[0]

        same_len = all(c.shape[1:] == ctxs[0].shape[1:] for c in ctxs)
        all_active = len(entries) == self._st.n_artists

        if all_active and same_len and all(c.shape[0] == 1 for c in ctxs):
            n = len(ctxs)
            ctx_cat = torch.cat(ctxs, dim=0)  # (N, L, D)
            x_rep = x_t.unsqueeze(0).expand(n, b_t, -1, -1).reshape(n * b_t, *x_t.shape[1:])
            ctx_rep = ctx_cat.unsqueeze(1).expand(n, b_t, -1, -1).reshape(n * b_t, *ctx_cat.shape[1:])
            out = self.original(x_rep, ctx_rep, rope_emb=rope_emb,
                                transformer_options=t_opts, **extra_kwargs)
            return out.reshape(n, b_t, *out.shape[1:]).float(), w

        if not same_len:
            self._warn_once(
                "serial",
                "[lxa_aat] 并行混合 第 %d 层: 画师 context 长度不一致，"
                "回退串行 forward", self._idx,
            )
        outs = []
        for k, (i, _f) in enumerate(entries):
            ctx_i = ctxs[k]
            if ctx_i.shape[0] == 1 and b_t != 1:
                ctx_i = ctx_i.expand(b_t, -1, -1)
            outs.append(self.original(x_t, ctx_i, rope_emb=rope_emb,
                                      transformer_options=t_opts, **extra_kwargs).float())
        return torch.stack(outs, dim=0), w

    def _artist_total(self, x_t, entries, rope_emb, t_opts, extra_kwargs):
        """返回 (artist_total, w_sum)：artist_total = Σ wᵢ·outᵢ（fp32），w_sum = Σwᵢ。"""
        outs, w = self._artist_outputs(x_t, entries, rope_emb, t_opts, extra_kwargs)
        return (w.view(-1, *([1] * (outs.ndim - 1))) * outs).sum(dim=0), float(w.sum())

    # ------------------------------------------------------------- 稳定器

    def _step_index(self, sig):
        if sig is None:
            return self._sc_count
        for idx, s in enumerate(self._sc_seen):
            if abs(s - sig) < 1e-6:
                return idx
        self._sc_seen.append(sig)
        return len(self._sc_seen) - 1

    def _stabilize(self, artist_total, t_opts):
        if self._ema_alpha <= 0.0 and self._static_k <= 0:
            return artist_total
        sig = self._current_sigma(t_opts)
        if sig is not None:
            if self._last_sigma is None or sig > self._last_sigma + 1e-3:
                # sigma 回升 → 新一次采样开始 → 清空跨步缓存
                self._ema = None
                self._sc_seen = []
                self._sc_sum = None
                self._sc_count = 0
                self._sc_frozen = None
            self._last_sigma = sig

        # static_capture_k：前 K 步平均后冻结复用
        if self._static_k > 0:
            if self._sc_frozen is not None:
                artist_total = self._sc_frozen
            else:
                step_idx = self._step_index(sig)
                if step_idx < self._static_k:
                    self._sc_sum = artist_total if self._sc_sum is None else self._sc_sum + artist_total
                    self._sc_count += 1
                elif self._sc_count > 0:
                    self._sc_frozen = self._sc_sum / self._sc_count
                    artist_total = self._sc_frozen

        # 跨步 EMA（fp32 累加）
        if self._ema_alpha > 0.0:
            a = self._ema_alpha
            self._ema = artist_total if self._ema is None else a * self._ema + (1.0 - a) * artist_total
            artist_total = self._ema
        return artist_total

    # ------------------------------------------------------------- 融合

    def _fuse_lowrank(self, x_t, base_t_raw, entries, rope_emb, t_opts, extra_kwargs):
        """lowrank_delta: out = base + s · Σ wᵢ_norm · SVD_topk(outᵢ − base)。

        逐行核对自 AAM _fwd_lowrank_avg @1000-1059（MIT，见文件头声明）：
        delta_i = A_i − A_base（@1029-1031）→ D (N, M)（@1033-1034）→
        SVD top-k 投影（@1036-1052）→ Σ wᵢ·D_lowrank[i]（@1054-1057）→
        artist_total = A_base + delta_avg（@1059）。全程 fp32。
        行为差异（有意，已记录 README）：
        - AAM 的 fusion=interpolate + strength=s 恰为 base·(1−s) + (base+Δ)·s
          = base + s·Δ，与本式恒等 —— 本实现直接走该式；
        - 叠加后加一道 RMS 钳制（_rms_clamp_rows，ratio 可调、0 关闭），
          AAM 无此保护（其仅对 normalize_weights=False 场景告警 @1816-1833）；
        - AAM 的 static/EMA 缓存层不在本移植范围（本节点稳定器作用于 delta_avg）。
        """
        outs, w_raw = self._artist_outputs(x_t, entries, rope_emb, t_opts, extra_kwargs)  # (N,B,S,D) fp32
        base_t = base_t_raw.float()
        n = outs.shape[0]
        # 权重归一化（默认开，AAM _normalize_weights @431-435 语义：wᵢ/Σ|wᵢ|）
        if self._lowrank_normalize:
            denom = float(w_raw.abs().sum())
            w = w_raw / denom if denom > 1e-8 else torch.full_like(w_raw, 1.0 / n)
        else:
            w = w_raw
        delta = outs - base_t.unsqueeze(0)          # (N, B, S, D)
        d_mat = delta.reshape(n, -1)                # (N, M) fp32
        k = max(1, min(int(self._lowrank_k), n))
        d_low = _svd_topk_project(d_mat, k)         # (N, M) fp32
        delta_avg = (w.view(n, 1) * d_low).sum(dim=0).reshape(base_t.shape)
        delta_avg = self._stabilize(delta_avg, t_opts)
        out = base_t + self._strength * delta_avg
        if self._rms_clamp_ratio > 0:
            out = _rms_clamp_rows(out, base_t, self._rms_clamp_ratio)
        return out

    def _fuse(self, base_t, artist_total, w_sum):
        """base_t / artist_total 均为 fp32；返回 fp32。w_sum = Σwᵢ（原始权重和）。"""
        s = self._strength
        mode = self._fusion_mode
        if mode == "base_preserve":
            # 逐 token 投影剔除平行于 base_out 的分量（分母 clamp，fp32）
            delta = artist_total - base_t
            denom = (base_t * base_t).sum(dim=-1, keepdim=True).clamp(min=1e-8)
            coef = (delta * base_t).sum(dim=-1, keepdim=True) / denom
            return base_t + s * (delta - coef * base_t)
        if mode == "interpolate":
            # delta 堆叠：out = base + s · Σ wᵢ_raw · (outᵢ − base)
            # 分支增量先减 base（消除 base 重复计数），再做 RMS 幅度匹配
            delta = artist_total - w_sum * base_t
            return _rms_match_rows(base_t + s * delta, base_t)
        # output_avg / interpolate_legacy（公式相同，权重语义差异见 _weights）
        return base_t * (1.0 - s) + artist_total * s


# ---------------------------------------------------------------------------
# 层-步调度：层×时间路由
# ---------------------------------------------------------------------------

def _first_float(v):
    if v is None:
        return None
    try:
        return float(v.reshape(-1)[0])
    except Exception:
        return None


class RouterState:
    """一次 层-步调度 节点执行创建的共享路由状态，被各层 RoutedBlockWrapper 引用。"""

    def __init__(self, artist_state, layer_map, transition_fn, transition_width, coop_mode):
        self.artist_state = artist_state  # 独立模式用于混合画师 conditioning（惰性预处理）
        self.layer_map = layer_map        # {block_index: [(artist_idx, weight, (lo, hi))]}
        self.transition_fn = transition_fn
        self.transition_width = float(transition_width)
        self.coop_mode = bool(coop_mode)
        # 由 apply_model 级 wrapper 捕获（make_sigma_capture_wrapper）
        self.current_sigma = None
        self.sample_sigmas = None

    def progress(self, t_opts):
        """优先读 transformer_options['sigmas']，回退 apply_model 捕获值；拿不到返回 None。"""
        s = _first_float(t_opts.get("sigmas"))
        if s is None:
            s = self.current_sigma
        if s is None:
            return None
        ss = t_opts.get("sample_sigmas")
        if ss is None:
            ss = self.sample_sigmas
        return sigma_progress(s, ss)

    def active_entries(self, block_idx, progress):
        """[(artist_idx, eff_weight)]，eff_weight = 声明权重 × 过渡系数；progress=None 时全额。"""
        declared = self.layer_map.get(block_idx, [])
        if progress is None:
            return [(a, w) for a, w, _ in declared]
        out = []
        for a, w, (lo, hi) in declared:
            f = boundary_factor(progress, lo, hi, self.transition_width, self.transition_fn)
            if f > 0.0:
                out.append((a, w * f))
        return out

    def blended_context(self, entries, ref):
        """激活画师 conditioning 按 eff_weight 归一化加权混合（独立模式），返回 (1, L, D)。"""
        if self.artist_state is None:
            return None
        ctxs = self.artist_state.artist_contexts(ref.device, ref.dtype)
        sel = [(ctxs[i], w) for i, w in entries if 0 <= i < len(ctxs)]
        if not sel:
            return None
        if any(c.shape[1:] != sel[0][0].shape[1:] for c, _ in sel):
            return None
        ws = torch.tensor([w for _, w in sel], dtype=torch.float32, device=ref.device)
        s = float(ws.sum())
        ws = ws / s if abs(s) > 1e-8 else torch.full_like(ws, 1.0 / len(sel))
        stack = torch.cat([c.float() for c, _ in sel], dim=0)  # (n, L, D)
        return (ws.view(-1, 1, 1) * stack).sum(dim=0, keepdim=True).to(ref.dtype)


class RoutedBlockWrapper(nn.Module):
    """层-步调度 wrapper：包整个 block 的 forward。

    协同模式（检测到 并行混合 标记）：不替换 context，只把激活集合写入
    transformer_options[NS_KEY]['active_set'][block_index] 后调用 inner；
    独立模式：把 cond 行的 context 替换为激活画师的加权混合 conditioning。
    """

    def __init__(self, original, state, layer_idx, apply_to_uncond=False):
        super().__init__()
        self.original = original
        self._st = state
        self._idx = layer_idx
        self._apply_to_uncond = bool(apply_to_uncond)
        self._disabled = False
        self._warned = set()

    def __getattr__(self, name):
        """自身没有的属性沿 .original 链委托（干净权重键兼容，同 并行混合）。"""
        return _delegated_attr(self, name)

    def _warn_once(self, key, msg, *args):
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg, *args)

    def forward(self, x, emb, context, **kwargs):
        if self._disabled:
            return self.original(x, emb, context, **kwargs)
        try:
            return self._routed(x, emb, context, kwargs)
        except Exception:
            self._disabled = True
            logger.exception(
                "[lxa_aat] 层-步调度 第 %d 层 forward 异常，后续永久短路到 original",
                self._idx,
            )
            return self.original(x, emb, context, **kwargs)

    def _routed(self, x, emb, context, kwargs):
        t_opts = kwargs.get("transformer_options") or {}
        progress = self._st.progress(t_opts)
        if progress is None:
            self._warn_once(
                "progress",
                "[lxa_aat] 层-步调度 第 %d 层: 拿不到当前 sigma，回退为始终激活",
                self._idx,
            )
        entries = self._st.active_entries(self._idx, progress)

        if self._st.coop_mode:
            # 协同：总是写入（含空集合 = 本层全关），不触碰 context；
            # transformer_options 是跨 block 共享对象，必须拷贝后修改，禁止原地写。
            ns = dict(t_opts.get(NS_KEY) or {})
            active_set = dict(ns.get("active_set") or {})
            active_set[self._idx] = ([i for i, _ in entries], [w for _, w in entries])
            ns["active_set"] = active_set
            t2 = dict(t_opts)
            t2[NS_KEY] = ns
            kwargs["transformer_options"] = t2
            return self.original(x, emb, context, **kwargs)

        # 独立模式：未激活 → 透传
        if not entries:
            return self.original(x, emb, context, **kwargs)
        mask = build_cond_mask(t_opts, context.shape[0], context.device,
                               self._apply_to_uncond, self._warn_once)
        if not bool(mask.any()):
            return self.original(x, emb, context, **kwargs)
        blended = self._st.blended_context(entries, context)
        if blended is None or blended.shape[1:] != context.shape[1:]:
            self._warn_once(
                "blend",
                "[lxa_aat] 层-步调度 第 %d 层: 画师 conditioning 形状与 context "
                "不匹配，本层透传", self._idx,
            )
            return self.original(x, emb, context, **kwargs)
        ctx2 = context.clone()
        ctx2[mask] = blended.expand(int(mask.sum()), -1, -1)
        return self.original(x, emb, ctx2, **kwargs)


def make_sigma_capture_wrapper(state, prev):
    """apply_model 级 wrapper：记录 current_sigma / sample_sigmas 到 RouterState。

    链式调用前一个 wrapper（AAM 等插件同款约定：options = {input, timestep, c,
    cond_or_uncond}，真实调用形式 apply_model(input, timestep, **c)）。
    """
    def wrapper(apply_model, options):
        try:
            ts = options.get("timestep")
            if ts is not None:
                state.current_sigma = float(ts.reshape(-1)[0])
            c = options.get("c") or {}
            to = c.get("transformer_options") or {}
            ss = to.get("sample_sigmas")
            if ss is not None:
                state.sample_sigmas = ss
        except Exception:
            pass
        if prev is not None:
            return prev(apply_model, options)
        return apply_model(options["input"], options["timestep"], **options["c"])
    return wrapper
