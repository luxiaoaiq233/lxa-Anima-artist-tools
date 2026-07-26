"""lxa AAT (Anima Artist Tools) 节点定义。

AATArtistPack —— 纯编码节点（不做任何 model patch）。
对 clean_base 与每位画师分别独立编码，杜绝多画师在同一文本编码中
互相渗透（LLM 上下文渗透），为 并行混合/层-步调度 提供隔离的画师条件。
"""

import logging
import re

import torch
import torch.nn.functional as F

from .core import (
    FUSION_MODES,
    NS_KEY,
    ArtistState,
    ParallelArtistCrossAttn,
    RoutedBlockWrapper,
    RouterState,
    make_detach_strip_callback,
    make_sigma_capture_wrapper,
    recursive_unwrap,
    sanitize_weight_backups,
    unwrap_own,
)
from .guider import (
    CFG_ORDERS,
    FALLBACK_BASE,
    FALLBACK_LAST,
    MODE_ALTERNATE_EVERY,
    MODE_ALTERNATE_N,
    MODES,
    EpsilonMultiGuideGuider,
    StepAlternatorGuider,
    slice_conditioning,
)
from .scheduler import TRANSITION_FNS, parse_layer_config

logger = logging.getLogger("lxa_aat")

ARTIST_CONTEXTS_TYPE = "ARTIST_CONTEXTS"


# ---------------------------------------------------------------------------
# artist_chain 解析
# ---------------------------------------------------------------------------

def normalize_artist_label(raw_name):
    """把 'wlop' / '@wlop' / 'by wlop' 统一归一化为 '@wlop'。

    画师名本体逐字保留用户输入（如 'makoto_shinkai' 不做下划线转换、
    不翻译、不加任何前后缀）。归一化失败（空名）返回 None。
    """
    name = raw_name.strip()
    changed = True
    while changed:
        changed = False
        if name.lower().startswith("by "):
            name = name[3:].strip()
            changed = True
        if name.startswith("@"):
            name = name[1:].strip()
            changed = True
    if not name:
        return None
    return "@" + name


def parse_artist_chain(artist_chain):
    """解析 '(@wlop:1.1), makoto_shinkai' → [('@wlop', 1.1), ('@makoto_shinkai', 1.0)]。

    逗号或换行分隔；权重仅支持 '(@画师名:1.1)' 写法，无权重数值默认 1.0；
    权重解析失败回退 1.0 并告警；'名::权重' 写法无效，告警后按权重 1.0 处理。
    同名画师去重，保留首次出现。
    """
    artists = []
    seen = set()
    for entry in re.split(r"[,\n]", artist_chain or ""):
        entry = entry.strip()
        if not entry:
            continue
        name_raw, weight = entry, 1.0

        paren = re.match(r"^\((.*)\)$", entry)
        if paren:
            inner = paren.group(1).strip()
            if ":" in inner:
                name_part, _, weight_part = inner.rpartition(":")
                try:
                    weight = float(weight_part.strip())
                except ValueError:
                    logger.warning(
                        "[lxa_aat] 权重 %r 无法解析（条目 %r），回退为 1.0",
                        weight_part, entry,
                    )
                name_raw = name_part
            else:
                name_raw = inner
        elif "::" in entry:
            logger.warning(
                "[lxa_aat] '名::权重' 写法无效（条目 %r），"
                "仅支持 '(@画师名:权重)'；本条按权重 1.0 处理",
                entry,
            )
            name_raw = entry.split("::", 1)[0]

        label = normalize_artist_label(name_raw)
        if label is None:
            logger.warning("[lxa_aat] 跳过空画师名（条目 %r）", entry)
            continue
        if label in seen:
            logger.warning("[lxa_aat] 重复画师 %r，仅保留首次出现", label)
            continue
        seen.add(label)
        artists.append((label, weight))
    return artists


# ---------------------------------------------------------------------------
# 编码与序列对齐
# ---------------------------------------------------------------------------

def encode_text(clip, text):
    """与 nodes.py CLIPTextEncode.encode（ComfyUI 0.27.0, nodes.py:73-77）完全一致的入口。

    返回标准 CONDITIONING: [[cond_tensor, pooled_dict]]。
    对 Anima，pooled_dict 内含 't5xxl_ids' / 't5xxl_weights'
    （comfy/text_encoders/anima.py:48-52，经 sd.py:402-407 并入 dict）。
    """
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


def _pad_seq_dim(tensor, target_len):
    """在 dim=1（序列维）右侧零 pad 到 target_len。tensor: (B, L, ...)。"""
    cur = tensor.shape[1]
    if cur >= target_len:
        return tensor
    return F.pad(tensor, (0, 0, 0, target_len - cur))


def _pad_1d(tensor, target_len):
    """在最后一维右侧零 pad 到 target_len（用于 t5xxl_ids / t5xxl_weights）。"""
    cur = tensor.shape[0]
    if cur >= target_len:
        return tensor
    return F.pad(tensor, (0, target_len - cur))


def _single_entry(cond_list, label):
    """encode_from_tokens_scheduled 正常返回 1 段；clip schedule 激活时可能多段。"""
    if len(cond_list) != 1:
        logger.warning(
            "[lxa_aat] 画师 %s 编码返回 %d 段 cond（clip schedule?），仅使用第一段",
            label, len(cond_list),
        )
    return cond_list[0]


# ---------------------------------------------------------------------------
# 画师槽位提取（base_prompt = 含画师标签的完整提示词）
# ---------------------------------------------------------------------------

ARTIST_SLOT = "\uE000ARTIST_SLOT\uE001"  # 私用区槽位标记，正常提示词不会出现


def _clean_prompt_residue(text):
    """清理移除画师标签后的残留：连续逗号、逗号前后多余空格、句首/句尾逗号、双空格。"""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)
    text = re.sub(r"^[\s,]+|[\s,]+$", "", text)
    return text.strip()


def extract_artist_slot(base_prompt, labels):
    """在 base_prompt 全文中逐字匹配并移除所有画师标签。

    按标签长度降序收集匹配（防 '@hal' 误伤 '@hal aluha' 这类包含关系），
    重叠时保留较长者；每个标签移除全部出现位置；最左侧出现位置替换为
    ARTIST_SLOT 槽位标记，其字符索引即画师槽位。

    返回 (text_with_slot, slot_index)；一个标签都没匹配到时返回 None。
    """
    spans = []  # (start, end, label_len)
    for label in labels:
        for m in re.finditer(re.escape(label), base_prompt):
            spans.append((m.start(), m.end(), len(label)))
    if not spans:
        return None
    # 重叠消解：长度降序优先，其次位置靠前；接受不重叠的区间
    accepted = []
    for s, e, ln in sorted(spans, key=lambda t: (-t[2], t[0])):
        if all(e <= s2 or s >= e2 for s2, e2, _ in accepted):
            accepted.append((s, e, ln))
    accepted.sort(key=lambda t: t[0])

    slot_start = accepted[0][0]
    parts = []
    prev = 0
    for s, e, _ in accepted:
        parts.append(base_prompt[prev:s])
        if s == slot_start:
            parts.append(ARTIST_SLOT)
        prev = e
    parts.append(base_prompt[prev:])
    text = _clean_prompt_residue("".join(parts))
    return text, text.find(ARTIST_SLOT)


def _token_counts(clip, text):
    """分词计数（不跑模型）。返回 (qwen_count, t5_count)；非 Anima 键缺失时回退/None。"""
    if not text or not text.strip():
        return 0, 0
    tk = clip.tokenize(text)
    if not isinstance(tk, dict) or not tk:
        return 0, None
    q = tk.get("qwen3_06b")
    if q is None:
        q = next(iter(tk.values()))
    qc = sum(len(chunk) for chunk in q)
    t5 = tk.get("t5xxl")
    tc = sum(len(chunk) for chunk in t5) if t5 is not None else None
    return qc, tc


def _compute_boundaries(clip, prefix_text, suffix_text, labels, token_lengths):
    """导出 token 级三段边界（Blender 选项 a：Pack 侧计算，向后兼容增量字段）。

    BPE 在段边界可能有 ±1 token 合并误差：画师区长一律用
    token_lengths 真长反推校正（zone = L_i − prefix − suffix，clamp ≥0），
    保证 prefix + zone + suffix == L_i 严格成立。
    """
    P, P5 = _token_counts(clip, prefix_text)
    S, S5 = _token_counts(clip, suffix_text)
    zone_q, zone_t = [], []
    for i, label in enumerate(labels):
        zq, zt = _token_counts(clip, label)
        L_i = int(token_lengths[i].get("qwen") or 0)
        zq_rec = max(0, L_i - P - S)
        if zq_rec != zq:
            logger.debug(
                "[lxa_aat] 画师 %s 画师区 token 数 %d ≠ 真长反推 %d（BPE 边界合并），以反推为准",
                label, zq, zq_rec,
            )
        zone_q.append(zq_rec)
        t5_i = int(token_lengths[i].get("t5") or 0)
        if P5 is not None and S5 is not None:
            zone_t.append(max(0, t5_i - P5 - S5))
        else:
            zone_t.append(zt)
    return {
        "qwen": {"prefix": P, "suffix": S, "zone": zone_q},
        "t5": {"prefix": P5, "suffix": S5, "zone": zone_t},
    }


class AATArtistPack:
    """解析 artist_chain，对每位画师独立编码（画师槽位替换式）。

    base_prompt 接收【包含画师标签的完整提示词】：节点逐字匹配并移除所有
    画师标签得到 clean_base（base_conditioning 的编码文本），再在画师槽位处
    分别插入 '@画师i' 作为各画师的编码文本——绝不允许把多个画师名写进同一次
    编码。输出 ARTIST_CONTEXTS 与可直接接 KSampler positive 的 base
    CONDITIONING。本节点不做任何 model patch。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "base_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "包含画师标签（@xxx）的完整提示词，通常来自上游文本拼接节点；"
                               "与 artist_chain 两串需保持一致。画师标签会被逐字移除，"
                               "并在原槽位分别插入单个画师名独立编码。",
                }),
                "artist_chain": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "逗号或换行分隔的画师列表；权重写法 '(@画师名:1.1)'，"
                               "无权重数值默认 1.0。'wlop' / '@wlop' / 'by wlop' 均归一化为 '@wlop'。",
                }),
            }
        }

    RETURN_TYPES = (ARTIST_CONTEXTS_TYPE, "CONDITIONING")
    RETURN_NAMES = ("artist_contexts", "base_conditioning")
    FUNCTION = "build"
    CATEGORY = "lxa_aat"
    DESCRIPTION = "从含画师标签的完整提示词中提取槽位，独立编码 clean_base 与每位画师。"

    def build(self, clip, base_prompt, artist_chain):
        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\n"
                "If the clip is from a checkpoint loader node your checkpoint "
                "does not contain a valid clip or text encoder model."
            )
        base_prompt = base_prompt or ""
        artists = parse_artist_chain(artist_chain)
        labels = [label for label, _ in artists]

        # 画师槽位提取：clean_base + 槽位替换文本；一个标签都没匹配到时回退旧行为
        extraction = extract_artist_slot(base_prompt, labels) if artists else None
        if extraction is not None:
            text_with_slot, slot_idx = extraction
            base_text = _clean_prompt_residue(text_with_slot.replace(ARTIST_SLOT, ""))
            artist_text = {label: text_with_slot.replace(ARTIST_SLOT, label)
                           for label in labels}
            # 三段边界文本（供 token 级边界导出；槽位前后即前缀/后缀）
            prefix_text = _clean_prompt_residue(text_with_slot[:slot_idx])
            suffix_text = _clean_prompt_residue(text_with_slot[slot_idx + len(ARTIST_SLOT):])
            logger.info(
                "[lxa_aat] 画师槽位 @ 字符 %d；clean_base=%r",
                slot_idx, base_text[:80],
            )
        else:
            if artists:
                logger.warning(
                    "[lxa_aat] base_prompt 中未匹配到任何画师标签"
                    "（base_prompt 与 artist_chain 可能不一致），回退为末尾追加旧行为"
                )
            base_text = base_prompt
            artist_text = {
                label: (f"{base_prompt.strip()}, {label}" if base_prompt.strip() else label)
                for label in labels
            }
            # 回退结构为 'base, @画师'：槽位在 base 末尾，无后缀
            prefix_text = base_prompt
            suffix_text = ""

        # base 单独编码一次（clean_base，绝不含任何画师标签）
        base_cond = encode_text(clip, base_text)

        if not artists:
            logger.warning("[lxa_aat] artist_chain 为空，输出不含任何画师")
            return ({
                "labels": [],
                "weights": [],
                "conditionings": [],
                "token_lengths": [],
                "base_conditioning": base_cond,
                "base_prompt": base_prompt,
                "clean_base": base_text,
                "boundaries": None,
            }, base_cond)

        # 1) 每位画师独立编码（槽位处插入各自的 '@画师名'，绝不合并在同一次编码中）
        encoded = []
        for label, weight in artists:
            tensor, meta = _single_entry(encode_text(clip, artist_text[label]), label)
            encoded.append({
                "label": label,
                "weight": weight,
                "tensor": tensor,
                "meta": dict(meta),  # 复制，避免改动 encode 返回的原始 dict
            })

        # 2) 序列维对齐：pad 到最长者（零向量），真实长度记入 token_lengths。
        #    qwen 隐藏态 (1, L_q, D)；t5xxl_ids / t5xxl_weights (L_t,) 一并对齐，
        #    供后续 batched forward 使用；运行时需按 token_lengths 截回真实长度。
        max_q = max(int(e["tensor"].shape[1]) for e in encoded)
        max_t5 = max(
            (int(e["meta"]["t5xxl_ids"].shape[0]) for e in encoded
             if e["meta"].get("t5xxl_ids") is not None),
            default=0,
        )

        conditionings = []
        token_lengths = []
        for e in encoded:
            q_len = int(e["tensor"].shape[1])
            ids = e["meta"].get("t5xxl_ids")
            t5_len = int(ids.shape[0]) if ids is not None else 0

            padded_tensor = _pad_seq_dim(e["tensor"], max_q)
            if ids is not None:
                e["meta"]["t5xxl_ids"] = _pad_1d(ids, max_t5)
            t5_weights = e["meta"].get("t5xxl_weights")
            if t5_weights is not None:
                e["meta"]["t5xxl_weights"] = _pad_1d(t5_weights, max_t5)

            conditionings.append([[padded_tensor, e["meta"]]])
            token_lengths.append({"qwen": q_len, "t5": t5_len})
            logger.info(
                "[lxa_aat] 画师 %s: weight=%.3g, qwen_len=%d, t5_len=%d",
                e["label"], e["weight"], q_len, t5_len,
            )

        # 3) token 级三段边界（Blender 选项 a：Pack 侧导出，向后兼容增量字段）
        boundaries = _compute_boundaries(clip, prefix_text, suffix_text,
                                         [e["label"] for e in encoded], token_lengths)
        logger.info(
            "[lxa_aat] 边界: qwen prefix=%d suffix=%d zone=%s",
            boundaries["qwen"]["prefix"], boundaries["qwen"]["suffix"],
            boundaries["qwen"]["zone"],
        )

        contexts = {
            "labels": [e["label"] for e in encoded],
            "weights": [e["weight"] for e in encoded],  # 归一化前的原始值
            "conditionings": conditionings,
            "token_lengths": token_lengths,
            "base_conditioning": base_cond,
            "base_prompt": base_prompt,
            "clean_base": base_text,
            "boundaries": boundaries,
        }
        return (contexts, base_cond)


class AATParallelArtistMixer:
    """并行混合：把每位画师作为独立 cross-attn 分支，输出在特征空间加权融合。

    patch 点仅为 diffusion_model.blocks.{i}.cross_attn；先 model.clone() 再 patch；
    transformer_options["lxa_aat"] 写入 l1_installed 标记供 层-步调度 检测。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artist_contexts": (ARTIST_CONTEXTS_TYPE,),
                "fusion_mode": (list(FUSION_MODES), {"default": "output_avg"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "start_block": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "end_block": ("INT", {
                    "default": -1, "min": -1, "max": 9999,
                    "tooltip": "-1 = 到最后一层",
                }),
                "apply_to_uncond": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "ema_alpha": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "跨步 EMA 系数（fp32 缓存；sigma 回升即新采样开始时清空）。0 = 关闭。",
                }),
                "static_capture_k": ("INT", {
                    "default": 0, "min": 0, "max": 999,
                    "tooltip": "前 K 步平均后冻结复用。0 = 关闭。",
                }),
                "lowrank_k": ("INT", {
                    "default": 1, "min": 1, "max": 99,
                    "tooltip": "lowrank_delta：SVD top-k 子空间维数（仅 fusion_mode=lowrank_delta 生效）",
                }),
                "lowrank_normalize_weights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "lowrank_delta：权重归一化 wᵢ/Σ|wᵢ|（AAM 同语义，默认开）",
                }),
                "rms_clamp_ratio": ("FLOAT", {
                    "default": 1.2, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "lowrank_delta：RMS 钳制阈值倍率（超出 base_out RMS × 该值时钳回；0 = 关闭）",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "lxa_aat"
    DESCRIPTION = "并行混合：每位画师独立 K/V 前向，特征空间加权融合（Cross-Attention 并行分支）。"

    def patch(self, model, artist_contexts, fusion_mode, strength, start_block,
              end_block, apply_to_uncond, ema_alpha=0.0, static_capture_k=0,
              lowrank_k=1, lowrank_normalize_weights=True, rms_clamp_ratio=1.0):
        labels = list(artist_contexts.get("labels", []))
        if not labels:
            logger.warning("[lxa_aat] 并行混合: artist_contexts 为空，原样返回 MODEL")
            return (model,)

        dm = model.get_model_object("diffusion_model")
        if not hasattr(dm, "blocks") or len(dm.blocks) == 0:
            raise ValueError(
                "[lxa_aat] 并行混合: 目标模型没有 .blocks "
                f"（{type(dm).__name__}，非 Anima/MiniTrainDIT 架构）"
            )
        num_blocks = len(dm.blocks)
        start = max(0, int(start_block))
        end = num_blocks - 1 if end_block < 0 or end_block >= num_blocks else int(end_block)
        if start > end:
            raise ValueError(
                f"[lxa_aat] 并行混合: start_block({start}) > end_block({end})"
            )

        m = model.clone()
        # 串联顺序自检：层-步调度（整层包装）必须在 并行混合 下游；若目标 block 已被整体包装，
        # patch_model 应用顺序会让 并行混合 落到 层-步调度 wrapper 上（空挂），并行混合 失效。
        if any(f"diffusion_model.blocks.{i}" in m.object_patches
               for i in range(start, end + 1)):
            logger.warning(
                "[lxa_aat] 并行混合: 检测到目标 block 已被整体包装（层-步调度 在 并行混合 上游？）。"
                "正确串联顺序为 并行混合 → 层-步调度，否则 并行混合 不会生效。"
            )
        state = ArtistState(
            dm, labels,
            artist_contexts.get("weights", [1.0] * len(labels)),
            artist_contexts["conditionings"],
            artist_contexts["token_lengths"],
        )

        patched = 0
        patched_prefixes = set()
        own_wrappers = {}
        for i in range(start, end + 1):
            name = f"diffusion_model.blocks.{i}.cross_attn"
            try:
                # 当前链顶模块（含外部插件 wrapper）；只解开本套件自己的 wrapper
                current = m.get_model_object(name)
                base_mod = unwrap_own(current, ParallelArtistCrossAttn)
                bottom = recursive_unwrap(base_mod)
                if bottom is None or not callable(getattr(bottom, "forward", None)):
                    raise TypeError(f"底层模块 {type(bottom).__name__} 无 forward")
                wrapper = ParallelArtistCrossAttn(
                    base_mod, state, i,
                    fusion_mode=fusion_mode, strength=strength,
                    block_range=(start, end), apply_to_uncond=apply_to_uncond,
                    ema_alpha=ema_alpha, static_capture_k=static_capture_k,
                    lowrank_k=lowrank_k,
                    lowrank_normalize=lowrank_normalize_weights,
                    rms_clamp_ratio=rms_clamp_ratio,
                )
                m.add_object_patch(name, wrapper)
                patched += 1
                patched_prefixes.add(name)
                own_wrappers[name] = wrapper
            except Exception as e:
                # 任何一层失败只跳过该层并打印日志，不中断整体
                logger.warning("[lxa_aat] 并行混合 第 %d 层 patch 失败（跳过）: %s", i, e)

        if patched == 0:
            logger.error("[lxa_aat] 并行混合: 所有层 patch 均失败，原样返回 MODEL")
            return (model,)

        # 权重备份键卫生（动态 VRAM 加载器会把 wrapper 路径写进跨 clone 共享的
        # backup dict，导致后续干净 run restore 崩溃，详见 core.sanitize_weight_backups）。
        # 在 patch 时、ON_PRE_RUN、ON_CLEANUP 三处重映射；回调随 clone 链传播。
        # ON_DETACH 再按身份剥离本 clone 的 wrapper，防残留泄漏到后续干净运行。
        from comfy.patcher_extension import CallbacksMP

        def _sanitize_cb(patcher, *_args):
            sanitize_weight_backups(patcher, patched_prefixes)

        m.add_callback(CallbacksMP.ON_PRE_RUN, _sanitize_cb)
        m.add_callback(CallbacksMP.ON_CLEANUP, _sanitize_cb)
        m.add_callback(CallbacksMP.ON_DETACH, make_detach_strip_callback(own_wrappers))
        sanitize_weight_backups(m, patched_prefixes)

        ns = m.model_options.setdefault("transformer_options", {}).setdefault(NS_KEY, {})
        ns.update({
            "l1_installed": True,
            "fusion_mode": fusion_mode,
            "strength": float(strength),
        })
        logger.info(
            "[lxa_aat] 并行混合: %d 位画师 × block [%d, %d]（patch 成功 %d 层），"
            "fusion=%s, strength=%.3g, apply_to_uncond=%s",
            state.n_artists, start, end, patched, fusion_mode, strength, apply_to_uncond,
        )
        return (m,)


class AATLayerStepScheduler:
    """层-步调度：按 block 索引 × 采样进度（sigma 区间）把画师路由到不同层/步区间。

    patch 点仅为 diffusion_model.blocks.{i}（block 整体 forward）；
    检测到 transformer_options["lxa_aat"].l1_installed 时为协同模式
    （只写激活集合，不碰 context），否则为独立模式（替换 cond 行 context 为
    激活画师的加权混合 conditioning）。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artist_contexts": (ARTIST_CONTEXTS_TYPE,),
                "layer_config": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "每行 '画师序号::权重@层范围,step=lo-hi'（step 可省略=全程）。"
                               "层/步均闭区间，进度 0=高噪 1=结束；后声明优先覆盖。\n"
                               "示例：\n"
                               "0::1.0@0-13\n"
                               "1::0.8@14-27,step=0-0.6",
                }),
                "transition_fn": (list(TRANSITION_FNS), {"default": "cosine"}),
                "transition_width": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "过渡区宽度（进度单位，收缩式作用于区间内侧）；0 = 硬切换。",
                }),
            },
            "optional": {
                "apply_to_uncond": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "lxa_aat"
    DESCRIPTION = "层-步调度：按 DiT block 索引与采样进度（sigma 区间）路由画师生效范围。"

    def patch(self, model, artist_contexts, layer_config, transition_fn,
              transition_width, apply_to_uncond=False):
        labels = list(artist_contexts.get("labels", []))
        if not labels:
            logger.warning("[lxa_aat] 层-步调度: artist_contexts 为空，原样返回 MODEL")
            return (model,)
        dm = model.get_model_object("diffusion_model")
        if not hasattr(dm, "blocks") or len(dm.blocks) == 0:
            raise ValueError(
                "[lxa_aat] 层-步调度: 目标模型没有 .blocks "
                f"（{type(dm).__name__}，非 Anima/MiniTrainDIT 架构）"
            )
        num_blocks = len(dm.blocks)
        layer_map = parse_layer_config(layer_config, len(labels), num_blocks)
        if not layer_map:
            logger.warning("[lxa_aat] 层-步调度: layer_config 无有效条目，原样返回 MODEL")
            return (model,)

        m = model.clone()
        # 协同模式检测：命名空间内是否存在 并行混合 写入的标记
        ns = m.model_options.get("transformer_options", {}).get(NS_KEY, {})
        coop = bool(ns.get("l1_installed"))

        state = RouterState(
            ArtistState(
                dm, labels,
                artist_contexts.get("weights", [1.0] * len(labels)),
                artist_contexts["conditionings"],
                artist_contexts["token_lengths"],
            ),
            layer_map, transition_fn, transition_width, coop,
        )

        patched = 0
        patched_prefixes = set()
        own_wrappers = {}
        for i in sorted(layer_map):
            name = f"diffusion_model.blocks.{i}"
            try:
                current = m.get_model_object(name)
                base_mod = unwrap_own(current, RoutedBlockWrapper)
                bottom = recursive_unwrap(base_mod)
                if bottom is None or not hasattr(bottom, "cross_attn"):
                    raise TypeError(f"底层模块 {type(bottom).__name__} 无 cross_attn")
                wrapper = RoutedBlockWrapper(base_mod, state, i,
                                             apply_to_uncond=apply_to_uncond)
                m.add_object_patch(name, wrapper)
                patched += 1
                patched_prefixes.add(name)
                own_wrappers[name] = wrapper
            except Exception as e:
                # 任何一层失败只跳过该层并打印日志，不中断整体
                logger.warning("[lxa_aat] 层-步调度 第 %d 层 patch 失败（跳过）: %s", i, e)

        if patched == 0:
            logger.error("[lxa_aat] 层-步调度: 所有层 patch 均失败，原样返回 MODEL")
            return (model,)

        # sigma 表捕获（apply_model 级，链式调用前一个 wrapper）
        prev = m.model_options.get("model_function_wrapper")
        m.set_model_unet_function_wrapper(make_sigma_capture_wrapper(state, prev))

        # 权重备份键卫生（整层包装同样会产生 '.original.' 脏键，见 core 注释）；
        # ON_DETACH 再按身份剥离本 clone 的 wrapper，防残留泄漏到后续干净运行。
        from comfy.patcher_extension import CallbacksMP

        def _sanitize_cb(patcher, *_args):
            sanitize_weight_backups(patcher, patched_prefixes)

        m.add_callback(CallbacksMP.ON_PRE_RUN, _sanitize_cb)
        m.add_callback(CallbacksMP.ON_CLEANUP, _sanitize_cb)
        m.add_callback(CallbacksMP.ON_DETACH, make_detach_strip_callback(own_wrappers))
        sanitize_weight_backups(m, patched_prefixes)

        logger.info(
            "[lxa_aat] 层-步调度: %s 模式，%d 层路由（共 %d 条声明），"
            "transition=%s/%.3g, apply_to_uncond=%s",
            "协同" if coop else "独立", patched,
            sum(len(v) for v in layer_map.values()),
            transition_fn, transition_width, apply_to_uncond,
        )
        return (m,)


class AATConditioningBlender:
    """条件混合器：把多位画师的 conditioning 在条件空间直接混合成一条普通 CONDITIONING。

    原理：各画师条件共享同一 base 前缀（Qwen3 因果注意力保证画师槽位之前的
    token 逐位相同），差异集中在画师区 token 与其后的后缀 token。
    三段处理：前缀直通（取首位画师）；画师区按槽位对齐、短者零 pad 后加权求和；
    后缀按相同系数加权求和（禁止直通某一位画师的后缀——会造成不对称画师泄漏）。
    混合后只剩一组融合 token，softmax 竞争从结构上消失。
    不打任何 patch、不留任何状态。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "artist_contexts": (ARTIST_CONTEXTS_TYPE,),
                "blend_coeff": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "节点系数：最终混合系数 = Pack权重 × 节点系数。"
                               "默认 0.6（双画师即 0.6/0.6，允许 Σ>1）。",
                }),
                "renorm": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "开启后把混合系数归一化到 Σ=1（Σ=0 时自动跳过，行为不变）。",
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("blended_conditioning",)
    FUNCTION = "blend"
    CATEGORY = "lxa_aat"
    DESCRIPTION = "把多位画师的 conditioning 在条件空间按三段（前缀/画师区/后缀）混合成一条 CONDITIONING。"

    def blend(self, artist_contexts, blend_coeff, renorm):
        labels = list(artist_contexts.get("labels", []))
        base_cond = artist_contexts.get("base_conditioning")
        if not labels:
            logger.warning("[lxa_aat] Blender: artist_contexts 为空，直通 base_conditioning")
            return (base_cond,)

        conds = artist_contexts["conditionings"]
        tls = artist_contexts["token_lengths"]

        if len(labels) == 1:
            logger.warning("[lxa_aat] Blender: 仅 1 位画师，直通其 conditioning")
            t, meta = conds[0][0]
            q = tls[0].get("qwen") or t.shape[1]
            t5 = tls[0].get("t5") or 0
            new_meta = dict(meta)
            if new_meta.get("t5xxl_ids") is not None and t5:
                new_meta["t5xxl_ids"] = new_meta["t5xxl_ids"][:t5]
                if new_meta.get("t5xxl_weights") is not None:
                    new_meta["t5xxl_weights"] = new_meta["t5xxl_weights"][:t5]
            return ([[t[:, :q], new_meta]],)

        # 混合系数 = Pack权重 × 节点系数；允许 Σ>1；可选归一化（Σ=0 跳过）
        weights = artist_contexts.get("weights", [1.0] * len(labels))
        coeffs = [float(w) * float(blend_coeff) for w in weights]
        total = sum(coeffs)
        if renorm and abs(total) > 1e-8:
            coeffs = [c / total for c in coeffs]
        if all(abs(c) < 1e-8 for c in coeffs):
            logger.info("[lxa_aat] Blender: 混合系数全 0，直通 base_conditioning")
            return (base_cond,)

        b = artist_contexts.get("boundaries")
        if not b:
            logger.warning(
                "[lxa_aat] Blender: 缺少 boundaries 字段（旧版 Pack?），"
                "直通 base_conditioning"
            )
            return (base_cond,)
        P = int(b["qwen"]["prefix"])
        S = int(b["qwen"]["suffix"])
        Zs = [int(z) for z in b["qwen"]["zone"]]
        Z_max = max(Zs) if Zs else 0
        L_new = P + Z_max + S

        src = conds[0][0][0]
        dev, dt = src.device, src.dtype
        out = torch.zeros(1, L_new, src.shape[-1], dtype=torch.float32, device=dev)

        # 前缀：各画师逐位相同 → 直通（取首位画师）
        if P > 0:
            out[:, :P] = src[:, :P].float()
        # 画师区：按槽位对齐、短者零 pad 后加权求和；后缀：同权加权求和
        for i, c in enumerate(coeffs):
            if abs(c) < 1e-12:
                continue
            t = conds[i][0][0]
            L_i = int(tls[i].get("qwen") or t.shape[1])
            if Zs[i] > 0:
                out[:, P:P + Zs[i]] += c * t[:, P:P + Zs[i]].float()
            if S > 0:
                out[:, P + Z_max:] += c * t[:, L_i - S:L_i].float()
        out = out.to(dt)

        # 元数据：ids 取权重最高画师一路；数值型 weights 同权混合（零 pad 后 Σcᵢ·wᵢ）
        top = max(range(len(labels)), key=lambda i: coeffs[i])
        meta_top = dict(conds[top][0][1])
        t5_top = int(tls[top].get("t5") or 0)
        if meta_top.get("t5xxl_ids") is not None and t5_top:
            meta_top["t5xxl_ids"] = meta_top["t5xxl_ids"][:t5_top]
            if meta_top.get("t5xxl_weights") is not None:
                max_t5 = max(int(tl.get("t5") or 0) for tl in tls)
                acc = torch.zeros(max_t5, dtype=torch.float32, device=out.device)
                for i, c in enumerate(coeffs):
                    wi = conds[i][0][1].get("t5xxl_weights")
                    if wi is not None:
                        acc += c * wi.to(device=out.device, dtype=torch.float32)
                meta_top["t5xxl_weights"] = acc[:t5_top].to(meta_top["t5xxl_weights"].dtype)
        meta_top["aat_qwen_len"] = L_new

        logger.info(
            "[lxa_aat] Blender: %d 位画师混合完成 L_new=%d (P=%d Z_max=%d S=%d)，"
            "系数=%s%s", len(labels), L_new, P, Z_max, S,
            ["%.3g" % c for c in coeffs], ", renorm" if renorm else "",
        )
        return ([[out, meta_top]],)


class AATStepAlternator:
    """AAT Step Alternator (Guider)：按序数步轮换画师条件的自定义 GUIDER。

    序数步控制需要"现在是第几步"，普通节点求值发生在采样开始前拿不到总步数，
    因此实现为自定义 GUIDER（guider 每步被调用，经 sigmas 表换算当前步索引），
    配合 SamplerCustomAdvanced（guider 输入）使用。内部完整保留 CFGGuider 的
    cond/uncond 行为，只在每步切换 positive 条件键，不做任何 model patch。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artist_contexts": (ARTIST_CONTEXTS_TYPE,),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "mode": (list(MODES), {"default": MODE_ALTERNATE_EVERY}),
                "n_every": ("INT", {
                    "default": 2, "min": 1, "max": 999,
                    "tooltip": "alternate_n 模式：每 N 步轮换一次",
                }),
                "fallback": ((FALLBACK_BASE, FALLBACK_LAST), {
                    "default": FALLBACK_BASE,
                    "tooltip": "custom_ranges 未覆盖的步：base=回退 base 条件（无画师注入）；"
                               "last=延续上一位激活画师",
                }),
                "final_k": ("INT", {
                    "default": 0, "min": 0, "max": 999,
                    "tooltip": "最后 K 步固定画师收尾（防末期细节抖动）；0 = 关闭",
                }),
                "final_artist": ("INT", {
                    "default": -1, "min": -1, "max": 99,
                    "tooltip": "收尾画师序号；-1 = 上一位激活画师",
                }),
                "debug_logging": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "开启后逐步打印当前激活画师（调试用）",
                }),
            },
            "optional": {
                "range1_enable": ("BOOLEAN", {"default": False}),
                "range1_artist": ("INT", {"default": 0, "min": 0, "max": 99}),
                "range1_start": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "range1_end": ("INT", {"default": -1, "min": -1, "max": 9999,
                                       "tooltip": "-1 = 末步"}),
                "range2_enable": ("BOOLEAN", {"default": False}),
                "range2_artist": ("INT", {"default": 1, "min": 0, "max": 99}),
                "range2_start": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "range2_end": ("INT", {"default": -1, "min": -1, "max": 9999}),
                "range3_enable": ("BOOLEAN", {"default": False}),
                "range3_artist": ("INT", {"default": 2, "min": 0, "max": 99}),
                "range3_start": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "range3_end": ("INT", {"default": -1, "min": -1, "max": 9999}),
            },
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "build"
    CATEGORY = "lxa_aat"
    DESCRIPTION = "序数步画师轮换的自定义 GUIDER（配合 SamplerCustomAdvanced 使用；内部保留完整 CFG）。"

    def build(self, model, artist_contexts, negative, cfg, mode, n_every,
              fallback, final_k, final_artist, debug_logging,
              range1_enable=False, range1_artist=0, range1_start=0, range1_end=-1,
              range2_enable=False, range2_artist=1, range2_start=0, range2_end=-1,
              range3_enable=False, range3_artist=2, range3_start=0, range3_end=-1):
        labels = list(artist_contexts.get("labels", []))
        base_positive = artist_contexts.get("base_conditioning")

        if not labels:
            logger.warning(
                "[lxa_aat] Alternator: artist_contexts 为空，"
                "退化为普通 CFGGuider 行为（positive = base_conditioning）"
            )
            artist_conds = []
        else:
            n_total = len(labels)
            if n_total > 3:
                logger.warning(
                    "[lxa_aat] Alternator: 画师数 %d 超过 3，仅取前 3 位（%s）",
                    n_total, ", ".join(labels[:3]),
                )
            n_use = min(n_total, 3)
            conds = artist_contexts["conditionings"]
            tls = artist_contexts["token_lengths"]
            # 零 pad 条件切回真长（防零向量 token 稀释注意力，M1 契约）
            artist_conds = [slice_conditioning(conds[i], tls[i]) for i in range(n_use)]

        # custom_ranges 槽位翻译（越界跳过 + WARNING）
        ranges = []
        for en, a, s, e in (
            (range1_enable, range1_artist, range1_start, range1_end),
            (range2_enable, range2_artist, range2_start, range2_end),
            (range3_enable, range3_artist, range3_start, range3_end),
        ):
            if not en:
                continue
            if not (0 <= int(a) < max(1, len(artist_conds))):
                logger.warning(
                    "[lxa_aat] Alternator: range 画师序号 %d 越界（共 %d 位），该槽位跳过",
                    a, len(artist_conds),
                )
                continue
            ranges.append((int(a), int(s), int(e)))

        config = {
            "mode": mode,
            "n_every": int(n_every),
            "ranges": ranges,
            "fallback": fallback,
            "final_k": int(final_k),
            "final_artist": int(final_artist),
            "n_artists": len(artist_conds),
            "debug_logging": bool(debug_logging),
        }
        guider = StepAlternatorGuider(model, config)
        guider.set_conds(base_positive, negative, artist_conds)
        guider.set_cfg(float(cfg))
        logger.info(
            "[lxa_aat] Alternator: %d 位画师轮换，mode=%s n=%d ranges=%s "
            "fallback=%s final_k=%d final_artist=%d cfg=%.3g",
            len(artist_conds), mode, n_every, ranges,
            fallback, final_k, final_artist, cfg,
        )
        return (guider,)


class AATEpsilonMultiGuide:
    """AAT Epsilon Multi-Guide：每步在网络输出端合成 ε = ε_base + Σ sᵢ·(εᵢ − ε_base)。

    cond batch 塞 [uncond, base, base+@A, base+@B(, base+@C)]，一次前向后合成，
    模型内部零接触；内部完成 CFG（同 Step Alternator 约束，不退化为 cfg=1）。
    复用 Guider 骨架（comfy/samplers.py:1182 系），配合 SamplerCustomAdvanced。
    """

    @classmethod
    def INPUT_TYPES(cls):
        strengths = {}
        defaults = {"s_a": 0.5, "s_b": 0.5}
        for i, letter in enumerate("abcdefgh"):
            strengths[f"s_{letter}"] = ("FLOAT", {
                "default": defaults.get(f"s_{letter}", 0.0), "min": -4.0, "max": 4.0, "step": 0.05,
                "tooltip": f"画师 {letter.upper()}（序号 {i}）的强度系数；0 = 该画师不占 batch 路数",
            })
        return {
            "required": {
                "model": ("MODEL",),
                "artist_contexts": (ARTIST_CONTEXTS_TYPE,),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                **strengths,
                "cfg_order": (list(CFG_ORDERS), {
                    "default": "stack_then_cfg",
                    "tooltip": "CFG 结合次序：stack_then_cfg=先 delta 堆叠再做 CFG（默认）；"
                               "cfg_then_stack=各条件先各自 CFG 再堆叠（线性 CFG 下两者恒等）",
                }),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "build"
    CATEGORY = "lxa_aat"
    DESCRIPTION = "每步在网络输出端按 ε = ε_base + Σ sᵢ·(εᵢ − ε_base) 合成（Guider，最多 8 位画师，CFG 次序可切换）。"

    def build(self, model, artist_contexts, negative, cfg, cfg_order, **strength_kwargs):
        labels = list(artist_contexts.get("labels", []))
        base_positive = artist_contexts.get("base_conditioning")

        strengths_all = [float(strength_kwargs[f"s_{letter}"]) for letter in "abcdefgh"]
        if not labels:
            logger.warning(
                "[lxa_aat] EpsilonGuide: artist_contexts 为空，"
                "退化为普通 CFGGuider 行为（positive = base_conditioning）"
            )
            artist_conds = []
            strengths = []
        else:
            if len(labels) > 8:
                logger.warning(
                    "[lxa_aat] EpsilonGuide: 画师数 %d 超过 8，仅取前 8 位（%s）",
                    len(labels), ", ".join(labels[:8]),
                )
            n_use = min(len(labels), 8)
            conds = artist_contexts["conditionings"]
            tls = artist_contexts["token_lengths"]
            # 零 pad 条件切回真长（防零向量 token 稀释注意力，M1 契约）
            artist_conds = [slice_conditioning(conds[i], tls[i]) for i in range(n_use)]
            # 空栏位静默无视：Pack 不足 8 位时，超出数量的 s_x 栏位不告警不生效
            strengths = strengths_all[:n_use]

        guider = EpsilonMultiGuideGuider(model, strengths, cfg_order)
        guider.set_conds(base_positive, negative, artist_conds)
        guider.set_cfg(float(cfg))
        logger.info(
            "[lxa_aat] EpsilonGuide: %d 位画师，s=%s, cfg=%.3g, cfg_order=%s",
            len(artist_conds), strengths, cfg, cfg_order,
        )
        return (guider,)


NODE_CLASS_MAPPINGS = {
    # lxa AAT v0.1.1：仅 6 个新类名；旧类别名已全部移除（v0.1.1 起），
    # 含旧类名的工作流需手动替换节点类名
    "AATArtistPack": AATArtistPack,
    "AATParallelArtistMixer": AATParallelArtistMixer,
    "AATLayerStepScheduler": AATLayerStepScheduler,
    "AATConditioningBlender": AATConditioningBlender,
    "AATStepAlternator": AATStepAlternator,
    "AATEpsilonMultiGuide": AATEpsilonMultiGuide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 仅新类名（v0.1.1：别名已移除，前端菜单干净 6 行）
    "AATArtistPack": "AAT Artist Pack (Encode)",
    "AATParallelArtistMixer": "AAT Parallel Artist Mixer",
    "AATLayerStepScheduler": "AAT Layer-Step Scheduler",
    "AATConditioningBlender": "AAT Conditioning Blender",
    "AATStepAlternator": "AAT Step Alternator (Guider)",
    "AATEpsilonMultiGuide": "AAT Epsilon Multi-Guide",
}
