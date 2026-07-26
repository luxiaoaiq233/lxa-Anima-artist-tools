"""lxa AAT 离线单元测试：fake harness（fake DiT / fake ModelPatcher / fake CLIP）。

无模型、无 GPU、无本机路径依赖；在 ComfyUI 仓库根或 custom_nodes 下运行：
    python -m pytest custom_nodes/lxa_aat/tests/ -q
"""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import lxa_aat
import lxa_aat.core as core
import lxa_aat.guider as gd
import lxa_aat.nodes as nodes_mod
import lxa_aat.scheduler as sch

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# fake harness
# ---------------------------------------------------------------------------

class FakeAttn(nn.Module):
    """out = x + context.mean(逐行)：输出仅随 context 平移，便于解析验证。"""

    def forward(self, x, context=None, rope_emb=None, transformer_options={}):
        return x + context.mean(dim=(1, 2), keepdim=True)


class ForeignWrapper(nn.Module):
    """模拟 IPAdapter/AAM 类第三方 wrapper：包一层并 +100。"""

    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x, context=None, rope_emb=None, transformer_options={}):
        return self.original(x, context, rope_emb=rope_emb,
                             transformer_options=transformer_options) + 100.0


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = FakeAttn()

    def forward(self, x, emb, context, **kwargs):
        return self.cross_attn(x, context,
                               transformer_options=kwargs.get("transformer_options", {}))


class FakeDM(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock() for _ in range(n)])
        self.preprocess_calls = []

    def preprocess_text_embeds(self, embeds, ids, t5xxl_weights=None):
        self.preprocess_calls.append({
            "embeds_shape": tuple(embeds.shape),
            "ids_shape": tuple(ids.shape),
            "ids": ids.clone(),
            "w_shape": None if t5xxl_weights is None else tuple(t5xxl_weights.shape),
        })
        v = embeds.mean(dim=1, keepdim=True)
        if t5xxl_weights is not None:
            v = v * t5xxl_weights.mean()
        return v


class FakePatcher:
    """模拟 ModelPatcher 的相关接口（clone 隔离 / 共享 backup / 回调）。"""

    def __init__(self, model_ns, ns=None):
        self.model = model_ns
        self.object_patches = {}
        self.object_patches_backup = {}
        self.model_options = {"transformer_options": {}}
        if ns:
            self.model_options["transformer_options"][core.NS_KEY] = ns
        self.backup = {}
        self.backup_buffers = {}
        self.callbacks = {}

    def clone(self):
        n = FakePatcher(self.model)
        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup               # 共享（与 model_patcher.clone 一致）
        n.backup_buffers = self.backup_buffers
        n.callbacks = {k: {k1: v1[:] for k1, v1 in v.items()}
                       for k, v in self.callbacks.items()}
        return n

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        obj = self.model
        for part in name.split("."):
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        return obj

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def add_callback(self, call_type, callback):
        self.callbacks.setdefault(call_type, {}).setdefault(None, []).append(callback)

    def fire(self, call_type, *args):
        for c in self.callbacks.get(call_type, {}).values():
            for cb in c:
                cb(self, *args)

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn

    def is_dynamic(self):
        return False


class FakeClip:
    """空格分词的 Anima 形 CLIP：tokenize 按空白计数，encode 返回对应长度。"""

    def tokenize(self, text):
        toks = [[(t, 1.0) for t in text.split()]] if text.strip() else [[]]
        return {"qwen3_06b": toks, "t5xxl": [[(t, 1.0) for t in text.split()]] if text.strip() else [[]]}

    def encode_from_tokens_scheduled(self, tokens):
        n = max(1, len(tokens["qwen3_06b"][0]))
        return [[torch.randn(1, n, 4), {
            "pooled_output": None,
            "t5xxl_ids": torch.arange(n * 2, dtype=torch.int),
            "t5xxl_weights": torch.ones(n * 2),
        }]]


def make_contexts(weights=(1.0, 3.0)):
    """2 位画师：artist0 真长 qwen=3/t5=4 值 4.0；artist1 真长 qwen=2/t5=3 值 -2.0。"""
    t0 = torch.zeros(1, 5, 4)
    t0[:, :3] = 4.0
    ids0 = torch.tensor([10, 11, 12, 13, 0, 0])
    w0 = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    t1 = torch.zeros(1, 5, 4)
    t1[:, :2] = -2.0
    ids1 = torch.tensor([20, 21, 22, 0, 0, 0])
    w1 = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    return {
        "labels": ["@a0", "@a1"],
        "weights": list(weights),
        "conditionings": [
            [[t0, {"t5xxl_ids": ids0, "t5xxl_weights": w0}]],
            [[t1, {"t5xxl_ids": ids1, "t5xxl_weights": w1}]],
        ],
        "token_lengths": [{"qwen": 3, "t5": 4}, {"qwen": 2, "t5": 3}],
        "base_conditioning": [[torch.ones(1, 5, 4), {}]],
        "base_prompt": "test",
    }


def run_wrapper(wrapper, x, base_ctx, cou=(0, 1), sigma=0.8, ns=None):
    to = {"sigmas": torch.tensor([sigma])}
    if cou is not None:
        to["cond_or_uncond"] = list(cou)
    if ns is not None:
        to[core.NS_KEY] = ns
    return wrapper(x, base_ctx, rope_emb=None, transformer_options=to)


X = torch.randn(2, 3, 4)
BASE_CTX = torch.full((2, 1, 4), 2.0)
BASE_OUT = X + 2.0  # FakeAttn 下 base 输出


# ---------------------------------------------------------------------------
# 1) 注册表 / 别名 / 版本
# ---------------------------------------------------------------------------

def test_registry_and_version():
    m = lxa_aat.NODE_CLASS_MAPPINGS
    d = lxa_aat.NODE_DISPLAY_NAME_MAPPINGS
    expect = {"AATArtistPack", "AATParallelArtistMixer", "AATLayerStepScheduler",
              "AATConditioningBlender", "AATStepAlternator", "AATEpsilonMultiGuide"}
    assert set(m) == expect
    assert set(d) == expect
    # v0.1.1 起旧类别名已全部移除
    for legacy in ("ArtistIsolationPack", "ArtistParallelCrossAttn",
                   "ArtistLayerStepRouter", "AATParallelCrossAttn", "AATLayerStepRouter"):
        assert legacy not in m
    # display name 无 L1/L2 代号
    for v in d.values():
        assert "L1" not in v and "L2" not in v
    assert lxa_aat.__version__ == "0.1.2"


# ---------------------------------------------------------------------------
# 2) Pack：解析归一化 + 槽位边界恒等式
# ---------------------------------------------------------------------------

def test_pack_parse_and_boundaries():
    # 解析归一化
    assert nodes_mod.parse_artist_chain("(@wlop:1.1), makoto_shinkai") == \
        [("@wlop", 1.1), ("@makoto_shinkai", 1.0)]
    for raw in ("wlop", "@wlop", "by wlop"):
        assert nodes_mod.normalize_artist_label(raw) == "@wlop"

    pack = nodes_mod.AATArtistPack()
    ctx, base = pack.build(FakeClip(), "1girl, @wlop, standing in a field, @makoto_shinkai",
                           "(@wlop:1.0), (@makoto_shinkai:0.8)")
    assert ctx["labels"] == ["@wlop", "@makoto_shinkai"]
    assert ctx["weights"] == [1.0, 0.8]                      # 权重只进元数据
    assert ctx["clean_base"] == "1girl, standing in a field"  # 槽位移除正确
    # 槽位边界恒等式：P + Z_i + S == L_i（每位画师严格成立）
    b = ctx["boundaries"]
    assert b is not None
    for i, tl in enumerate(ctx["token_lengths"]):
        assert b["qwen"]["prefix"] + b["qwen"]["zone"][i] + b["qwen"]["suffix"] == tl["qwen"]
    # 零 pad 对齐 + pad 区为 0
    max_q = max(t["qwen"] for t in ctx["token_lengths"])
    for c, tl in zip(ctx["conditionings"], ctx["token_lengths"]):
        t = c[0][0]
        assert t.shape[1] == max_q
        if tl["qwen"] < max_q:
            assert torch.all(t[:, tl["qwen"]:] == 0)


# ---------------------------------------------------------------------------
# 3) Mixer：s=0 归零探针 + 融合公式 + CFG mask
# ---------------------------------------------------------------------------

def _mixer(mode, strength, weights=(1.0, 3.0), idx=0, **kw):
    dm = FakeDM()
    st = core.ArtistState(dm, ["@a0", "@a1"], list(weights),
                          make_contexts(weights)["conditionings"],
                          [{"qwen": 3, "t5": 4}, {"qwen": 2, "t5": 3}])
    return core.ParallelArtistCrossAttn(FakeAttn(), st, idx, fusion_mode=mode,
                                        strength=strength, block_range=(0, 2), **kw)


def test_mixer_zero_probe_and_cfg():
    # s=0 归零：输出与 base 逐位一致（两种模式各一次）
    for mode in ("output_avg", "lowrank_delta"):
        w = _mixer(mode, 0.0)
        out = run_wrapper(w, X, BASE_CTX)
        assert torch.equal(out, BASE_OUT)
    # cond 行替换、uncond 行保留；cou=None 全部按 cond
    w = _mixer("output_avg", 1.0)
    out = run_wrapper(w, X, BASE_CTX)
    assert torch.allclose(out[0], X[0] - 0.5, atol=1e-5)     # (0.25*4 + 0.75*(-2)) = -0.5
    assert torch.allclose(out[1], BASE_OUT[1], atol=1e-6)
    out_all = run_wrapper(w, X, BASE_CTX, cou=None)
    assert torch.allclose(out_all[1], X[1] - 0.5, atol=1e-5)


def test_mixer_fusion_formulas():
    # interpolate：delta 堆叠 + RMS 匹配
    w = _mixer("interpolate", 1.0)
    out = run_wrapper(w, X, BASE_CTX)
    # 原始权重 (1,3)：artist_total = 1*(x+4)+3*(x-2) = 4x-2；w_sum=4
    # delta = artist_total − 4·base = −10；out_pre = base − 10；再逐行 RMS 匹配
    rms = lambda t: t.float().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
    artist_total = 1.0 * (X[0:1] + 4.0) + 3.0 * (X[0:1] - 2.0)
    pre = BASE_OUT[0:1] + (artist_total - 4.0 * BASE_OUT[0:1])
    expect = pre * (rms(BASE_OUT[0:1]) / rms(pre))
    assert torch.allclose(out[0:1], expect, atol=1e-4)
    assert torch.allclose(rms(out[0:1]), rms(BASE_OUT[0:1]), atol=1e-4)
    # interpolate_legacy：旧全量式（base*(1-s) + (Σ wᵢ_raw·outᵢ)·s）
    wl = _mixer("interpolate_legacy", 1.0)
    assert torch.allclose(run_wrapper(wl, X, BASE_CTX)[0], 4 * X[0] - 2.0, atol=1e-4)
    # base_preserve：(out − base) ⊥ base 逐 token
    wb = _mixer("base_preserve", 1.0)
    ob = run_wrapper(wb, X, BASE_CTX)
    dots = ((ob[0].float() - BASE_OUT[0].float()) * BASE_OUT[0].float()).sum(dim=-1)
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-3)
    # lowrank_delta k=n：不投影 ≡ 归一化 delta 加权和
    wk = _mixer("lowrank_delta", 1.0, lowrank_k=2, rms_clamp_ratio=0.0)
    dA = torch.full((12,), 2.0)
    dB = torch.full((12,), -4.0)
    expect = BASE_OUT[0] + (0.25 * dA + 0.75 * dB).reshape(3, 4)
    assert torch.allclose(run_wrapper(wk, X, BASE_CTX)[0], expect, atol=1e-4)


def test_svd_projection_math():
    d = torch.randn(4, 50)
    assert torch.equal(core._svd_topk_project(d, 4), d)      # k>=n 直通
    u, s, v = torch.linalg.svd(d, full_matrices=False)
    got = core._svd_topk_project(d, 2)
    gs = torch.linalg.svdvals(got)
    assert torch.allclose(gs[:2], s[:2], rtol=0.1)           # top-k 奇异值近似保持


# ---------------------------------------------------------------------------
# 4) Scheduler：active_set 三态语义 + 协同/独立
# ---------------------------------------------------------------------------

def _l1_wrapper_with_block():
    blk = FakeBlock()
    st = core.ArtistState(FakeDM(), ["@a0", "@a1"], [1.0, 3.0],
                          make_contexts()["conditionings"],
                          [{"qwen": 3, "t5": 4}, {"qwen": 2, "t5": 3}])
    w = core.ParallelArtistCrossAttn(blk.cross_attn, st, 1,
                                     fusion_mode="output_avg", strength=1.0,
                                     block_range=(0, 2))
    blk.cross_attn = w
    return blk


def test_active_set_tristate():
    blk = _l1_wrapper_with_block()
    # 态 1：无 active_set 键 → 默认全开（两位画师按归一化权重融合）
    out = blk(X, None, BASE_CTX, transformer_options={"cond_or_uncond": [0, 1]})
    assert torch.allclose(out[0], X[0] - 0.5, atol=1e-5)

    # 态 2：空集合 → 该层全部画师关闭，只剩 base
    ns = {"active_set": {1: ([], [])}}
    out2 = blk(X, None, BASE_CTX,
               transformer_options={"cond_or_uncond": [0, 1], core.NS_KEY: ns})
    assert torch.allclose(out2, BASE_OUT, atol=1e-6)

    # 态 3：有值 ([1],[1.0]) → 只激活画师1
    ns3 = {"active_set": {1: ([1], [1.0])}}
    out3 = blk(X, None, BASE_CTX,
               transformer_options={"cond_or_uncond": [0, 1], core.NS_KEY: ns3})
    assert torch.allclose(out3[0], X[0] - 2.0, atol=1e-5)


def test_scheduler_standalone_replaces_context_and_copies_opts():
    ast = core.ArtistState(FakeDM(), ["@a0"], [1.0],
                           [[[torch.full((1, 1, 4), 4.0), {}]]],
                           [{"qwen": 1, "t5": 0}])
    ast._cache_ctxs = [torch.full((1, 1, 4), 4.0)]
    ast._cache_key = (str(X.device), X.dtype)
    rst = core.RouterState(ast, {0: [(0, 1.0, (0.0, 1.0))]}, "cosine", 0.1, coop_mode=False)
    w = core.RoutedBlockWrapper(FakeBlock(), rst, 0)
    to = {"cond_or_uncond": [0, 1], "sigmas": torch.tensor([0.8])}
    out = w(X, None, BASE_CTX, transformer_options=to)
    assert torch.allclose(out[0], X[0] + 4.0, atol=1e-5)  # cond 行 context 被替换
    assert torch.allclose(out[1], BASE_OUT[1], atol=1e-6)  # uncond 行保留
    assert "active_set" not in to.get(core.NS_KEY, {})     # t_opts 未被原地污染


def test_boundary_factor():
    assert sch.boundary_factor(0.2, 0.3, 0.6, 0.1, "hard") == 0.0
    assert sch.boundary_factor(0.35, 0.3, 0.6, 0.1, "hard") == 1.0
    assert abs(sch.boundary_factor(0.35, 0.3, 0.6, 0.1, "cosine") - 0.5) < 1e-9
    assert sch.boundary_factor(0.0, 0.0, 0.4, 0.1, "cosine") == 1.0   # 端点放开
    assert sch.boundary_factor(0.65, 0.3, 0.6, 0.1, "cosine") == 0.0  # 区间外


# ---------------------------------------------------------------------------
# 5) Alternator：序数步轮换决策
# ---------------------------------------------------------------------------

def _alt_guider(n_artists=3, **cfg_over):
    cfg = {"mode": "alternate_every", "n_every": 2, "ranges": [],
           "fallback": "base", "final_k": 0, "final_artist": -1,
           "n_artists": n_artists, "debug_logging": False}
    cfg.update(cfg_over)
    g = gd.StepAlternatorGuider(SimpleNamespace(model_options={}), cfg)
    g._sigmas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]  # 5 步
    return g


def _route(g, sigma):
    idx = g._step_index(torch.tensor([sigma]))
    return g._route(idx, len(g._sigmas) - 1)


def test_alternator_routes():
    g = _alt_guider(3)
    seq = [_route(g, s) for s in (1.0, 0.8, 0.6, 0.4, 0.2)]
    assert seq == ["aat_artist_0", "aat_artist_1", "aat_artist_2",
                   "aat_artist_0", "aat_artist_1"]
    g2 = _alt_guider(3, mode="alternate_n", n_every=2)
    seq2 = [_route(g2, s) for s in (1.0, 0.8, 0.6, 0.4, 0.2)]
    assert seq2 == ["aat_artist_0", "aat_artist_0", "aat_artist_1",
                    "aat_artist_1", "aat_artist_2"]
    g3 = _alt_guider(2, mode="custom_ranges", ranges=[(0, 0, 1), (1, 2, -1)])
    assert _route(g3, 1.0) == "aat_artist_0"
    assert _route(g3, 0.6) == "aat_artist_1"
    assert _route(g3, 0.0) == "aat_artist_1"   # end=-1 = 末步
    g4 = _alt_guider(2, mode="custom_ranges", ranges=[(0, 0, 2)], fallback="last")
    g4._last_artist_key = "aat_artist_0"
    assert _route(g4, 0.2) == "aat_artist_0"   # 未覆盖 → 上一位
    g5 = _alt_guider(3, final_k=1, final_artist=1)
    assert _route(g5, 0.2) == "aat_artist_1"   # 最后 K 步固定收尾
    g6 = _alt_guider(0)
    assert _route(g6, 0.6) == "positive"       # 空画师 → 恒 base


# ---------------------------------------------------------------------------
# 6) Epsilon：公式、(a)≡(b)、s=0 标准路径、零系数剔除
# ---------------------------------------------------------------------------

def test_epsilon_guider(monkeypatch):
    import comfy.samplers
    u, eb, ea, ec = (torch.randn(2, 3) for _ in range(4))
    calls = []

    def fake_calc(model, conds, x_in, timestep, model_options):
        calls.append(list(conds))
        return [u, eb, ea, ec]

    monkeypatch.setattr(comfy.samplers, "calc_cond_batch", fake_calc)

    def make(strengths, order):
        g = gd.EpsilonMultiGuideGuider(SimpleNamespace(model_options={}), strengths, order)
        g.inner_model = None
        g.conds = {"positive": "BASE", "negative": "NEG",
                   "aat_artist_0": "A", "aat_artist_1": "B"}
        g.set_cfg(5.0)
        return g

    ga = make([0.7, 0.3], "stack_then_cfg")
    ra = ga.predict_noise(None, torch.tensor([0.5]))
    eps_c = eb + 0.7 * (ea - eb) + 0.3 * (ec - eb)
    assert torch.allclose(ra, u + (eps_c - u) * 5.0, atol=1e-6)
    assert calls[-1] == ["NEG", "BASE", "A", "B"]

    gb = make([0.7, 0.3], "cfg_then_stack")
    rb = gb.predict_noise(None, torch.tensor([0.5]))
    assert (ra - rb).abs().max().item() < 1e-4   # 线性 CFG 下 (a)≡(b)

    # s 全 0 → 标准 2 路公式（batch=[cond, uncond]，out[1] + (out[0] − out[1])·cfg）
    g0 = make([0.0, 0.0], "stack_then_cfg")
    r0 = g0.predict_noise(None, torch.tensor([0.5]))
    assert calls[-1] == ["BASE", "NEG"]
    assert torch.equal(r0, eb + (u - eb) * 5.0)

    # 零系数画师不占 batch 路数
    g1 = make([0.0, 0.5], "stack_then_cfg")
    g1.predict_noise(None, torch.tensor([0.5]))
    assert calls[-1] == ["NEG", "BASE", "B"]


# ---------------------------------------------------------------------------
# 7) Blender：三段混合 + 归零直通 + 元数据
# ---------------------------------------------------------------------------

def test_blender_segments_and_zero_coeff():
    blend = nodes_mod.AATConditioningBlender()
    A = torch.tensor([[[1.0], [10.0], [100.0], [101.0]]])
    B = torch.tensor([[[1.0], [20.0], [200.0], [201.0]]])
    ctx = {
        "labels": ["@a", "@b"], "weights": [1.0, 1.0],
        "conditionings": [
            [[A, {"t5xxl_ids": torch.arange(2), "t5xxl_weights": torch.ones(2)}]],
            [[B, {"t5xxl_ids": torch.arange(2, 4), "t5xxl_weights": torch.ones(2) * 2}]],
        ],
        "token_lengths": [{"qwen": 4, "t5": 2}, {"qwen": 4, "t5": 2}],
        "base_conditioning": [[torch.full((1, 4, 1), 7.0), {}]],
        "boundaries": {"qwen": {"prefix": 1, "suffix": 2, "zone": [1, 1]}, "t5": {}},
    }
    (out,) = blend.blend(ctx, 0.6, False)
    v = out[0][0][0, :, 0].tolist()
    assert v[0] == 1.0                                    # 前缀直通
    assert abs(v[1] - 0.6 * (10 + 20)) < 1e-5             # 画师区加权
    assert abs(v[2] - 0.6 * (100 + 200)) < 1e-4           # 后缀同权混合
    assert abs(v[3] - 0.6 * (101 + 201)) < 1e-4
    assert out[0][1]["t5xxl_ids"].tolist() == [0, 1]      # ids 取权重最高画师一路
    assert torch.allclose(out[0][1]["t5xxl_weights"], torch.ones(2) * 0.6 * 3, atol=1e-5)
    # 混合系数全 0 → 与 base 逐位一致（直通）
    (out0,) = blend.blend(ctx, 0.0, False)
    assert out0 is ctx["base_conditioning"]


# ---------------------------------------------------------------------------
# 8) 卫生机制：备份键重映射 + ON_DETACH 剥离
# ---------------------------------------------------------------------------

def test_sanitize_and_detach_strip():
    pt = SimpleNamespace(
        backup={"diffusion_model.blocks.1.cross_attn.original.q_proj.weight": "W",
                "diffusion_model.blocks.0.cross_attn.original.q_proj.weight": "FOREIGN"},
        backup_buffers={})
    n = core.sanitize_weight_backups(pt, {"diffusion_model.blocks.1.cross_attn"})
    assert n == 1
    assert pt.backup["diffusion_model.blocks.1.cross_attn.q_proj.weight"] == "W"
    assert pt.backup["diffusion_model.blocks.0.cross_attn.original.q_proj.weight"] == "FOREIGN"

    dm = FakeDM()
    foreign = ForeignWrapper(dm.blocks[1].cross_attn)
    ours = core.ParallelArtistCrossAttn(
        foreign, core.ArtistState(dm, ["@a0"], [1.0],
                                  [[[torch.ones(1, 1, 4), {}]]], [{"qwen": 1, "t5": 0}]),
        1, block_range=(0, 2))
    dm.blocks[1].cross_attn = ours
    cb = core.make_detach_strip_callback({"diffusion_model.blocks.1.cross_attn": ours,
                                          "diffusion_model.blocks.0.cross_attn": ours})
    cb(SimpleNamespace(model=SimpleNamespace(diffusion_model=dm)), False)
    # 身份一致处剥离为 foreign；身份不符处（blocks.0 非本实例）不动
    assert dm.blocks[1].cross_attn is foreign
    assert isinstance(dm.blocks[0].cross_attn, FakeAttn)
    cb(SimpleNamespace(model=SimpleNamespace(diffusion_model=dm)), False)  # 幂等
    assert dm.blocks[1].cross_attn is foreign
