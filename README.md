# lxa AAT (Anima Artist Tools) v0.1.2 — Anima 多画师语义隔离节点套件

在采样过程中实现多画师（`@xxx`）语义的相互隔离，解决两个痛点：

1. **LLM 上下文渗透**：把多个画师名写进同一次文本编码时，编码器内部注意力让语义互相污染（`base, @A, @B` 直写时强势画师主导、构图被改写）。
2. **Cross-Attention softmax 竞争**：多个画师条件在同一 softmax 分母中竞争，输出"回归均值"。

套件提供 **6 个节点类**，覆盖四种隔离路线，可独立使用、可叠加使用且不互相冲突：

| 节点 | display name | 路线 | 一句话定位 |
|---|---|---|---|
| `AATArtistPack` | AAT Artist Pack (Encode) | 编码 | 独立编码 base 与每位画师，杜绝渗透（全部路线的前提） |
| `AATParallelArtistMixer` | AAT Parallel Artist Mixer | 输出空间融合 | 每位画师独立 K/V 前向，特征空间加权融合 |
| `AATLayerStepScheduler` | AAT Layer-Step Scheduler | 层×时间路由 | 把画师按 DiT block 索引与采样进度路由生效 |
| `AATConditioningBlender` | AAT Conditioning Blender | 条件空间混合 | 三段（前缀/画师区/后缀）混合成一条 CONDITIONING |
| `AATStepAlternator` | AAT Step Alternator (Guider) | 序数步接力 | 自定义 GUIDER，按步序轮换画师条件（CFG 内置） |
| `AATEpsilonMultiGuide` | AAT Epsilon Multi-Guide | 输出端 ε 引导 | 每步在网络输出端按 ε = base + Σ sᵢΔᵢ 合成 |

适用模型：**Anima**（MiniTrainDIT 结构，28 层，投影命名 `q_proj/k_proj/v_proj/output_proj`；FLOW 采样 sigma∈[0,1]、timestep==sigma）。其他同构 DiT 理论可用（节点会校验 `blocks[i].cross_attn` 存在，不满足时按层跳过并告警）。

> **从 artist_isolation 迁移**：本套件前身为 `artist_isolation`。**v0.1.1 起旧类别名已全部移除**——含旧类名（`ArtistIsolationPack` / `ArtistParallelCrossAttn` / `ArtistLayerStepRouter` / `AATParallelCrossAttn` / `AATLayerStepRouter`）的工作流加载时会得到"未知节点"提示，需手动把节点类名替换为对应新类名（`AATArtistPack` / `AATParallelArtistMixer` / `AATLayerStepScheduler`）后重存。日志前缀为 `[lxa_aat]`。

## Installation / 安装

```bash
cd ComfyUI/custom_nodes
git clone <本仓库地址> lxa_aat
# 重启 ComfyUI
```

**环境要求**：ComfyUI ≥ 0.27.0（在 0.27.0、含动态 VRAM 加载器的环境上实测通过；更低版本未验证）；PyTorch ≥ 2.0；Python ≥ 3.10。目标模型为 Anima（MiniTrainDIT 结构）。

- **外部依赖**：无。仅使用 PyTorch 与 ComfyUI 自带模块；**不引入任何第三方包**。
- **网络/requests**：不需要。全部计算在本地完成，不发起任何网络请求。
- **不修改 ComfyUI 核心代码**：全部通过官方扩展点（`add_object_patch` / `model_options` / `add_callback`）工作。
- `NOTICE`：lowrank_delta 的 SVD 投影移植自 Anima-Artist-Mixer（MIT License, © 2026 An1X3R & 汐浮尘）。

## 选型建议（什么场景用哪个节点）

| 场景 | 首选 | 备选/组合 | 理由 |
|---|---|---|---|
| 多位画师**同时融合**于一张图（要"混合画风"） | **AAT Parallel Artist Mixer** | 轻量替代：**Conditioning Blender**（条件空间混合，零 patch 零额外算力，更温和） | Mixer 在输出空间隔离 softmax 竞争，隔离最彻底、算力最高（生效层每步每位画师一次 K/V 前向，全部激活等长时合并为一次 batched 调用） |
| 不同画师**分管层范围或采样时段**（"A 管构图、B 管笔触"） | **AAT Layer-Step Scheduler** | + Mixer（协同模式，调度只写激活集合） | 层维 × 时间维路由；与 Mixer 串联时 Mixer 必须在上游 |
| 逐画师**整段接力**（前段 A、后段 B） | **AAT Step Alternator** | — | 序数步轮换条件，CFG 内置；接 SamplerCustomAdvanced |
| 要**连续刻度**调风格浓度、不动模型内部 | **AAT Epsilon Multi-Guide** | — | 输出端 ε 堆叠，系数即刻度；代价是 N+2 路计算 |
| 只要**快速、便宜**的双画师糅合 | **AAT Conditioning Blender** | — | 单条 CONDITIONING，算力与常图相同 |

**性能基线**：Mixer = 生效层每步 N 路 K/V 前向（batched 合并后约 1 次调用）；Epsilon = 每步 (N+2)/2 倍于标准 CFG；Blender/Alternator = 与常图相同。

---

## 1. AATArtistPack — 画师编码包（全部路线的前提）

纯编码节点，**不做任何 model patch**。`base_prompt` 接收**包含画师标签的完整提示词**（通常来自上游文本拼接节点）：节点逐字匹配并移除所有画师标签得到 `clean_base`（作为 base_conditioning 的编码文本，绝不含画师标签），再在最左侧被移除标签的**画师槽位**处分别插入 `@画师i`，逐位独立编码——绝不允许把多个画师名写进同一次编码。

### 使用方法

```
文本拼接节点(完整提示词) ──→ AATArtistPack.base_prompt
画师串 (@A, @B)          ──→ AATArtistPack.artist_chain
CLIPLoader               ──→ AATArtistPack.clip
输出: artist_contexts ──→ 任何下游 AAT 节点；base_conditioning ──→ KSampler.positive
```

配置步骤：CLIPLoader 选 `qwen_3_06b_base.safetensors`（type `stable_diffusion`）→ 两路文本接好 → 下游任选。**两串需保持一致**（`artist_chain` 里声明的标签应能在 `base_prompt` 中逐字找到；找不到会 WARNING 并回退"末尾追加"旧行为）。

可直接抄改的示例：

```
base_prompt = 1girl, standing in a flower field, looking at viewer, upper body, @wlop, @makoto_shinkai
artist_chain = (@wlop:1.0), (@makoto_shinkai:1.0)
```

### 参数说明

| 参数 | 类型 | 默认 | 作用与预期 |
|---|---|---|---|
| `clip` | CLIP | — | Anima 文本编码器 |
| `base_prompt` | STRING 多行 | `""` | 含画师标签的完整提示词。画师标签会被逐字移除并记录槽位 |
| `artist_chain` | STRING 多行 | `""` | 逗号/换行分隔；权重写法 `(@画师名:1.1)`，缺省 1.0；`wlop`/`@wlop`/`by wlop` 均归一化为 `@wlop`；`::权重` 写法无效（告警后按 1.0）。**权重只存元数据，不进编码文本** |

**槽位规则**：按标签长度降序逐字匹配（防 `@hal` 误伤 `@hal aluha`）；每标签移除全部出现位置，最左移除点即槽位；清理残留（连续逗号、逗号前后空格、句首句尾逗号、双空格）。三段边界（前缀/画师区/后缀的 qwen+t5 token 长度）由 Pack 分词导出为 `boundaries` 字段（Blender 依赖；BPE 边界 ±1 token 误差用真长反推校正）。

**输出契约**：`labels` / `weights`（原始值）/ `conditionings`（零 pad 对齐 + `token_lengths` 真长记录）/ `base_conditioning` / `clean_base` / `boundaries`。

**观察点（终测目检用）**：
- 控制台 `画师槽位 @ 字符 N；clean_base=...` 一行：`clean_base` 是否正是你想要的"无画师场景文本"；画师标签是否被完整移除、没有半个标签残留；
- `画师 @xxx: weight=..., qwen_len=..., t5_len=...`：各画师长度是否符合标签长短的预期；
- `边界: qwen prefix=... suffix=... zone=[...]`：zone 数列与画师名长度是否对应（长名字 zone 应更大）；
- 若出现 `未匹配到任何画师标签` 的 WARNING：两串不一致，检查 base_prompt 里标签写法（必须带 `@` 且逐字一致）。

---

## 2. AATParallelArtistMixer — 并行混合（输出空间融合）

每位画师独立编码、独立 K/V 前向，输出在特征空间加权融合——各画师在各自独立的 softmax 中归一化，互不竞争。base 分支始终计算、始终保留；只 patch `diffusion_model.blocks.{i}.cross_attn`；先 `model.clone()` 再 patch。

### 使用方法

```
UNETLoader ──→ AATParallelArtistMixer ──→ KSampler.model
Pack.artist_contexts ──→ Mixer.artist_contexts
Pack.base_conditioning ──→ KSampler.positive
```

可抄改示例：`fusion_mode=output_avg, strength=0.8, start_block=0, end_block=-1`（全层生效）；要"去噪取主风格"改 `fusion_mode=lowrank_delta, lowrank_k=1`。

### 参数说明

| 参数 | 默认 | 作用 / 调大调小预期 / 联动 |
|---|---|---|
| `fusion_mode` | `output_avg` | 见下表 |
| `strength` | 1.0 | 融合强度 s（0~4）。s=0 → 输出与纯 base 逐位一致；s 越大画师影响越强；s>1 进入外推（放大）。与权重、调度过渡系数**三级相乘**（最终 ∝ s × Pack权重 × 过渡系数） |
| `start_block` / `end_block` | 0 / -1 | 生效层闭区间（-1 = 末层）。收窄可只让画师作用于浅层（构图）或深层（笔触） |
| `apply_to_uncond` | False | False = 只作用 cond 行（推荐）；True = 负条件也注入 |
| `ema_alpha` | 0.0 | 跨步 EMA（0 关）：调大 → 跨步更平滑、风格更稳但细节响应更钝 |
| `static_capture_k` | 0 | 前 K 步平均后冻结（0 关）：调大 → 后期更稳，早期定型错误也会被冻结 |
| `lowrank_k` | 1 | 仅 lowrank_delta：SVD 子空间维数。调大 → 保留更多画师细节方向（也更接近普通加权和）；调小 → 更"只留主风格" |
| `lowrank_normalize_weights` | True | 仅 lowrank_delta：权重归一化 wᵢ/Σ\|wᵢ\|（关掉后权重按原始值直接放大） |
| `rms_clamp_ratio` | 1.2 | 仅 lowrank_delta：RMS 钳制阈值（输出 RMS 超 base × 该值钳回；0 = 关）。调大 → 更少干预、允许更浓；调小 → 更严的幅度封顶 |

**`fusion_mode` 选型**：

| 模式 | 语义 | 适用 |
|---|---|---|
| `output_avg` | 归一化凸混合（调和） | 默认起点，多位画师均衡混合 |
| `interpolate` | delta 堆叠 + 逐行 RMS match（加强度） | 要"更浓"且要刻度；权重不归一、s>1 外推。Σw=1 时与 output_avg 公式恒等 |
| `lowrank_delta` | SVD top-k 投影 delta 堆叠 + RMS clamp（去噪取主风格方向） | 伪影区首选；希望滤掉画师细节的零碎噪声只留主方向 |
| `base_preserve` | 差值垂直投影（保构图） | 构图必须保住、只要风格方向 |
| `interpolate_legacy` | 旧全量式 lerp（兼容） | 旧工作流 |

**观察点（终测目检用）**：
- s 扫描（0.5/1.0/1.5）下风格浓度是否**单调**可预期（不出现 1.0 反而比 0.5 淡）；
- 同一 s 下换模式：lowrank_delta 与 interpolate 相比，零碎伪影/杂点是否更少（伪影区重点看）；
- s≥1.5 是否过饱和/崩；出现则先降 s 或开 `rms_clamp_ratio`；
- 负条件也受影响时检查是否误开了 `apply_to_uncond`。

---

## 3. AATLayerStepScheduler — 层-步调度（层×时间路由）

把画师按 DiT block 索引 × 采样进度（sigma 区间）路由到不同层范围/步区间生效。patch 点仅为 `diffusion_model.blocks.{i}` 整体 forward；两种模式：**独立模式**（无 Mixer 时，替换 cond 行 context 为激活画师的加权混合 conditioning）；**协同模式**（上游有 Mixer 时，只写激活集合到 `active_set`，由 Mixer 读取）。

### 使用方法

```
独立: UNETLoader ──→ AATLayerStepScheduler ──→ KSampler.model
协同: UNETLoader ──→ AATParallelArtistMixer ──→ AATLayerStepScheduler ──→ KSampler.model
```

**串联顺序硬性要求：Mixer → Scheduler**（Mixer 在上游）。接反时 Mixer 的 cross_attn wrapper 会被 set_attr 到 Scheduler wrapper 实例上（空挂），Mixer 完全失效且不报错——本节点会在检测到目标 block 已被整体包装时打印告警。

配置方式：**纯文本 `layer_config`**（v0.1.2 起槽位 UI 已移除，文本域是唯一配置路径）。文本框 tooltip 与本节均给出示例；默认值保持为空。

可抄改示例（文本域）：

```
0::1.0@0-13            # 画师0 全程作用于浅层
1::0.8@14-27,step=0-0.6  # 画师1 只在采样前 60% 作用于深层，权重 0.8
```

> **兼容性说明**：v0.1 / v0.1.1 保存的**槽位版**工作流中，槽位配置自 v0.1.2 起不再生效（节点只剩文本域），请把原槽位配置按上方格式改写为文本格式后重存。

### 参数说明

| 参数 | 默认 | 作用 / 预期 / 联动 |
|---|---|---|
| `layer_config` | `""` | 每行 `画师序号::权重@层范围,step=lo-hi`（step 可省=全程）。层/步均闭区间；进度 0=高噪、1=结束；**后声明优先覆盖** |
| `transition_fn` | `cosine` | 过渡函数：cosine（边界导数为零，最平滑）/ linear / hard |
| `transition_width` | 0.1 | 过渡区宽度（进度单位，作用于区间内侧；0 = 硬切换）。调大 → 画师进出更渐、边界更模糊 |
| `apply_to_uncond` | False | 同 Mixer |

**`active_set` 三态语义（配 layer_config 必读）**：

| 状态 | 含义 | Mixer 在该层的行为 |
|---|---|---|
| 无该 block 的键 | 该层不在 layer_map 中 | **默认全部画师激活**（不是关掉！） |
| 空集合 | 在 layer_map 中但当前步无条目生效 | 该层全部画师关闭，只剩 base |
| 有值 | 当前步激活画师及有效权重 | 仅计算激活分支，权重 = Pack权重 × 声明权重 × 过渡系数 |

**观察点（终测目检用）**：
- 日志 `协同/独立 模式，N 层路由（共 M 条声明）`：N/M 是否与你配置的预期一致；
- 浅层画师 vs 深层画师的生效部位是否符合分工预期（如"A 管构图 B 管笔触"看构图与笔触是否各归其主）；
- step 切换点前后是否有突兀跳变；有则调大 `transition_width`；

---

## 4. AATConditioningBlender — 条件空间混合器

把多位画师的 conditioning 直接混合成**一条**普通 CONDITIONING 直供 KSampler positive。前缀（槽位之前，各画师逐位相同）直通；画师区按槽位对齐、短者零 pad 加权求和；后缀同权加权求和（禁止单路直通）。混合后只剩一组融合 token，**softmax 竞争从结构上消失**。不打任何 patch、不留任何状态。

### 使用方法

```
Pack.artist_contexts ──→ AATConditioningBlender ──→ KSampler.positive
UNETLoader（不经过任何 patch 节点）──→ KSampler.model
```

可抄改示例：`blend_coeff=0.6, renorm=False`（双画师 0.6/0.6，允许 Σ>1）；要更温和开 `renorm=True`。

### 参数说明

| 参数 | 默认 | 作用 / 预期 / 联动 |
|---|---|---|
| `blend_coeff` | 0.6 | 节点系数：最终混合系数 = Pack权重 × 节点系数。调大 → 整体画师浓度升高（允许 Σ>1）；调小 → 更接近纯 base |
| `renorm` | False | 开 = 混合系数归一化到 Σ=1（Σ=0 时自动跳过，行为不变）。要"温和、不放大"时开 |

**优雅降级**：空画师 → WARNING + 直通 `base_conditioning`；单画师 → WARNING + 直通该画师；缺 `boundaries`（旧版 Pack 输出）→ WARNING + 直通 base；系数全 0 → 直通 base（此时输出与 base **逐位一致**）。

**元数据处置**：`t5xxl_ids` 取权重最高画师一路；数值型 `t5xxl_weights` 同权混合；输出按新真长截齐（`aat_qwen_len`）。

**观察点（终测目检用）**：
- 日志 `N 位画师混合完成 L_new=... (P=.. Z_max=.. S=..)`：L_new 与 Pack 边界日志是否自洽；
- 与直写 `"base, @A, @B"` 相比，同 seed 下构图是否更稳定（本节点设计目标）；
- 双画师是否都有体现，还是一方被压——被压时先检查权重比例与 `renorm`；
- `blend_coeff` 升降时整体浓淡是否平滑单调。

---

## 5. AATStepAlternator — 步序轮换（Guider，接力）

按**序数步**轮换画师条件的自定义 GUIDER（序数步控制在普通节点求值时拿不到总步数，必须在 guider 内实现）。内部完整保留 CFGGuider 的 cond/uncond 行为（**不会**退化为 cfg=1），只在每步把 positive 指到本步激活的条件键；画师条件直接用 Pack 已编码的 `base+@画师i`（零额外编码，进 guider 前按 `token_lengths` 切回真长）。

### 使用方法

```
UNETLoader ──→ AATStepAlternator.model
Pack.artist_contexts ──→ Alternator.artist_contexts（取前 3 位，超出 WARNING）
负条件 ──→ Alternator.negative
RandomNoise ──→ SamplerCustomAdvanced.noise
Alternator.GUIDER ──→ SamplerCustomAdvanced.guider
KSamplerSelect ──→ .sampler；BasicScheduler ──→ .sigmas；EmptyLatentImage ──→ .latent_image
```

可抄改示例：`mode=alternate_every`（A→B→C→A…）；要分块 `mode=alternate_n, n_every=2`；要自定份额 `mode=custom_ranges` 配 `range1: 画师0@0-10, range2: 画师1@11-(-1)`；末期防抖 `final_k=3, final_artist=-1`。

**互斥约定（重要）**：用本 Guider 时，**层-步调度节点只做层路由、不写 `step=`**（时间维调度不叠加——两边同时按时间切换会互相打架）；**层路由（层维）可以共存**（先 Mixer/层路由 patch，再把 MODEL 接给本 Guider）。

### 参数说明

| 参数 | 默认 | 作用 / 预期 / 联动 |
|---|---|---|
| `cfg` | 5.0 | CFG 值（guider 内部执行，与 KSampler 的 cfg 语义相同） |
| `mode` | `alternate_every` | 轮换模式（见上） |
| `n_every` | 2 | alternate_n：每 N 步换画师。调大 → 每位画师驻留更久、切换更缓 |
| `fallback` | `base` | custom_ranges 未覆盖的步：`base` = 无画师注入；`last` = 延续上一位 |
| `final_k` | 0 | 最后 K 步固定画师收尾（防末期细节抖动）；0 = 关 |
| `final_artist` | -1 | 收尾画师序号；-1 = 上一位激活画师 |
| `debug_logging` | False | 开 = 逐步打印当前激活画师（调边界配置时用） |
| `rangeN_*` | 禁用 | custom_ranges 槽位：画师序号 + 起止步（end=-1 = 末步）；后声明优先；越界跳过 + WARNING |

**观察点（终测目检用）**：
- 开 `debug_logging` 后日志步序是否严格等于你的区间配置（边界步重点核对）；
- 份额配置（如 7:3 vs 3:7）与出图风格占比是否对应；
- `n_every` 很小时风格是否碎（交替过快的典型症状）；碎则调大 n 或上 `final_k`；
- 与层-步调度同用时，确认调度侧没有写 `step=`（互斥约定）。

---

## 6. AATEpsilonMultiGuide — 输出端 ε 引导

每步在**网络输出端**按公式合成（模型内部零接触）：

```
ε = ε_base + Σ sᵢ·(εᵢ − ε_base)   （i = A…H，最多 8 位画师）
```

cond batch 塞 `[uncond, base, base+@A, …(, base+@H)]`，一次前向后合成。内部完成 CFG；接线同 Step Alternator（SamplerCustomAdvanced）。

### 使用方法

与第 5 节相同的 Guider 接线，把节点换成 `AATEpsilonMultiGuide`。可抄改示例：双画师 `s_a=0.6, s_b=0.6, 其余 0.0`；四画师把 `s_c / s_d` 提到 0.5 起步；画师取 Pack 前 8 位，**栏位多不代表全开——s=0 的画师不计费**。

### 参数说明

| 参数 | 默认 | 作用 / 预期 / 联动 |
|---|---|---|
| `cfg` | 5.0 | CFG 值 |
| `s_a` … `s_h` | 0.5 / 0.5 / 0×6 | 8 位画师的强度系数（-4~4）。0 = 该画师不占 batch 路数；s>1 外推放大；负值 = 反向（"去画师化"方向）。画师取 Pack 前 8 位；**Pack 不足 8 位时超出数量的栏位静默无视**（不告警不生效），Pack 超过 8 位 → 取前 8 + WARNING。与 Pack 权重无联动（权重只作用于 Mixer 路线） |
| `cfg_order` | `stack_then_cfg` | CFG 结合次序：(a) 先 delta 堆叠再做 CFG（默认）；(b) 各条件先各自 CFG 再堆叠。**标准线性 CFG 下两者代数恒等**（推导见下），保留枚举供未来非线性 guidance |

**CFG 次序推导**：(a) `u + c·[(ε_b − u) + Σ sᵢ(εᵢ − ε_b)]`；(b) `[u + c(ε_b − u)] + Σ sᵢ[(u + c(εᵢ − u)) − (u + c(ε_b − u))]` = 展开后同 (a)。若上游经 `model_options` 注入了自定义 `sampler_cfg_function`，本节点按标准公式执行、会绕过该钩子。

**性能基线**：条件路数 2 → N+2（uncond + base + N 位非零系数画师），每步计算约 (N+2)/2 倍（batch 单前向），**8 画师满配约为标准 5 倍**——注意**栏位多不代表全开：s=0 的画师不计费**（不占 batch 路数）；全部系数为 0 时走标准 2 路（零开销，且输出与纯 base 逐位一致）。

**观察点（终测目检用）**：
- 系数扫描（0.5/1.0/1.5）下风格浓度刻度是否平滑、可预期；
- 某画师系数为负时的"去除"方向是否符合预期；
- 与 Mixer(interpolate) 同强度下的体感差异（输出端堆叠 vs 层内融合）；
- 三画师时注意显存与时间（N+2 路）。

---

## 与其他插件共存

- **Anima-Artist-Mixer (AAM)**：**本套件应串在 AAM 之后**（AAM → lxa AAT）。AAM 的解包只解自己的 wrapper 类且直接读裸模型属性、不看 `object_patches`；把它放在本套件之后（上游）会绕过并覆盖本套件的 patch 条目。本套件 patch 前经 `get_model_object` 取当前链顶模块（含对方 wrapper）整体包入 `.original`，对方逻辑在我们的 base 分支中照常生效。
- **IPAdapter 类（同样 patch cross_attn 的插件）**：只要对方也遵循"get_model_object 取链顶 + 包当前模块"的约定，任意顺序都可组合。
- **VRAM 脏键与残留说明**：ComfyUI 0.27 的动态 VRAM 加载器会把"wrapper 内含 original 子模块"产生的参数路径写进跨 clone 共享的权重备份 dict；wrapper 卸载后，干净工作流在恢复备份时会 `AttributeError: 'Attention' object has no attribute 'original'`。此外，`load_models_gpu` 对同模型 clone 执行 `detach(unpatch_all=False)` 时不会还原 object_patch 属性，wrapper 会残留在共享模块上并被后续干净 patcher 的运行继承。**本套件对两者均已自洁**：权重备份键重映射（patch 时 / ON_PRE_RUN / ON_CLEANUP 三处）+ **ON_DETACH 按身份剥离本 clone 的 wrapper**（实测「节点→干净→节点」连跑干净输出与外部基准逐位一致）。但 **AAM 及任何同类第三方 wrapper 插件理论上仍会触发上述两类问题**——遇到 AttributeError 或"没用插件的画师效果却残留"时，根因在此，不是 ComfyUI 坏了，也不是本套件引起的。

## 已知限制

- **LLM 编码器非线性天花板**：独立编码保证"编码阶段零渗透"，但 Anima 的 `llm_adapter` 与 DiT 都是高度非线性的——本套件**做不到也不承诺 SDXL 式"无损混合"的天花板**；它消除的是 softmax 竞争导致的回归均值与编码渗透，不是让任意画师组合都 100% 保真。
- **计算成本**：每画师一次独立 TE 编码（Pack）；Mixer 在生效层每步每位激活画师一次额外 K/V 前向；Epsilon 为 N+2 路。画师越多越贵，建议配合调度收敛生效范围。
- **进度换算**针对 FLOW 采样（sigma∈[0,1] 降序）设计；非 FLOW 模型请自行验证 `step=` 区间含义。
- 任一 wrapper forward 抛异常时该层永久短路到原始模块并打印带层号的日志（不会因单层问题中断出图）。

## 终测说明（用户侧）

过程验收只做过流程/恒等探针/机制数值/锚点归因指标；**全部视觉类结论留待你用真实工作流统一终测**（本套件不附专门测试工作流——自动测试的写法与真实用法差距大，中途目检价值有限）。终测时各节点"该看什么"已列在上方每个节点章节的**观察点**清单中；发现与预期不符的条目，按条目回报即可定位回炉。P1/P2 期间被降级为"待目检"的历史条目（双画师可辨度、s 高档位不崩坏等）一并纳入本次终测。

## Testing / 测试说明

仓库附带离线单元测试（fake DiT / fake ModelPatcher / fake CLIP harness，无模型、无 GPU、无本机路径依赖），覆盖：注册表/版本、s=0 归零探针、`active_set` 三态语义、Pack 槽位边界恒等式（P+Z+S==L）、全部融合模式公式（含 SVD 投影数学）、guider 轮换决策与 ε 公式、批次组装、备份键重映射与 ON_DETACH 剥离。运行方式：

```bash
python -m pytest custom_nodes/lxa_aat/tests/ -q
```

（pytest 为开发侧工具，非运行时依赖；在 ComfyUI 仓库根或 custom_nodes 目录下运行均可。）

## 文件结构

```
custom_nodes/lxa_aat/
├── __init__.py            # 节点注册（新类名 + 永久别名）
├── nodes.py               # 六个节点类
├── core.py                # wrapper、解包、融合、CFG mask、备份键卫生、SVD 投影
├── scheduler.py           # 层范围/步区间解析、sigma 换算、过渡函数
├── guider.py              # StepAlternator / EpsilonMultiGuide 自定义 guider
├── pyproject.toml         # 包元数据（v0.1.2）
├── NOTICE                 # 第三方许可（Anima-Artist-Mixer, MIT）
├── README.md
└── workflows/example.json # 示例工作流（每节点至少一条可加载分支）
```

`workflows/example.json` 在一张画布上包含共享加载器与六条分支（A. Mixer / B. Scheduler 独立 / C. Mixer→Scheduler 协同（文本域配置） / D. Blender / E. Step Alternator / F. Epsilon Multi-Guide），各分支独立采样与 SaveImage（前缀 `aat_ex_*`）。默认模型文件：`anima-base-v1.0.safetensors`（unet）、`qwen_3_06b_base.safetensors`（clip）、`qwen_image_vae.safetensors`（vae）——请按本机实际文件名调整。不需要的分支整体右键 Mute 即可。
