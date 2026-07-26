# lxa AAT (Anima Artist Tools) v0.1.2

ComfyUI / Anima 多画师语义隔离节点套件——6 个节点，解决多画师混用时的**强画师压制**与**种子抽签不稳定**。

> **Vibe Coding 声明**：本项目为**完全的 vibe coding 产物**——全部代码由 AI 编写，人类仅做方向性指导与最终结果测试。使用时请自行评估风险。

## 这套节点实际能做什么（先说实话）

经真实工作流终测，本套件的**实际效果**如下：

- ✅ **能做到**：给画师组合提供**稳定、可控、有刻度**的合作方式——压得住强画师、跨 seed 更稳、能按步数/层段排班、融合强度连续可调；
- ⚠️ **代价**：融合强度与画质代价**正相关**——越保持 base 构图越清晰好看（归一化融合≈base+轻调味），越深度融合画质税越重（伪影/糊化）；
- ❌ **做不到**：比直写更好看的融合（多画师直写时，融合由大语言模型在语言空间完成，质量仍是上限）；单图内两位画师完全隔离互不沾染（扩散单轨迹单构图的硬约束）；
- 🔬 **副产品**：实测发现该模型的文本条件仅由第 **10–14 层**（共 28 层）读取——画师注入放在其他层完全没有效果。

**第一建议**：某组画师直写就好看 → 直接用直写，别上插件。插件是"直写不听话时的控制工具"。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone <本仓库地址> lxa_aat   # 重启 ComfyUI
```

要求：ComfyUI ≥ 0.27.0、PyTorch ≥ 2.0、Python ≥ 3.10；目标模型 **Anima**（MiniTrainDIT 结构，28 层）。零第三方依赖、零网络请求、不修改 ComfyUI 核心。

## 选型速查

| 你想要的 | 用哪个 | 代价/备注 |
|---|---|---|
| 稳定出片，多画师轻调味（最好看最稳） | Mixer 的 `output_avg` / `base_preserve` / `lowrank_delta`，或 Step Alternator | 构图风格与 base 接近，差异在细节 |
| 可辨的融合风格 | Mixer `interpolate` → Epsilon（Σs≤1）→ Blender（实验性） | 按此顺序画质税递增 |
| A 起稿 B 收尾 / 按步数分份额 | Step Alternator | 配 SamplerCustomAdvanced |
| 按层段×时间段排班 | Layer-Step Scheduler | **层段勿重叠**（同层多画师会崩） |
| 连续刻度调风格浓度 | Epsilon Multi-Guide | 每步 (N+2)/2 倍算力 |

## 节点与参数

### 1. AAT Artist Pack (Encode) —— 编码包（所有路线的前提）

把含画师标签的完整提示词拆成"无画师底料"+ 每位画师一份独立编码（画师串位置符合官方规范）。

| 参数 | 作用 |
|---|---|
| `clip` | Anima 文本编码器（如 `qwen_3_06b_base.safetensors`） |
| `base_prompt` | 含 `@画师` 的完整提示词（标签会被移除并记录槽位） |
| `artist_chain` | 画师串：`(@名字:权重)`，逗号/换行分隔；不写权重=1.0；权重只记账不进编码 |

### 2. AAT Parallel Artist Mixer —— 并行混合（特征空间融合）

每位画师独立 K/V 前向，输出加权融合，各画师互不竞争。

| 参数 | 默认 | 作用 |
|---|---|---|
| `fusion_mode` | `output_avg` | `output_avg`=调和（稳）；`interpolate`=加强度有刻度（有伪影代价）；`lowrank_delta`=SVD 去噪取主风格（伪影区首选）；`base_preserve`=保构图；`interpolate_legacy`=旧版勿用 |
| `strength` | 1.0 | 总强度（0~4），与 Pack 权重相乘；s=0 与纯 base 一致 |
| `start_block` / `end_block` | 0 / -1 | 生效层范围（-1=末层） |
| `apply_to_uncond` | False | 是否也注入负条件 |
| `lowrank_k` | 1 | lowrank 模式保留的主方向数 |
| `rms_clamp_ratio` | 1.2 | lowrank 模式幅度保护阈值（0=关） |
| `ema_alpha` / `static_capture_k` | 0 | 跨步平滑/前 K 步冻结（性能与稳定性微调） |

### 3. AAT Layer-Step Scheduler —— 层-步调度

按文本配置把画师路由到指定层范围和采样时段。文本域每行一条：`画师序号::权重@起层-止层,step=lo-hi`（step 可省=全程；序号 0 起）。

```
0::1.0@0-13              # 画师0 全程作用于浅层
1::0.8@14-27,step=0-0.6  # 画师1 前 60% 步作用于深层
```

| 参数 | 默认 | 作用 |
|---|---|---|
| `layer_config` | 空 | 上述文本配置；实测**画风主控在中层（约 10-14 层）**，浅层管构图、深层管细节；**层段勿重叠** |
| `transition_fn` / `transition_width` | cosine / 0.1 | 边界过渡函数与宽度（0=硬切换） |
| `apply_to_uncond` | False | 同 Mixer |

串联顺序（与 Mixer 同用时）：**Mixer → Scheduler**，接反 Mixer 会静默失效（节点会告警）。

### 4. AAT Conditioning Blender —— 条件混合（实验性）

进模型前把多位画师的条件向量揉成一条。**实测融合最强但画质代价最大**（后缀混合产生语义混浊），需开 renorm、压低系数、放低审美预期。

| 参数 | 默认 | 作用 |
|---|---|---|
| `blend_coeff` | 0.6 | 节点系数（与 Pack 权重相乘）；建议最终系数总和 ≤1 |
| `renorm` | False | 混合系数归一化到 Σ=1；**建议常开** |

### 5. AAT Step Alternator (Guider) —— 步序接力

按步数轮换画师（每步都是标准采样，稳定；本质是"先后接力融合"：前者定构图、后者做表面）。接 **SamplerCustomAdvanced**，内部完成 CFG。

| 参数 | 默认 | 作用 |
|---|---|---|
| `mode` | `alternate_every` | `alternate_every`=逐步轮换；`alternate_n`=每 N 步换；`custom_ranges`=自定义区间（3 行槽位：画师+起止步） |
| `n_every` | 2 | alternate_n 的步数 |
| `fallback` | `base` | 区间未覆盖的步：base=无画师 / last=延续上一位 |
| `final_k` / `final_artist` | 0 / -1 | 最后 K 步固定某位收尾（防抖） |
| `cfg` / `negative` | 5.0 | CFG 值与负条件（由本节点接管） |
| `debug_logging` | False | 逐步打印激活画师 |

**互斥**：用本节点时 Scheduler 只做层路由、不写 `step=`。画师取 Pack 前 3 位。

### 6. AAT Epsilon Multi-Guide —— ε 空间引导

每步在网络输出端合成方向：`ε = ε_base + Σ sᵢ·(εᵢ − ε_base)`（CFG 同款数学，模型内部零接触）。接线同 Alternator。**画质安全区：所有画师 s 之和 ≤1**；单画师 s=1.0 等价于直写。

| 参数 | 默认 | 作用 |
|---|---|---|
| `s_a` … `s_h` | 0.5/0.5/0×6 | 8 位画师强度（-4~4）；0=不占算力；负值=去画师化；Pack 不足 8 位时空栏位静默无视 |
| `cfg` | 5.0 | CFG 值 |
| `cfg_order` | stack_then_cfg | 与 CFG 的结合次序（两者代数恒等，默认即可） |

## 与其他插件共存

串在 **Anima-Artist-Mixer（AAM）之后**（AAM → lxa AAT）。本套件已内置 VRAM 脏键清理与跨运行 wrapper 剥离（ComfyUI 0.27 动态加载器的两类已知问题）；AAM 等同类插件未做此清理，如遇"没用插件画师效果却残留"，根因在对方不在本套件。

## 已知限制

- LLM 非线性天花板：消除的是竞争与渗透，不承诺任意组合 100% 保真；
- 成本：Pack 每画师一次独立编码；Epsilon 每步 N+2 路；Mixer 生效层每步每画师一次 K/V 前向；
- `step=` 区间按 FLOW 采样（sigma∈[0,1]）设计，非 FLOW 模型请自行验证。

## 测试

仓库附离线单元测试（fake 环境，无模型无 GPU）：

```bash
python -m pytest custom_nodes/lxa_aat/tests/ -q   # 12 passed
```

## 许可

MIT（见 `LICENSE`）。`lowrank_delta` 模式的 SVD 投影移植自 Anima-Artist-Mixer（MIT, © 2026 An1X3R & 汐浮尘，见 `NOTICE`），其余为原创。

> 从 `artist_isolation` 迁移：v0.1.1 起旧类别名已移除，旧工作流需手动替换节点类名为 `AATArtistPack` / `AATParallelArtistMixer` / `AATLayerStepScheduler`。日志前缀 `[lxa_aat]`。
