# lxa AAT (Anima Artist Tools) v0.1.2

一套给 ComfyUI / Anima 用的多画师节点包，共 6 个节点。它想解决的问题很具体：多位画师写在一起出图时，**强画师总是盖过弱画师**，而且**换个种子效果就变**。

> **先声明：这是完全的 vibe coding 产物**——所有代码都是 AI 写的，人类只负责指方向和最后看效果。用之前请自己掂量。

## 这套节点到底什么效果（说在前面，都是实测）

- **它能做的**：让画师组合变得**听话**——压得住强画师、换种子不大变、能决定谁画哪几步、融合多少可以用旋钮一点点调；
- **它的代价**：**融合越深，画质越糊**。想清晰好看，融合就只能点到为止（出来的图和 base 差不多，带点画师调味）；真把几位画师狠狠揉在一起，就要接受伪影和糊化；
- **它做不到的**：比"直接写画师串"更好看的融合——直写时是大语言模型在理解"这几位画师合在一起是什么"，这个质量目前仍是天花板；另外一张图里两位画师完全各画各的、互不影响，做不到（一张图只有一个构图，这是扩散模型的死规矩）；
- **一个意外发现**：实测发现这个模型 28 层里**只有第 10–14 层在"读"提示词**——画师放在别的层，一点反应都没有。

**一句话建议**：如果某几位画师直接写着就好看，那就直接写，别用这套东西。它是直写出问题时的修理工具。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone <本仓库地址> lxa_aat   # 然后重启 ComfyUI
```

要求：ComfyUI ≥ 0.27.0、PyTorch ≥ 2.0、Python ≥ 3.10，模型用 **Anima**。不装任何第三方依赖、不联网、不动 ComfyUI 本体。

## 该用哪个节点

| 你想要的效果 | 用什么 | 说明 |
|---|---|---|
| 稳，好看，画师味淡一点就行 | Mixer 的 `output_avg` / `base_preserve` / `lowrank_delta` | 构图基本和 base 一样，差别在细节 |
| 明显能看出融合了，愿意糊一点 | Mixer `interpolate` → Epsilon（强度总和≤1）→ Blender（慎入） | 越往后画质税越重 |
| 谁画哪几步、一人占几成步数 | Step Alternator | 效果大多数时候和 Epsilon 接近（见下文） |
| 连续地调"风格浓度" | Epsilon Multi-Guide | 算力开销约翻倍 |
| 指定谁管哪几层、哪个时段 | Layer-Step Scheduler | **同一层别放多位画师，会崩** |

## 六个节点

### 1. AAT Artist Pack (Encode) —— 编码包

所有其他节点都要先接它。它把你写的完整提示词拆开：一份"去掉画师"的底料，再加每位画师单独一份编码。

| 参数 | 干什么用 |
|---|---|
| `clip` | Anima 的文本编码器（如 `qwen_3_06b_base.safetensors`） |
| `base_prompt` | 完整提示词，画师按 `@画师名` 写在里面 |
| `artist_chain` | 画师名单：`(@名字:权重)`，逗号或换行分隔；不写权重就是 1.0 |

### 2. AAT Parallel Artist Mixer —— 并行混合

让每位画师各自完整地画一遍，再把结果按权重混在一起。好处是互相不抢，坏处是混得越狠画质越糊。

| 参数 | 默认 | 干什么用 |
|---|---|---|
| `fusion_mode` | `output_avg` | 混合方式：`output_avg`=平均，最稳；`interpolate`=加强度，可调但有伪影；`lowrank_delta`=先给画师增量去噪再混，**画面出伪影时换它**；`base_preserve`=只要风格不动构图；`interpolate_legacy`=旧版，别用 |
| `strength` | 1.0 | 总强度，0=完全没效果（和 base 一样） |
| `start_block` / `end_block` | 0 / -1 | 管哪几层（-1=最后一层） |
| `lowrank_k` | 1 | lowrank 模式保留几个主方向 |
| `rms_clamp_ratio` | 1.2 | lowrank 模式的保险丝，别调小 |
| `apply_to_uncond` | False | 要不要也作用在负面提示上 |
| `ema_alpha` / `static_capture_k` | 0 | 跨步平滑/冻结，性能微调，一般用默认 |

### 3. AAT Layer-Step Scheduler —— 排班表

用一小段文字指定"哪位画师管哪几层、哪个时段"。每行一条：

```
0::1.0@0-13              # 0号画师（名单第1位）：管 0-13 层，全程
1::0.8@14-27,step=0-0.6  # 1号画师：管 14-27 层，只在前 60% 步
```

| 参数 | 默认 | 干什么用 |
|---|---|---|
| `layer_config` | 空 | 就是上面这种写法；实测**画风主要由中层（10-14 层左右）决定**，浅层管构图、深层管细节 |
| `transition_fn` / `transition_width` | cosine / 0.1 | 交界处要不要平滑过渡 |
| `apply_to_uncond` | False | 同上 |

两个提醒：**同一层别分配给多位画师**（会严重崩图）；和 Mixer 一起用时顺序必须是 **Mixer → Scheduler**，接反了 Mixer 会悄悄失效（节点会报警）。

### 4. AAT Conditioning Blender —— 条件混合（实验性）

在送进模型之前，把几位画师的"菜谱"直接揉成一条。实测下来它**融合得最狠，但也糊得最狠**——多数情况不好看，建议开 `renorm`、压低系数、放低期待，当实验品玩。

| 参数 | 默认 | 干什么用 |
|---|---|---|
| `blend_coeff` | 0.6 | 混合力度；所有画师加起来的总和最好别超过 1 |
| `renorm` | False | 把混完的强度拉回正常值，**建议常开** |

### 5. AAT Step Alternator (Guider) —— 步数接力

按步数换人：这一步 A 画，下一步 B 画（或按你定的区间一人跑一棒）。每一棒都是正常采样，所以稳；但最终效果是"A 起稿、B 在上面接着画"的接力融合，不是各画一半。

**实测注意：它的出图效果大多数时候和 Epsilon Multi-Guide 很接近**，而不是和 Mixer 接近——想想也合理：它和 Epsilon 都是引导器，每一步模型都在正常干活，只是一个靠"换人"、一个靠"调方向"。

要接 **SamplerCustomAdvanced**（不是普通 KSampler），负面提示和 cfg 改由它管。

| 参数 | 默认 | 干什么用 |
|---|---|---|
| `mode` | `alternate_every` | `alternate_every`=一步一步换；`alternate_n`=每 N 步换；`custom_ranges`=自己定区间 |
| `n_every` | 2 | alternate_n 模式下几步换一次 |
| `fallback` | `base` | 没被区间覆盖的步：`base`=不带画师 / `last`=接着上一位 |
| `final_k` / `final_artist` | 0 / -1 | 最后 K 步固定让某位收尾（防止结尾抖） |
| `cfg` / `negative` | 5.0 | CFG 值和负面提示 |
| `debug_logging` | False | 打开后控制台能看到每步是谁在画 |

画师取 Pack 名单前 3 位。注意：用它的时候 Scheduler 就别设 `step=` 了，两个"排时间的"会打架。

### 6. AAT Epsilon Multi-Guide —— ε 引导

每一步在模型的输出端把"画内容"和"画内容+某画师"两个方向一减，得到纯风格方向，然后同时往几位画师各偏一点。数学上和 CFG 是同一套（所以名字里带 ε），模型内部完全不受影响。**大部分情况下效果和 Step Alternator 接近**，区别是它给的是连续旋钮（风格浓度可以一点点调），Alternator 给的是步数份额（谁占几成步数）。

**安全线：所有画师的强度加起来别超过 1**，超了画质会明显劣化。另外单画师强度=1 时它就等于直写，所以单画师没必要开它。

| 参数 | 默认 | 干什么用 |
|---|---|---|
| `s_a` … `s_h` | 0.5/0.5/其余0 | 8 位画师各自的强度（-4~4）；0 等于不花钱；负数是"去掉这位画师味"；名单不满 8 位时，多出来的栏位自动无视 |
| `cfg` | 5.0 | CFG 值 |
| `cfg_order` | stack_then_cfg | 和 CFG 结合的顺序，两种数学上等价，用默认 |

## 和其他插件一起用

顺序：**Anima-Artist-Mixer（AAM）在前，本套件在后**。本套件自带内存清理机制（ComfyUI 0.27 的动态加载有两个已知坑，都处理了）；AAM 这类插件没做清理，如果你发现"明明没用画师插件，画风却残留下来"，那是对方的问题，不是这套的。

## 已知限制

- 它消除的是"画师互相抢"和"渗透"，不保证任何组合都 100% 好——这是大语言模型非线性决定的天花板；
- 开销：Pack 每位画师多一次编码；Epsilon 每步多算 N 路；Mixer 生效层每步每位画师多一次前向；
- `step=` 按 FLOW 采样（sigma 0~1）设计，换别的采样类型请先自己验证。

## 测试

仓库里有离线测试（不需要模型和显卡）：

```bash
python -m pytest custom_nodes/lxa_aat/tests/ -q   # 12 passed
```

## 许可

MIT（见 `LICENSE`）。其中 `lowrank_delta` 模式的 SVD 投影移植自 Anima-Artist-Mixer（MIT，© 2026 An1X3R & 汐浮尘，见 `NOTICE`），其余都是原创。

> 从老版本 `artist_isolation` 升级来的：v0.1.1 起旧节点名已经删了，旧工作流打不开的，把节点换成新名字（`AATArtistPack` / `AATParallelArtistMixer` / `AATLayerStepScheduler`）就行。
