# Pharma Target Agent

一个面向药物化学场景的靶点 lead 迭代优化工具。

给定靶点名称和一个起始 `SMILES` 后，系统会：

- 检索 PubMed 和本地药化规则库
- 让大模型在尽量保留核心 scaffold 的前提下生成候选分子
- 用 `SwissADME + RDKit + ProTox3` 做多参数评分
- 支持自动迭代和手动逐轮优化
- 显示起始分子、当前最优分子、候选池、执行链路和文献引用

当前项目目录位于 `C:\Users\Lenovo\Documents\Playground`。

## 核心能力

- 多轮 lead optimization，而不是单次静态打分
- 多目标优化，而不是只看一个综合分
- 同时考虑 ADME、药化风险、毒性和可开发性
- 自动模式可持续迭代到目标达成或触发停止条件
- 手动模式可由用户从候选池中指定下一轮种子分子
- 对外部服务做了缓存、节流、重试和回退处理

## 当前评分体系

系统现在不是“只按综合分选最优”，而是一个多目标约束框架。

前端可以配置 5 个核心目标参数：

1. `目标综合分`
2. `目标风险分上限`
3. `目标 ProTox3 分`
4. `目标 Bioavailability`
5. `目标毒性等级下限`

这些参数的作用如下。

### 1. 目标综合分

这是总目标分，表示系统希望分子在整体多参数空间里达到的水平。

它的作用：

- 作为全局质量指标，综合考虑 ADME、药化风险、QED 和毒性修正
- 在候选排序时仍然占主要权重
- 在自动模式下，如果最优分子满足所有目标并且综合分达标，可触发停止

它适合回答的问题是：

- “这个分子整体上是不是比起始分子更像一个可继续推进的候选物？”

注意：

- 综合分高，不代表毒性一定低
- 综合分高，也不代表 Bioavailability 一定足够
- 所以它必须和其他目标一起看

### 2. 目标风险分上限

风险分是一个“越低越好”的指标。

当前风险分主要受这些因素影响：

- Lipinski / Veber 等规则违规
- PAINS / Brenk alerts
- P-gp substrate 风险
- CYP 抑制负担
- LogP 偏高
- 合成可及性偏差
- GI absorption 不佳
- ProTox3 毒性风险修正

它的作用：

- 避免系统只追求高综合分，却选出一个明显更“脏”的分子
- 强行约束候选必须在药化风险上可接受
- 让自动优化更偏向“可推进分子”而不是“纸面高分分子”

它适合回答的问题是：

- “这个分子是否伴随过高的药化或安全性负担？”

一般理解：

- 综合分解决“整体好不好”
- 风险分解决“代价大不大”

### 3. 目标 ProTox3 分

这是系统基于 ProTox3 输出整理出的毒性档案分，越高越好。

它不是 ProTox3 官网原生单一分数，而是本项目基于以下信息整合出来的：

- `Predicted Toxicity Class`
- `Predicted LD50`
- 预测准确率
- 相似性百分比
- Active toxicity models 数量
- Organ toxicity active 数量
- Toxicity target hit 数量

它的作用：

- 把离散毒性信号整合成一个更方便做排序和阈值控制的目标
- 防止系统在 ADME 很好时忽略明显的毒性信号
- 在候选排序时作为重要辅助信号

它适合回答的问题是：

- “从 ProTox3 的角度，这个分子的整体毒性档案是不是足够干净？”

一般理解：

- `目标 ProTox3 分` 是“连续值约束”，便于排序
- `目标毒性等级下限` 是“硬门槛约束”，便于快速筛掉明显高毒分子

### 4. 目标 Bioavailability

这里指的是 SwissADME 的 `Bioavailability Score`，越高越好。

它的作用：

- 约束分子不要只在理论结构上好看，却口服开发性偏弱
- 帮助系统更偏向“可能有较好体内暴露”的分子
- 在综合分相近时，优先选择更有口服推进价值的候选

它适合回答的问题是：

- “这个分子是否更有机会成为一个可口服开发的候选？”

注意：

- 它不是 PK 实验结果
- 它是药化早期的经验型开发性指标
- 但在早期筛选里很有价值

### 5. 目标毒性等级下限

这里对应 ProTox3 的 `Predicted Toxicity Class`。

在 ProTox3 里，通常是：

- `Class 1` 更毒
- `Class 6` 更安全

所以这个目标是“越高越好”的下限约束。

它的作用：

- 作为一个非常直观的安全性门槛
- 用于快速排除急性毒性预警明显的分子
- 比连续的 ProTox3 分更适合作为硬阈值

它适合回答的问题是：

- “这个分子的急性毒性等级至少有没有到我能接受的水平？”

## 这 5 个目标怎么一起工作

可以这样理解：

- `目标综合分`
  - 控总体验收线
- `目标风险分上限`
  - 控药化和开发风险别太高
- `目标 ProTox3 分`
  - 控整体毒性档案
- `目标 Bioavailability`
  - 控口服开发性
- `目标毒性等级下限`
  - 控最基础的急性毒性门槛

在当前实现中，它们会同时影响：

- 候选排序
- 最优分子判断
- 自动模式停止条件
- 目标达成卡片展示

排序逻辑不是单纯按综合分，而是更接近：

1. 先看是否满足全部已启用目标
2. 再看满足了多少个目标
3. 再看综合分
4. 再看风险分
5. 再看 ProTox3 分和相似性

这意味着：

- 一个综合分稍低，但更安全、更满足目标的分子，可能会排在前面
- 这比“只追求高分”更符合药化真实优化逻辑

## 推荐设置思路

### 偏探索

适合早期发散：

- 目标综合分：`78-84`
- 目标风险分上限：`45-55`
- 目标 ProTox3 分：`50-60`
- 目标 Bioavailability：`0.40-0.55`
- 目标毒性等级下限：`3-4`

特点：

- 放宽限制
- 先找到方向
- 不要过早把候选池卡死

### 偏收敛

适合筛选更干净的候选：

- 目标综合分：`84-90`
- 目标风险分上限：`30-42`
- 目标 ProTox3 分：`58-72`
- 目标 Bioavailability：`0.55-0.70`
- 目标毒性等级下限：`4-5`

特点：

- 更像“往 pre-candidate 靠拢”的设置
- 速度会更慢
- 候选更容易因为条件太严而难以全部达标

## 项目结构

```text
Playground/
├─ README.md
├─ requirements.txt
├─ start_app.ps1
├─ .env
├─ .env.example
└─ pharma_agent/
   ├─ config.py
   ├─ agent/
   │  ├─ core.py
   │  ├─ memory.py
   │  └─ tools.py
   ├─ data/
   │  ├─ drug_rules.txt
   │  ├─ swissadme_cache.json
   │  ├─ protox3_cache.json
   │  └─ faiss_index/
   │     ├─ index.faiss
   │     └─ metadata.json
   ├─ mol/
   │  ├─ evaluator.py
   │  ├─ swissadme_client.py
   │  └─ protox3_client.py
   ├─ rag/
   │  ├─ build_index.py
   │  ├─ embeddings.py
   │  └─ retriever.py
   └─ ui/
      └─ app.py
```

## 环境要求

- Python `3.10` 或 `3.11`
- 不建议直接使用较新的未充分验证版本
- 需要可以访问：
  - PubMed
  - SwissADME
  - ProTox3
  - 你的候选生成 API

## 安装

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

## `.env` 配置

最少需要配置候选生成 API：

```env
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

其他可选配置：

```env
PUBMED_EMAIL=
PUBMED_TOOL=pharma-agent-demo
LOCAL_EMBEDDING_MODE=hash

SWISSADME_TIMEOUT=90
SWISSADME_CACHE_ENABLED=true
SWISSADME_DELAY_SECONDS=3.0

PROTOX3_TIMEOUT=120
PROTOX3_CACHE_ENABLED=true
PROTOX3_DELAY_SECONDS=6.0
PROTOX3_REFINE_TOP_N=2
PROTOX3_MAX_RETRIES=3
PROTOX3_RETRY_BACKOFF_SECONDS=8.0
```

### 参数说明

- `SWISSADME_DELAY_SECONDS`
  - SwissADME 请求节流时间
- `PROTOX3_DELAY_SECONDS`
  - ProTox3 请求节流时间
- `PROTOX3_REFINE_TOP_N`
  - 每轮只对基础预筛靠前的若干候选做 ProTox3 精排
- `PROTOX3_MAX_RETRIES`
  - ProTox3 单次请求失败时的最大重试次数
- `PROTOX3_RETRY_BACKOFF_SECONDS`
  - ProTox3 重试退避基线

## 构建本地 RAG 索引

```powershell
python -m pharma_agent.rag.build_index
```

## 启动 UI

```powershell
.\start_app.ps1
```

或：

```powershell
python -m streamlit run .\pharma_agent\ui\app.py
```

## 使用流程

1. 输入靶点名称
2. 输入起始 lead 分子 `SMILES`
3. 选择优化模式
4. 设定目标综合分、风险分、毒性分、Bioavailability 等约束
5. 设置最大迭代轮数和每轮候选数
6. 点击 `启动 Agent 分析`

结果页会展示：

- 起始分子和最优分子的结构对比
- 综合评分卡
- ProTox3 毒性卡
- 目标约束达成情况
- 候选池
- 迭代历史曲线
- ProTox3 模型明细
- 文献与规则引用
- 执行链路

## 自动模式与手动模式

### 自动迭代

- 系统自动生成候选
- 自动评估候选
- 自动选择下一轮 seed
- 满足目标或达到最大轮数后停止

适合：

- 快速跑通一个靶点
- 看系统能否自行收敛

### 手动逐轮

- 每轮给出候选池
- 用户自己挑选下一轮种子分子
- 全局最优分子单独维护

适合：

- 人工控制 SAR 方向
- 某些候选虽然总分略低，但化学直觉更合理时

## 评分逻辑概览

### 1. SwissADME

主要使用：

- GI absorption
- Bioavailability score
- Lipinski / Veber / Ghose / Egan / Muegge
- PAINS / Brenk / Leadlikeness
- Synthetic accessibility
- CYP 抑制和 P-gp 信息

### 2. RDKit

主要使用：

- 结构合法性检查
- 理化性质补充
- Bemis-Murcko scaffold
- 指纹相似性
- QED
- 分子结构图渲染

### 3. ProTox3

主要使用：

- Predicted Toxicity Class
- Predicted LD50
- toxicity model active 数
- organ toxicity active 数
- toxicity target binding 情况

### 4. 综合评分

系统内部大致分为：

- `base_overall_score`
  - 不含 ProTox3 的基础分
- `risk_score`
  - 风险分，越低越好
- `toxicity_score`
  - 整合后的 ProTox3 毒性分，越高越好
- `overall_score`
  - 最终综合分

## ProTox3 稳定性策略

因为 ProTox3 容易触发风控或短暂失效，项目做了这些保护：

- 本地缓存
- 请求节流
- 请求重试
- 退避等待
- 批量评估时只对前若干候选精排
- 如果一轮毒性精排全部失败，会对关键候选做一次保底重试
- 即使 ProTox3 不可用，系统也会回退到非毒性精排流程，而不是整轮崩掉

## 已知限制

- 评分是早期药化筛选逻辑，不是实验结论
- ProTox3 和 SwissADME 都是外部服务，稳定性受网络和站点状态影响
- 大模型生成的候选仍可能需要人工筛选
- 高分不代表真实活性强，也不代表真实体内 PK 或毒性一定理想
