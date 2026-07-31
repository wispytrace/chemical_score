# Chemical Score

一个可解释、可扩展的化学反应规则评分引擎。返回结构为：

完整的维度、叶子指标、公式和 JSON 数据结构见
[SCORING_STRUCTURE.md](SCORING_STRUCTURE.md)。

```text
综合评分
├── 可行性
│   ├── 组成守恒
│   ├── 反应一致性
│   ├── 结构与转化
│   └── 选择性与变化幅度
├── 证据支持度（配置历史反应库后启用）
│   ├── 反应空间相似性
│   ├── 反应先例
│   └── 历史结果
├── 安全性
│   ├── 结构风险
│   └── 反应性状态
└── 经济性
    ├── 物料效率
    └── 合成策略
```

每个节点都有 `score`、配置权重、归一化后的 `effective_weight` 和对父节点的
`contribution`。叶子节点另外返回原始值、证据、警告、状态和耗时，因此可以从总分
逐级追踪到具体规则。

> 这是规则启发式筛查工具，不是实验成功率、预测产率、SDS 或完整工艺安全评价。

## 安装

```bash
pip install -e ".[api,dev]"
```

## Python 接口

```python
from chemical_score import evaluate_reaction

result = evaluate_reaction(
    reaction_smiles="CC(=O)O.CCO>>CCOC(C)=O",
)

print(result["score"])
print(result["score_tree"])
```

也可以传分离字段，以免把试剂计入原子经济性：

```python
result = evaluate_reaction(
    reactants_smiles="CC(=O)O.CCO",
    agents_smiles="O=S(Cl)Cl",
    product_smiles="CCOC(C)=O",
)
```

批量接口：

```python
from chemical_score import evaluate_reactions

results = evaluate_reactions(
    [
        {"reaction_smiles": "CC(=O)O.CCO>>CCOC(C)=O"},
        {"reactants_smiles": "CCBr.CN", "product_smiles": "CCNC"},
    ],
    concurrency=2,
)
```

### 证据支持度

可以用 JSON 或 JSONL 历史反应库启用证据支持度。每条记录支持以下字段：

```json
{
  "id": "ord-123",
  "reaction_smiles": "CC(=O)O.CCO>>CCOC(C)=O",
  "yield_percent": 82.0,
  "success": true,
  "source": "ORD",
  "metadata": {"split": "train"}
}
```

Python 中显式构造评分器：

```python
from chemical_score import EvidenceIndex, ReactionEvaluator, evaluate_reaction

evidence = EvidenceIndex.from_file(
    "data/reactions.jsonl",
    similarity_threshold=0.5,
    top_k=5,
    strict=False,
)
evaluator = ReactionEvaluator(evidence_index=evidence)
result = evaluate_reaction(
    reaction_smiles="CC(=O)O.CCO>>CCOC(C)=O",
    evaluator=evaluator,
)
```

五个叶子分数分别为：

- `nearest_reaction_similarity`：最近历史反应的 RDKit 差分指纹相似度；
- `local_precedent_density`：相似度阈值以上的局部先例数量；
- `exact_reaction_precedent`：规范化后完全相同的历史反应数量；
- `mapped_transformation_precedent`：原子映射反应中相同成键/断键/键级变化的数量；
- `historical_outcome_support`：相似先例的收率或成功标签加权结果。

索引在加载时预计算指纹和键变换，查询用批量 Tanimoto 相似度和 Top-K 选择；五个
叶子指标在一次评价中复用同一查询结果。没有证据库时整个维度为
`not_applicable`，不改变原三维评分；配置后一级权重为可行性 48%、证据支持度 20%、
安全性 16%、经济性 16%。

自定义指标只需实现一个包含 `spec` 和 `evaluate(context)` 的无状态对象，再注册到
`MetricRegistry`，无需修改聚合器或 HTTP 层。

## HTTP API

```bash
python app.py
```

也可以通过 Uvicorn 启动：

```bash
uvicorn app:app --host 0.0.0.0 --port 9528
```

HTTP 服务可通过环境变量加载证据库：

```bash
CHEMICAL_SCORE_EVIDENCE_PATH=data/reactions.jsonl \
  uvicorn app:app --host 0.0.0.0 --port 9528
```

PowerShell：

```powershell
$env:CHEMICAL_SCORE_EVIDENCE_PATH = "data/reactions.jsonl"
python app.py
```

评分：

```bash
curl -X POST http://127.0.0.1:9528/v1/evaluations \
  -H "Content-Type: application/json" \
  -d '{"reaction_smiles":"CC(=O)O.CCO>>CCOC(C)=O"}'
```

HTTP 评分接口只需要一个必填字段 `reaction_smiles`。无试剂时使用
`反应物>>产物`，有单独试剂时使用 `反应物>试剂>产物`。

接口列表：

- `GET /health`
- `GET /v1/metrics`：返回完整指标分类和权重
- `GET /v1/evidence/status`：返回证据库大小、拒绝记录数和近邻参数
- `POST /v1/evaluations`：评价单个反应
- `POST /v1/evaluations/batch`：一次评价最多 100 个反应
- `GET /docs`：FastAPI 自动生成的交互文档

批量 HTTP 请求可增加 `"concurrency": 4`（范围 1–16）；返回顺序与输入顺序一致。

## 评分约定和局限

- 所有节点使用 0–100 分；父节点对当前可评估的子节点做加权平均。
- `not_applicable` 不会被当成 0 分；执行错误会明确标记为 `error` 并降低 `coverage`。
- 证据支持度只表示“被所提供语料支持的程度”，强烈依赖语料覆盖率、重复记录和标签
  质量；低分也可能只是反应新颖，不能作为不可行的硬否决。
- 历史结果若同时混用收率与成功布尔标签，接口会返回警告；正式部署应统一标签定义，
  去重并按数据来源、时间和质量做外部校准。
- 恒等反应、核心元素无来源等关键错误会触发透明的总分上限，记录在 `flags` 和根节点
  `evidence` 中。
- 原子经济性没有当量、收率、溶剂和后处理数据，只是基于反应 SMILES 的估计。
- 官能团和选择性规则目前覆盖有限；若要做某一反应域的可靠排序，应使用带实验标签的
  数据集校准叶子分数和权重，并增加正向反应模型或产率模型。
