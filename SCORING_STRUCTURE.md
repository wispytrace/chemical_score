# Chemical Score 评分结构与数据规范

本文档对应 `chemical-score 0.4.0`、结果模式 `schema_version = 1.1`。

## 1. 适用范围

系统只需要一个反应 SMILES：

```text
反应物>>产物
反应物>试剂/催化剂>产物
```

评分完全在本地执行，不调用大模型。主要依据 RDKit 分子解析、结构描述符、SMARTS
规则、可选原子映射和可选历史反应证据库。总分是可解释的筛查分，不是实验成功率、
预测收率、SDS 或工艺安全结论。

## 2. 评分树

```text
综合评分 overall
├── 可行性 feasibility（配置权重 0.48）
│   ├── 组成守恒 conservation（0.30）
│   ├── 反应一致性 consistency（0.20）
│   ├── 结构与转化 structure（0.30）
│   └── 选择性与变化幅度 selectivity（0.20）
├── 证据支持度 evidence_support（0.20，可选）
│   ├── 反应空间相似性 similarity（0.50）
│   ├── 反应先例 precedent（0.30）
│   └── 历史结果 outcomes（0.20）
├── 安全性 safety（0.16）
│   ├── 结构风险 structural_hazards（0.60）
│   └── 反应性状态 reactive_state（0.40）
└── 经济性 economy（0.16）
    ├── 物料效率 material_efficiency（0.60）
    └── 合成策略 synthesis_strategy（0.40）
```

没有配置证据库时，`evidence_support` 为 `not_applicable`，其有效权重为 0；其余三个
维度自动归一化为可行性 0.60、安全性 0.20、经济性 0.20。

## 3. 聚合规则

每个可评估子节点的有效权重为：

```text
effective_weight_i = configured_weight_i / Σ available_configured_weights
contribution_i     = score_i × effective_weight_i
parent_score       = Σ contribution_i
```

- 所有分数限制在 0–100。
- `not_applicable` 表示输入信息不足或规则不适用，不当作 0 分。
- `error` 表示指标执行失败，从父节点加权中排除，并降低执行覆盖率。
- 恒等反应总分上限为 20。
- 主产物 C/N/O 或关键杂元素无反应物来源时，总分上限为 35。

## 4. 叶子指标

### 4.1 可行性 / 组成守恒

| 指标 ID | 权重 | 含义 |
|---|---:|---|
| `core_element_conservation` | 2.0 | 检查主产物 C/N/O 是否能由反应物提供。 |
| `key_element_conservation` | 1.4 | 检查 F/Cl/Br/I/P/S/B/Si 等关键元素来源。 |
| `product_atom_traceability` | 1.2 | 按全部重元素计算主产物原子可追溯比例。 |

`product_atom_traceability` 的基础值为：

```text
Σ_element min(产物该元素数, 反应物该元素数) / 产物重原子总数
```

### 4.2 可行性 / 反应一致性

| 指标 ID | 权重 | 含义 |
|---|---:|---|
| `identity_check` | 2.0 | 判断产物是否与某个反应物完全相同。 |
| `meaningful_change` | 1.1 | 识别高相似主前体加微小碎片形成的伪转化。 |
| `fragmentation_and_size` | 0.8 | 检查组分过多或反应物/产物尺寸比异常。 |

### 4.3 可行性 / 结构与转化

| 指标 ID | 权重 | 含义 | 不适用条件 |
|---|---:|---|---|
| `structural_continuity` | 1.4 | Morgan 指纹衡量主反应物与主产物结构连续性。 | 无 |
| `scaffold_continuity` | 1.0 | 比较 Bemis–Murcko 骨架是否相同或存在包含关系。 | 一侧没有环骨架 |
| `functional_group_plausibility` | 1.2 | 识别常见官能团变化并检查前体支持。 | 未命中规则 |
| `leaving_group_support` | 0.5 | 对新增酯、酰胺、醚检查常见活化/离去基。 | 非对应成键场景 |
| `mapped_bond_change_complexity` | 0.8 | 统计映射反应中的成键、断键和键级变化。 | 无映射、重复映射或映射覆盖不足 |

当前通用官能团规则包括：

- 酯形成、酰胺形成、醚形成；
- 芳基卤化物/硼酸类偶联前体；
- 醇氧化、醛酮还原；
- 烯烃和炔烃消耗；
- 酯水解、酰胺水解；
- 卤代烃被胺或含氧亲核体取代。

映射键变化数量为 1–3 时视为常见单步变化；超过 5 个时逐步降低分数并发出复杂重排
或多步合并警告。系统不会为了这个指标自动运行原子映射模型。

### 4.4 可行性 / 选择性与变化幅度

| 指标 ID | 权重 | 含义 | 不适用条件 |
|---|---:|---|---|
| `ring_topology_change` | 0.6 | 对单步中极端芳香环数量变化温和惩罚。 | 无 |
| `stereochemistry_change` | 0.5 | 比较反应两侧手性中心数量。 | 两侧均无手性中心 |
| `descriptor_change` | 0.7 | 比较环、杂原子、柔性、HBA/HBD 等描述符。 | 无 |
| `chemoselectivity_risk` | 1.0 | 检查游离胺与酯化反应的特定竞争风险。 | 非胺/酯竞争场景 |
| `reactive_site_competition` | 0.9 | 统计常见成键反应中多余亲核/亲电候选位点。 | 未识别成键场景或缺少一类位点 |

位点竞争基础惩罚为每个额外候选位点扣 18 分，最低为 0。它只代表选择性歧义，不能
依据 reaction SMILES 判断催化剂、溶剂或保护策略是否已经解决该问题。

### 4.5 证据支持度

| 指标 ID | 权重 | 含义 |
|---|---:|---|
| `nearest_reaction_similarity` | 1.2 | RDKit 反应差分指纹的最高 Tanimoto 相似度。 |
| `local_precedent_density` | 0.8 | 相似度阈值以上的历史先例数量。 |
| `exact_reaction_precedent` | 0.8 | 忽略组分顺序和映射编号后的精确先例数量。 |
| `mapped_transformation_precedent` | 1.2 | 相同通用成键/断键/键级变化的先例数量。 |
| `historical_outcome_support` | 1.0 | 相似先例收率或成功标签的相似度加权结果。 |

证据低分表示当前语料支持不足，也可能只是反应新颖，不能作为不可行的硬否决。

### 4.6 安全性

| 指标 ID | 权重 | 含义 |
|---|---:|---|
| `structural_alerts` | 1.5 | 匹配过氧键、叠氮、长杂原子链等结构警报。 |
| `radical_state` | 1.2 | 检查反应物侧额外自由基电子。 |
| `charge_balance` | 0.8 | 比较反应物与主产物形式电荷。 |

试剂字段中的组分参加结构安全筛查，但不参加元素守恒和物料效率。温度、压力、浓度、
加料速度未知，因此安全性只能作为结构筛查。

### 4.7 经济性 / 物料效率

| 指标 ID | 权重 | 含义 |
|---|---:|---|
| `atom_economy_estimate` | 1.2 | 主产物分子量 / 反应物总分子量。 |
| `carbon_efficiency` | 1.0 | 主产物碳数 / 反应物总碳数。 |
| `heavy_atom_efficiency` | 0.8 | 可追溯产物重原子 / 反应物总重原子。 |

这些指标没有化学计量数、收率、溶剂和后处理数据，只是由 reaction SMILES 推导的物料
保留估计。

### 4.8 经济性 / 合成策略

| 指标 ID | 权重 | 含义 |
|---|---:|---|
| `product_synthetic_accessibility` | 0.8 | 将产物 RDKit SA Score 映射为易合成分数。 |
| `synthetic_accessibility_change` | 0.6 | 产物相对最难前体的单步合成复杂度增益。 |
| `protecting_group_burden` | 0.7 | 识别常见保护基引入或脱除。 |

SA Score 约为 1（易）到 10（难）：

```text
product_synthetic_accessibility = clamp((10 - product_sa) / 9 × 100)
complexity_delta = product_sa - max(reactant_sa_scores)
```

两个 SA 指标复用相同缓存。SA Score 是分子层面的启发式，不代表具体反应成功率。

## 5. 覆盖率与输入质量

顶层保留兼容字段 `coverage`，它等于 `execution_coverage`。新增：

```json
{
  "coverage_details": {
    "metric_counts": {
      "total": 30,
      "evaluated": 21,
      "not_applicable": 9,
      "error": 0
    },
    "execution_coverage": 1.0,
    "applicability_coverage": 0.7,
    "core_applicability_coverage": 0.84,
    "evidence_applicability_coverage": 0.0,
    "evaluated_dimensions": ["feasibility", "safety", "economy"]
  }
}
```

- `execution_coverage`：没有执行错误的指标比例。
- `applicability_coverage`：全部非错误指标中实际产生分数的比例。
- `core_applicability_coverage`：排除可选证据模块后的适用比例。
- `evidence_applicability_coverage`：证据模块自身的适用比例。

反应对象同时返回输入质量：

```json
{
  "reaction": {
    "input_quality": {
      "atom_mapping": {
        "present": false,
        "reactant_coverage": 0.0,
        "product_coverage": 0.0,
        "traceable_product_fraction": 0.0,
        "duplicate_map_numbers": [],
        "element_mismatches": [],
        "product_maps_missing_from_reactants": []
      },
      "reactant_component_count": 2,
      "product_component_count": 1,
      "agents_separated": false
    }
  }
}
```

没有原子映射是正常输入，不产生惩罚；只有映射相关叶子为 `not_applicable`。

## 6. API 数据结构

### 6.1 单反应请求

```http
POST /v1/evaluations
Content-Type: application/json
```

```json
{
  "reaction_smiles": "CC(=O)O.CCO>>CCOC(C)=O"
}
```

### 6.2 顶层响应

```json
{
  "schema_version": "1.1",
  "engine_version": "0.4.0",
  "status": "success",
  "score": 92.6111,
  "coverage": 1.0,
  "coverage_details": {},
  "reaction": {},
  "flags": [],
  "warnings": [],
  "errors": [],
  "duration_ms": 2.0,
  "score_tree": {}
}
```

`status` 可为：

- `success`：输入有效且所有指标均正常执行；
- `partial_success`：至少一个指标执行异常，其他结果仍返回；
- `invalid_input`：reaction SMILES 无法完整解析。

### 6.3 ScoreNode

总分、维度、分组和叶子统一使用以下结构：

```json
{
  "id": "product_synthetic_accessibility",
  "name": "产物合成可及性",
  "type": "metric",
  "score": 91.5967,
  "weight": 0.8,
  "effective_weight": 0.381,
  "contribution": 34.894,
  "status": "evaluated",
  "description": "...",
  "raw_value": 1.7563,
  "unit": "sa_score",
  "evidence": {"product_sa_score": 1.7563},
  "warnings": [],
  "duration_ms": 0.2,
  "children": []
}
```

- `type`：`total`、`dimension`、`group` 或 `metric`。
- `weight`：相对同级节点的配置权重。
- `effective_weight`：排除不适用/错误节点后归一化的实际权重。
- `contribution`：该节点对父节点分数的贡献。
- `status`：`evaluated`、`not_applicable` 或 `error`。
- `raw_value`、`unit`、`evidence`、`warnings` 主要用于解释叶子分数。
- `children` 仅在存在下级节点时返回。

## 7. 历史证据库数据结构

JSON 或 JSONL 每条记录：

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

- `reaction_smiles` 必填。
- `yield_percent` 范围为 0–100。
- `success` 为布尔值。
- 同时存在收率与成功标签时，历史结果优先使用收率。
- 正式数据应先去重并统一标签定义，避免精确重复反应人为提高证据分。

## 8. 当前局限

- 单个 reaction SMILES 不包含当量、温度、压力、时间、溶剂、收率和后处理信息。
- 未映射输入无法精确判断具体成键原子、立体保持/翻转或消旋。
- SMARTS 官能团规则覆盖常见反应，不是完整反应模板库。
- 多产物输入只使用最大主产物参与大部分评分。
- 父节点会对可用子节点重新归一化，因此比较不同数据完整度的反应时应同时查看
  `core_applicability_coverage`。
- 所有规则阈值和权重仍需使用目标业务数据集做外部校准。
