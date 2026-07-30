import sys
import os
import math
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import BRICS
from rdkit import RDLogger
import json
# 禁用 RDKit 的多余警告
RDLogger.DisableLog('rdApp.*')
try:
    from openai import OpenAI
except Exception:  # openai is only required when LLM report is enabled
    OpenAI = None
import time
# ==========================================
# 0. 基础配置与 SMARTS 库
# ==========================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
try:
    from rdkit.Chem.RDConfig import RDContribDir
    sys.path.append(os.path.join(RDContribDir, 'SA_Score'))
    import sascorer
    HAS_SASCORER = True
except ImportError:
    HAS_SASCORER = False

FG_PATTERNS = {
    'ester': Chem.MolFromSmarts('[CX3](=O)[OX2][#6]'),
    'carboxylic_acid': Chem.MolFromSmarts('[CX3](=O)[OX2H1]'),
    'alcohol': Chem.MolFromSmarts('[OX2H][#6]'),
    'acyl_halide': Chem.MolFromSmarts('[CX3](=O)[Cl,Br]'),
    'amide': Chem.MolFromSmarts('[CX3](=O)[NX3]'),
    'amine': Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]'),
    'ether': Chem.MolFromSmarts('[OD2]([#6])[#6]'),
    'aryl_halide': Chem.MolFromSmarts('[c][Cl,Br,I]'),
    'boronic_acid_or_ester': Chem.MolFromSmarts('[B]([O])[O]'),
    'sulfonate': Chem.MolFromSmarts('S(=O)(=O)[O][#6]')
}

PROTECTING_GROUPS_SMARTS = {
    'Boc': Chem.MolFromSmarts('CC(C)(C)OC(=O)'),
    'Cbz': Chem.MolFromSmarts('O=C(OCH2c1ccccc1)'),
    'Ts': Chem.MolFromSmarts('CS(=O)(=O)c1ccc(C)cc1'),
    'Ms': Chem.MolFromSmarts('CS(=O)(=O)'),
    'Bpin_like': Chem.MolFromSmarts('B1OC(C)(C)C(C)(C)O1'),
    'silyl_like': Chem.MolFromSmarts('[Si]([C])([C])[C]')
}

LEAVE_PATTERNS = {
    'alkyl_or_aryl_br': Chem.MolFromSmarts('[C,c]-Br'),
    'alkyl_or_aryl_i': Chem.MolFromSmarts('[C,c]-I'),
    'alkyl_or_aryl_cl': Chem.MolFromSmarts('[C,c]-Cl'),
    'alkyl_or_aryl_f': Chem.MolFromSmarts('[C,c]-F'),
    'tosylate': Chem.MolFromSmarts('[O]S(=O)(=O)c1ccc(C)cc1'),
    'mesylate': Chem.MolFromSmarts('[O]S(=O)(=O)C'),
    'acyl_halide': Chem.MolFromSmarts('[CX3](=O)[Cl,Br]'),
    'free_alcohol': Chem.MolFromSmarts('[OX2H][#6]')
}

# ==========================================
# 1. 核心工具与解析函数
# ==========================================

def select_main_reactant(r_mols, p_mol):
    if not r_mols: return None
    try:
        p_fp = AllChem.GetMorganFingerprintAsBitVect(p_mol, 2, nBits=1024)
        best_mol, best_score = None, -1e9
        for m in r_mols:
            score = 0.0
            r_fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
            score += 1.5 * DataStructs.TanimotoSimilarity(r_fp, p_fp)
            score += 0.02 * min(m.GetNumHeavyAtoms(), 30)
            try:
                p_scaf = MurckoScaffold.GetScaffoldForMol(p_mol)
                r_scaf = MurckoScaffold.GetScaffoldForMol(m)
                if p_scaf.GetNumHeavyAtoms() > 0 and r_scaf.GetNumHeavyAtoms() > 0:
                    if p_scaf.HasSubstructMatch(r_scaf) or r_scaf.HasSubstructMatch(p_scaf):
                        score += 0.2
            except Exception:
                pass
            if score > best_score:
                best_score, best_mol = score, m
        return best_mol
    except Exception:
        return max(r_mols, key=lambda m: m.GetNumHeavyAtoms())

def parse_and_validate_molecules(reactants_smi, product_smi):
    if not reactants_smi or not product_smi: return None, None, None
    r_mols = [m for m in (Chem.MolFromSmiles(s) for s in reactants_smi.split('.') if s) if m is not None]
    if not r_mols: return None, None, None
    p_mol = Chem.MolFromSmiles(product_smi)
    if p_mol is None: return None, None, None
    return r_mols, p_mol, select_main_reactant(r_mols, p_mol)

def smooth_rl_reward(raw_score, center=0.45, pos_temp=1.6, neg_temp=1.1, clip_range=(-6.0, 6.0)):
    shifted = max(clip_range[0], min(clip_range[1], raw_score - center))
    return math.tanh(shifted / pos_temp) if shifted >= 0 else math.tanh(shifted / neg_temp)

# ==========================================
# 2. 评价指标函数库 (直接复用你的函数)
# ==========================================
# (为了代码简洁，去掉了部分长注释，保留了你所有的核心运算逻辑)

# --- A. 可行性 (Feasibility) ---
def eval_fragmentation_and_size(r_mols, p_mol, main_r_mol):
    score = 0.0
    if len(r_mols) > 4: score -= 0.2 * (len(r_mols) - 4)
    if sum(m.GetNumHeavyAtoms() for m in r_mols) > p_mol.GetNumHeavyAtoms() * 3.0: score -= 0.5 
    return score

def eval_close_loop_consistency(r_mols, p_mol, main_r_mol):
    p_smi = Chem.MolToSmiles(p_mol, canonical=True)
    return -0.8 if any(Chem.MolToSmiles(m, canonical=True) == p_smi for m in r_mols) else 0.0

def eval_structural_similarity(r_mols, p_mol, main_r_mol):
    try:
        if Chem.MolToSmiles(main_r_mol) == Chem.MolToSmiles(p_mol): return -0.8
        p_fp = AllChem.GetMorganFingerprintAsBitVect(p_mol, 2, nBits=1024)
        r_fp = AllChem.GetMorganFingerprintAsBitVect(main_r_mol, 2, nBits=1024)
        sim = DataStructs.TanimotoSimilarity(p_fp, r_fp)
        if sim < 0.1: return -0.6
        elif sim > 0.95: return -0.3
        else: return 0.5 * sim
    except Exception: return 0.0

def eval_heavy_heteroatom_match(r_mols, p_mol, main_r_mol):
    score = 0.0
    tracked = {'F', 'Cl', 'Br', 'I', 'P', 'S', 'B', 'Si'}
    p_counts = Counter([a.GetSymbol() for a in p_mol.GetAtoms() if a.GetSymbol() in tracked])
    r_counts = Counter()
    for m in r_mols: r_counts.update([a.GetSymbol() for a in m.GetAtoms() if a.GetSymbol() in tracked])
    for elem, req in p_counts.items():
        if r_counts[elem] < req: score -= 0.3 * (req - r_counts[elem])
    good_lvg = {'Cl', 'Br', 'I', 'S'} 
    for elem, prov in r_counts.items():
        if prov > p_counts[elem] and elem not in good_lvg:
            score -= 0.15 * (prov - p_counts[elem])
    return score

def eval_chemical_stability(r_mols, p_mol, main_r_mol):
    score = 0.0
    if sum(Descriptors.NumRadicalElectrons(m) for m in r_mols) > 0: score -= 0.8
    p_charge = sum(a.GetFormalCharge() for a in p_mol.GetAtoms())
    r_charge = sum(sum(a.GetFormalCharge() for a in m.GetAtoms()) for m in r_mols)
    if p_charge != r_charge: score -= 0.3 * abs(p_charge - r_charge)
    return score

def eval_ring_stability(r_mols, p_mol, main_r_mol):
    p_rings = rdMolDescriptors.CalcNumAromaticRings(p_mol)
    r_rings = sum(rdMolDescriptors.CalcNumAromaticRings(m) for m in r_mols)
    return -0.3 * (p_rings - r_rings) if r_rings < p_rings else 0.0

def eval_stereochemistry_sanity(r_mols, p_mol, main_r_mol):
    score = 0.0
    p_chiral = len(Chem.FindMolChiralCenters(p_mol, includeUnassigned=True))
    r_chiral = sum(len(Chem.FindMolChiralCenters(m, includeUnassigned=True)) for m in r_mols)
    if p_chiral == 0 and r_chiral > 0: score -= 0.2 * r_chiral
    if p_chiral > 0:
        if r_chiral > p_chiral + 1: score -= 0.3 
        elif r_chiral >= 1: score += 0.1 
    return score

def eval_murcko_scaffold_overlap(r_mols, p_mol, main_r_mol):
    try:
        p_scaf = MurckoScaffold.GetScaffoldForMol(p_mol)
        r_scaf = MurckoScaffold.GetScaffoldForMol(main_r_mol)
        if Chem.MolToSmiles(p_scaf) == Chem.MolToSmiles(r_scaf) and Chem.MolToSmiles(p_scaf): return 0.3
        if p_scaf.GetNumHeavyAtoms() > 0 and r_scaf.GetNumHeavyAtoms() > 0:
            if p_scaf.HasSubstructMatch(r_scaf) or r_scaf.HasSubstructMatch(p_scaf): return 0.1
            else: return -0.2
    except Exception: return 0.0
    return 0.0

def eval_element_balance(r_mols, p_mol, main_r_mol):
    score = 0.0
    tracked = {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si'}
    p_counts = Counter(a.GetSymbol() for a in p_mol.GetAtoms() if a.GetSymbol() in tracked)
    r_counts = Counter()
    for m in r_mols: r_counts.update(a.GetSymbol() for a in m.GetAtoms() if a.GetSymbol() in tracked)
    high, med = {'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'S'}, {'N', 'O'}
    for elem in tracked:
        diff = r_counts[elem] - p_counts[elem]
        weight = 0.25 if elem in high else 0.15 if elem in med else 0.08
        if diff < 0: score -= weight * abs(diff)
        elif diff > 3: score -= (weight * 0.4) * (diff - 3)
    return score

def eval_descriptor_delta(r_mols, p_mol, main_r_mol):
    try:
        return (
            -0.08 * abs(rdMolDescriptors.CalcNumRings(p_mol) - rdMolDescriptors.CalcNumRings(main_r_mol))
            -0.08 * abs(rdMolDescriptors.CalcNumAromaticRings(p_mol) - rdMolDescriptors.CalcNumAromaticRings(main_r_mol))
            -0.05 * abs(rdMolDescriptors.CalcNumHeteroatoms(p_mol) - rdMolDescriptors.CalcNumHeteroatoms(main_r_mol))
            -0.03 * abs(rdMolDescriptors.CalcNumRotatableBonds(p_mol) - rdMolDescriptors.CalcNumRotatableBonds(main_r_mol))
        )
    except Exception: return 0.0

def eval_functional_group_transform(r_mols, p_mol, main_r_mol):
    score = 0.0
    has_any = lambda k: any(m.HasSubstructMatch(FG_PATTERNS[k]) for m in r_mols if FG_PATTERNS[k])
    try:
        if p_mol.HasSubstructMatch(FG_PATTERNS['ester']):
            score += 0.3 if (has_any('carboxylic_acid') or has_any('acyl_halide')) and has_any('alcohol') else -0.2
        if p_mol.HasSubstructMatch(FG_PATTERNS['amide']):
            score += 0.3 if has_any('amine') and (has_any('carboxylic_acid') or has_any('acyl_halide')) else -0.2
        if rdMolDescriptors.CalcNumAromaticRings(p_mol) >= 1 and has_any('aryl_halide') and has_any('boronic_acid_or_ester'):
            score += 0.25
    except Exception: pass
    return score

def eval_pseudo_retro_copy(r_mols, p_mol, main_r_mol):
    try:
        sim = DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(p_mol, 2), AllChem.GetMorganFingerprintAsBitVect(main_r_mol, 2))
        others = [m for m in r_mols if Chem.MolToSmiles(m, canonical=True) != Chem.MolToSmiles(main_r_mol, canonical=True)]
        if sim > 0.90:
            if not others: return -0.40
            elif all(m.GetNumHeavyAtoms() <= 2 for m in others): return -0.30
    except Exception: pass
    return 0.0

def eval_leaving_group_quality(r_mols, p_mol, main_r_mol):
    score = 0.0
    has_good = False
    for m in r_mols:
        for k, v in {'alkyl_or_aryl_br': 0.15, 'alkyl_or_aryl_i': 0.18, 'tosylate': 0.2, 'acyl_halide': 0.2}.items():
            if m.HasSubstructMatch(LEAVE_PATTERNS[k]): score += v; has_good = True
    return score

def eval_chemoselectivity(r_mols, p_mol, main_r_mol):
    try:
        amine_pat = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')
        if p_mol.HasSubstructMatch(Chem.MolFromSmarts('[CX3](=O)[OX2][#6]')) and \
           p_mol.HasSubstructMatch(amine_pat) and any(m.HasSubstructMatch(amine_pat) for m in r_mols):
            return -0.45
    except Exception: pass
    return 0.0

# --- B. 经济性 (Economy) ---
def eval_synthetic_accessibility(r_mols, p_mol, main_r_mol):
    if HAS_SASCORER:
        try:
            p_sa, r_sa = sascorer.calculateScore(p_mol), sascorer.calculateScore(main_r_mol)
            return 0.1 * (p_sa - r_sa) if r_sa < p_sa else -0.3
        except Exception: pass
    return 0.0

def eval_atom_economy_penalty(r_mols, p_mol, main_r_mol):
    try:
        rmw = sum(Descriptors.MolWt(m) for m in r_mols)
        if rmw <= 0: return 0.0
        ae = Descriptors.MolWt(p_mol) / rmw
        return -0.5 if ae < 0.3 else -0.2 if ae < 0.5 else 0.0
    except Exception: return 0.0

def eval_protecting_group_reasonableness(r_mols, p_mol, main_r_mol):
    score = 0.0
    for pat in PROTECTING_GROUPS_SMARTS.values():
        if pat and any(m.HasSubstructMatch(pat) for m in r_mols) and not p_mol.HasSubstructMatch(pat):
            score += 0.10
    return score

def eval_carbon_efficiency_penalty(r_mols, p_mol, main_r_mol):
    try:
        rc = sum(sum(1 for a in m.GetAtoms() if a.GetAtomicNum()==6) for m in r_mols)
        if rc == 0: return 0.0
        ce = sum(1 for a in p_mol.GetAtoms() if a.GetAtomicNum()==6) / rc
        return -0.6 if ce < 0.4 else -0.25 if ce < 0.6 else 0.0
    except Exception: return 0.0

# --- C. 安全性 (Safety) ---
def eval_structural_alerts(r_mols, p_mol, main_r_mol):
    # 这里直接使用你在上下文中更新的分级+补偿的打分系统
    score = 0.0
    ALERTS = {
        "[*-1]-[*-1]": (-1.0, "ABSURD"), "[O,S]-[O,S]-[O,S]": (-0.9, "EXPLOSIVE"),
        "[N]-[N]-[N]-[N]": (-0.9, "EXPLOSIVE"), "[N]=[N]=[N]": (-0.5, "AZIDE"),
        "[OX2]-[OX2]": (-0.4, "PEROXIDE")
    }
    triazole = Chem.MolFromSmarts('c1nnnc1')
    for m in r_mols:
        for smi, (base_pen, a_type) in ALERTS.items():
            pat = Chem.MolFromSmarts(smi)
            if pat and m.HasSubstructMatch(pat):
                pen = base_pen
                if a_type == "AZIDE":
                    nc = sum(1 for a in m.GetAtoms() if a.GetAtomicNum()==6)
                    no = sum(1 for a in m.GetAtoms() if a.GetAtomicNum()==8)
                    nn = sum(1 for a in m.GetAtoms() if a.GetAtomicNum()==7)
                    if nn > 0 and (nc+no)/nn >= 3.0: pen = -0.15
                    if p_mol and triazole and p_mol.HasSubstructMatch(triazole): pen = 0.20
                score += pen
    return score

# ==========================================
# 3. 评价指标注册表 (核心配置字典)
# ==========================================
# 统一接口：func(r_mols, p_mol, main_r_mol) -> float
METRICS_REGISTRY = {
    "feasibility": {
        "close_loop_consistency":    {"func": eval_close_loop_consistency,    "weight": 1.8, "avg_impact_score": -0.003, "desc": "禁止反应物与产物完全相同(死循环)"},
        "pseudo_retro_copy":         {"func": eval_pseudo_retro_copy,         "weight": 1.4, "avg_impact_score": -0.003, "desc": "惩罚无实质性化学转化的伪拆分"},
        "heavy_heteroatom_match":    {"func": eval_heavy_heteroatom_match,    "weight": 1.2, "avg_impact_score": -0.179, "desc": "严惩重/稀缺杂原子的凭空出现"},
        "element_balance":           {"func": eval_element_balance,           "weight": 1.0, "avg_impact_score": -0.685, "desc": "评估全局元素配平守恒关系"},
        "structural_similarity":     {"func": eval_structural_similarity,     "weight": 1.3, "avg_impact_score":  0.217, "desc": "评估主前体与产物的骨架相似度"},
        "murcko_scaffold_overlap":   {"func": eval_murcko_scaffold_overlap,   "weight": 1.0, "avg_impact_score":  0.085, "desc": "检查核心拓扑骨架的遗传与继承"},
        "functional_group_transform":{"func": eval_functional_group_transform,"weight": 1.1, "avg_impact_score": -0.065, "desc": "基于常见官能团的逆合成前体检查"},
        "chemical_stability":        {"func": eval_chemical_stability,        "weight": 1.0, "avg_impact_score": -0.020, "desc": "评估电荷守恒与自由基稳定性"},
        "chemoselectivity":          {"func": eval_chemoselectivity,          "weight": 1.2, "avg_impact_score": -0.010, "desc": "捕捉违背常识的化学选择性错误"},
        "leaving_group_quality":     {"func": eval_leaving_group_quality,     "weight": 0.5, "avg_impact_score":  0.020, "desc": "评估反应物中是否存在良好的离去基团"},
        "stereochemistry_sanity":    {"func": eval_stereochemistry_sanity,    "weight": 0.6, "avg_impact_score":  0.006, "desc": "严惩无端捏造手性中心"},
        "ring_stability":            {"func": eval_ring_stability,            "weight": 0.6, "avg_impact_score": -0.098, "desc": "防止异常地破坏芳香环"},
        "descriptor_delta":          {"func": eval_descriptor_delta,          "weight": 0.7, "avg_impact_score": -0.257, "desc": "评估分子整体理化性质描述符的差异"},
        "fragmentation_and_size":    {"func": eval_fragmentation_and_size,    "weight": 0.5, "avg_impact_score": -0.006, "desc": "惩罚拆分过度碎片化或质量荒谬膨胀"}
    },
    "economy": {
        "synthetic_accessibility":   {"func": eval_synthetic_accessibility,   "weight": 0.8, "avg_impact_score":  0.000, "desc": "评估前体SA Score是否比产物更容易获取"},
        "protecting_group_usage":    {"func": eval_protecting_group_reasonableness, "weight": 0.4, "avg_impact_score": 0.003, "desc": "奖励合理的保护基或活化基团使用"},
        "atom_economy":              {"func": eval_atom_economy_penalty,      "weight": 0.6, "avg_impact_score": -0.009, "desc": "惩罚理论原子利用率(AE)过低的废料反应"},
        "carbon_efficiency":         {"func": eval_carbon_efficiency_penalty, "weight": 0.6, "avg_impact_score": -0.015, "desc": "惩罚丢弃大量碳骨架的无经济性转化"}
    },
    "safety": {
        "structural_alerts":         {"func": eval_structural_alerts,         "weight": 1.2, "avg_impact_score": -0.012, "desc": "捕捉极其不稳定或危险的爆炸性/高能结构"}
    }
}

# ==========================================
# 4. 主评价类 (对外暴露接口)
# ==========================================

class ReactionEvaluator:
    def __init__(self, registry=METRICS_REGISTRY):
        self.registry = registry

    def evaluate(self, reactants_smi: str, product_smi: str) -> dict:
        """
        综合评估反应，返回打分结果和明细。
        """
        r_mols, p_mol, main_r_mol = parse_and_validate_molecules(reactants_smi, product_smi)
        
        # 解析失败兜底
        if r_mols is None:
            return {"total_reward": -1.0, "status": "invalid_smiles", "details": {}}

        raw_score = 0.35 # 基础分
        details = {}
        severe_penalty_count = 0

        # 遍历三个维度进行打分
        for dimension, metrics in self.registry.items():
            details[dimension] = {}
            for metric_name, config in metrics.items():
                # 执行函数
                metric_val = config["func"](r_mols, p_mol, main_r_mol)
                weighted_val = metric_val * config["weight"]
                
                # 累加分数
                raw_score += weighted_val
                
                # 记录明细
                details[dimension][metric_name] = {
                    "raw_val": round(metric_val, 4),
                    "weight": config["weight"],
                    "weighted_val": round(weighted_val, 4),
                    "avg_impact_score": config.get("avg_impact_score"),
                    "delta_vs_avg": round(weighted_val - config.get("avg_impact_score", 0.0), 4),
                    "desc": config["desc"]
                }

                # 记录严重错误 (兜底机制)
                if metric_name in ["close_loop_consistency", "structural_alerts", "heavy_heteroatom_match", "element_balance"]:
                    if metric_val < -0.5: severe_penalty_count += 1
                if metric_name in ["pseudo_retro_copy", "chemoselectivity"]:
                    if metric_val <= -0.4: severe_penalty_count += 1

        # 极端错误兜底扣分
        if severe_penalty_count >= 3:
            raw_score -= 1.0
        elif severe_penalty_count == 2:
            raw_score -= 0.4

        # 平滑映射到 (-1, 1)
        final_reward = smooth_rl_reward(raw_score)

        return {
            "total_reward": round(final_reward, 4),
            "raw_score": round(raw_score, 4),
            "severe_errors": severe_penalty_count,
            "status": "success",
            "details": details
        }


class QwenReviewer:
    def __init__(self, api_key: str = None, model: str = "qwen-plus"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is required when LLM report is enabled")
        # 配置千问的兼容模式端点
        if OpenAI is None:
            raise ImportError("Install openai>=1.40.0 to enable Qwen/DashScope reports")
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        self.system_prompt = """
# Role
你是一位世界顶尖的计算化学家和逆合成分析（Retrosynthesis）专家。你擅长通过自动化评价系统的数据，结合庞大的化学常识，深入评估化学反应的合理性、安全性、经济性以及创新性。

# Task
我将为你提供一个化学反应（反应物和产物的 SMILES），以及由 RDKit 规则引擎自动生成的深度评估数据（JSON 格式）。
请你仔细解析 JSON 中的数据。特别注意以下字段的含义：
- `desc`: 该评价指标的化学物理意义。
- `raw_val`: **当前化学反应**在该指标上的实际得分（通常正数代表奖励/契合，负数代表违规/惩罚，0代表未触发）。
- `weighted_val`: 当前反应在该指标上的加权贡献，等于 `raw_val * weight`。
- `avg_impact_score`: 基于历史样本统计的该指标平均影响值，可作为大盘基准。
- `delta_vs_avg`: 当前加权贡献相对大盘均值的差异，负值越大表示越异常。

你需要**对比当前加权贡献 (`weighted_val`) 与大盘平均值 (`avg_impact_score`)**，判断该反应是属于常规操作、表现优异，还是存在罕见缺陷，并完成以下任务：
1. **百分制转换**：将可行性 (Feasibility)、安全性 (Safety)、经济性 (Economy) 映射为 0-100 分。如果单项实际得分远低于大盘平均值（尤其是严重扣分项），应大幅拉低该维度的百分制得分；如果表现优于平均值或未触发常规惩罚，应给予高分。
2. **创新性打分**：基于你自身的化学知识，对该步骤的“创新性 (Innovation)”进行独立打分（0-100 分）。
3. **输出比较分析报告**：撰写结构化的深度评价与改进建议。

# Inputs
- **反应物 (Reactants)**: {reactants_smiles}
- **产物 (Product)**: {product_smiles}
- **规则引擎评估数据 (Engine Data)**: 
{evaluator_json_output}

# Output Format Specification
请严格按照以下 Markdown 格式输出你的分析报告：

### 📊 综合评分 (百分制)
- **可行性 (Feasibility)**: [0-100] 分
- **安全性 (Safety)**: [0-100] 分
- **经济性 (Economy)**: [0-100] 分
- **创新性 (Innovation)**: [0-100] 分

### 🔬 深度对比评价分析
- **可行性分析**: [结合具体指标展开。例如对比 `structural_similarity` 或 `functional_group_transform` 的实际得分与平均值。指出其转化是否符合大盘规律，或者是否因为触发了元素不守恒等惩罚而低于基准。]
- **安全性分析**: [查看 `structural_alerts` 的得分，如果得分为 0，说明成功避开了危险基团。若有扣分，结合平均值指出其危险程度。补充你个人的安全建议。]
- **经济性分析**: [结合 `atom_economy` 和 `carbon_efficiency`，评估该反应产生的废料是否高于行业平均水平（平均值）。]
- **创新性分析**: [（由大模型自主生成）评价该逆合成拆分的断键位置是否精妙，是否采用了非传统的但高效的转化策略。]

### 💡 改进建议
[基于上述对比分析中暴露出的“低于平均值”或“触发惩罚”的弱点，给出 2-3 条具体的化学改进方案（如：优化离去基团、采用更经济的前体、调整保护基策略等）。]
"""

    def generate_report(self, reactants_smi: str, product_smi: str, evaluator_details: dict) -> str:
        # 组装发给大模型的数据
        user_prompt = f"""
- **反应物 (Reactants)**: {reactants_smi}
- **产物 (Product)**: {product_smi}
- **规则引擎评估数据**: 
{json.dumps(evaluator_details, indent=2, ensure_ascii=False)}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model, # 可选: qwen-turbo, qwen-plus, qwen-max
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3 # 偏向理性与客观分析
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ 大模型调用失败: {str(e)}"

