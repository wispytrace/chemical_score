import sys
import os
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import math
from rdkit.Chem import BRICS

# 假设 sascorer 已经在环境中正确配置
try:
    from rdkit.Chem.RDConfig import RDContribDir
    sys.path.append(os.path.join(RDContribDir, 'SA_Score'))
    import sascorer
except ImportError:
    print("Warning: sascorer not found. SA Score evaluation will be bypassed.")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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

# 结构警报的 SMARTS
ALERT_PATTERNS = [
    Chem.MolFromSmarts("[O,S]-[O,S]-[O,S]"),
    Chem.MolFromSmarts("[N]=[N]=[N]"),
    Chem.MolFromSmarts("[#6]1-[#6]-[#6]1#*"),
    Chem.MolFromSmarts("[*-1]-[*-1]")
]

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
def select_main_reactant(r_mols, p_mol):
    """
    综合 Morgan 相似性、Murcko 骨架相似倾向和分子大小，
    选择最可能的主反应物。
    """
    if not r_mols:
        return None

    try:
        p_fp = AllChem.GetMorganFingerprintAsBitVect(p_mol, 2, nBits=1024)
        best_mol = None
        best_score = -1e9

        for m in r_mols:
            score = 0.0

            # 1. Morgan 相似性
            r_fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
            sim = DataStructs.TanimotoSimilarity(r_fp, p_fp)
            score += 1.5 * sim

            # 2. 大小辅助项
            heavy_atoms = m.GetNumHeavyAtoms()
            score += 0.02 * min(heavy_atoms, 30)

            # 3. 若 Murcko scaffold 和产物有关系，再加一点分
            try:
                p_scaf = MurckoScaffold.GetScaffoldForMol(p_mol)
                r_scaf = MurckoScaffold.GetScaffoldForMol(m)

                if p_scaf.GetNumHeavyAtoms() > 0 and r_scaf.GetNumHeavyAtoms() > 0:
                    if p_scaf.HasSubstructMatch(r_scaf) or r_scaf.HasSubstructMatch(p_scaf):
                        score += 0.2
            except Exception:
                pass

            if score > best_score:
                best_score = score
                best_mol = m

        return best_mol

    except Exception:
        return max(r_mols, key=lambda m: m.GetNumHeavyAtoms())

def parse_and_validate_molecules(reactants_smi, product_smi):
    """验证 SMILES 合法性并提取 RDKit Mol 对象"""
    if not reactants_smi or not product_smi:
        return None, None, None

    r_mols = []
    for r_smi in reactants_smi.split('.'):
        if r_smi == "":
            continue
        m = Chem.MolFromSmiles(r_smi)
        if m is None:
            return None, None, None
        r_mols.append(m)

    if len(r_mols) == 0:
        return None, None, None

    p_mol = Chem.MolFromSmiles(product_smi)
    if p_mol is None:
        return None, None, None

    # 改为：基于相似性 + 大小综合选择主反应物
    main_r_mol = select_main_reactant(r_mols, p_mol)

    return r_mols, p_mol, main_r_mol


def eval_fragmentation_and_size(r_mols, p_mol):
    """惩罚碎片爆炸和质量荒谬膨胀"""
    score = 0.0
    num_reactants = len(r_mols)
    
    # 1. 碎片爆炸惩罚
    if num_reactants > 4:
        score -= 0.2 * (num_reactants - 4)

    # 2. 荒谬膨胀惩罚
    p_heavy_atoms = p_mol.GetNumHeavyAtoms()
    r_heavy_atoms = sum(m.GetNumHeavyAtoms() for m in r_mols)
    if r_heavy_atoms > p_heavy_atoms * 3.0:
        score -= 0.5 
        
    return score

def eval_close_loop_consistency(r_mols, p_mol):
    score = 0.0
    p_smi = Chem.MolToSmiles(p_mol, canonical=True)
    for m in r_mols:
        if Chem.MolToSmiles(m, canonical=True) == p_smi:
            score -= 0.8
            break
    return score

def eval_structural_similarity(main_r_mol, p_mol):
    """评估骨架的继承性，严惩复制粘贴和乱编分子"""
    score = 0.0
    try:
        # 绝对一致惩罚 (模型直接复制粘贴了产物)
        if Chem.MolToSmiles(main_r_mol) == Chem.MolToSmiles(p_mol):
            return -0.8

        p_fp = AllChem.GetMorganFingerprintAsBitVect(p_mol, 2, nBits=1024)
        r_fp = AllChem.GetMorganFingerprintAsBitVect(main_r_mol, 2, nBits=1024)
        similarity = DataStructs.TanimotoSimilarity(p_fp, r_fp)

        if similarity < 0.1:
            score -= 0.6  # 瞎造无关分子
        elif similarity > 0.95:
            score -= 0.3  # 极度相似，没发生实质化学变化
        else:
            score += 0.5 * similarity  # 正常骨架保留奖励 (把权重压到0.5防作弊)
    except Exception:
        pass
    
    return score

def eval_synthetic_accessibility(main_r_mol, p_mol):
    """评估反应物是否比产物更容易相对获取（逆合成逻辑）"""
    score = 0.0
    try:
        if 'sascorer' in sys.modules:
            p_sa = sascorer.calculateScore(p_mol)
            r_sa = sascorer.calculateScore(main_r_mol)

            if r_sa < p_sa:
                score += 0.1 * (p_sa - r_sa) # 越拆越简单，加分
            else:
                score -= 0.3  # 越拆越复杂，扣分
    except Exception:
        pass
    return score

def eval_heavy_heteroatom_match(r_mols, p_mol):
    """
    重杂原子追踪：
    专门监控那些不该无端出现，也不该无端消失的稀缺/关键元素。
    """
    score = 0.0
    
    # 关注的稀缺/重杂原子列表 (把 C, H, N, O 排除在外)
    # 我们主要监控 卤素(F, Cl, Br, I)、磷(P)、硫(S)、硼(B)、硅(Si) 等
    tracked_elements = {'F', 'Cl', 'Br', 'I', 'P', 'S', 'B', 'Si'}
    
    p_counts = Counter([atom.GetSymbol() for atom in p_mol.GetAtoms() if atom.GetSymbol() in tracked_elements])
    r_counts = Counter()
    for m in r_mols:
        r_counts.update([atom.GetSymbol() for atom in m.GetAtoms() if atom.GetSymbol() in tracked_elements])
        
    # ===============================================
    # 1. 严惩“无中生有”（Magic Appearance）
    # 产物里有，但预测的反应物完全没提供源头（绝对的扣分项）
    # ===============================================
    for elem, required_count in p_counts.items():
        if r_counts[elem] < required_count:
            missing_count = required_count - r_counts[elem]
            # 权重较重：每凭空捏造一个关键原子，扣大分
            score -= 0.3 * missing_count 

    # ===============================================
    # 2. 轻微惩罚“莫名消失”（Suspicious Disappearance）
    # 反应物里带了，但产物里没有。
    # ===============================================
    # 注意：在逆合成中，氯(Cl)、溴(Br)、碘(I)、硫(S，如硫醇/SO2) 是极好的“离去基团”！
    # 如果它们消失了，往往说明模型预测了一个非常聪明的卤代烃或磺酸酯反应物，这是对的！不能扣分！
    # 但如果模型无端端给反应物加上了 氟(F) 或 磷(P)、硼(B)，产物却没有，这就很可疑了。
    
    good_leaving_groups = {'Cl', 'Br', 'I', 'S'} 
    
    for elem, provided_count in r_counts.items():
        if provided_count > p_counts[elem]:
            # 如果是良好的离去基团，豁免惩罚（甚至你可以选择给一点小奖励）
            if elem in good_leaving_groups:
                pass  # 顺利离去，合理！
            else:
                extra_count = provided_count - p_counts[elem]
                # 比如产物没氟，模型非要拿个含氟的物质去反应，轻微扣分
                score -= 0.15 * extra_count 
                
    return score


def eval_chemical_stability(r_mols, p_mol):
    """评估电荷守恒、自由基惩罚以及离去基团合理性"""
    score = 0.0
    
    total_radicals = sum(Descriptors.NumRadicalElectrons(m) for m in r_mols)
    if total_radicals > 0:
        score -= 0.8
        
    p_charge = sum(atom.GetFormalCharge() for atom in p_mol.GetAtoms())
    r_charge = sum(sum(atom.GetFormalCharge() for atom in m.GetAtoms()) for m in r_mols)
    if p_charge != r_charge:
        score -= 0.3 * abs(p_charge - r_charge)
        
    REASONABLE_FRAGMENTS = {'O', 'Cl', 'Br', 'I', 'O=C=O', 'N', '[OH-]', '[Cl-]', '[Br-]'} 
    for m in r_mols:
        if m.GetNumHeavyAtoms() <= 3:
            frag_smi = Chem.MolToSmiles(m, canonical=True)
            if frag_smi in REASONABLE_FRAGMENTS:
                score += 0.1
            else:
                score -= 0.15
                
    return score

def eval_ring_stability(r_mols, p_mol):
    """防止模型极其异常地打碎或者凭空捏造芳香环"""
    score = 0.0
    p_rings = rdMolDescriptors.CalcNumAromaticRings(p_mol)
    r_rings = sum(rdMolDescriptors.CalcNumAromaticRings(m) for m in r_mols)

    # 逆向反应中，如果反应物的芳香环比产物少，说明模型预测出"破坏了芳香环"
    # 这在常规有机反应中能量壁垒极高
    if r_rings < p_rings:
        score -= 0.3 * (p_rings - r_rings) 
        
    return score


def eval_stereochemistry_sanity(r_mols, p_mol):
    """评估手性中心的合理性，严惩乱造手性"""
    score = 0.0
    
    # 查找产物和反应物的手性中心数量
    p_chiral_centers = Chem.FindMolChiralCenters(p_mol, includeUnassigned=True)
    
    r_chiral_count = 0
    for m in r_mols:
        r_chiral_count += len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
        
    p_chiral_count = len(p_chiral_centers)
    
    # 如果产物根本没有手性，但模型非要预测出带有极强手性的复杂反应物
    if p_chiral_count == 0 and r_chiral_count > 0:
        score -= 0.2 * r_chiral_count  # 惩罚画蛇添足，增加合成成本
        
    # 如果产物有手性，模型如果能保留手性，给予微小奖励；如果完全丢失，略微扣分
    if p_chiral_count > 0:
        if r_chiral_count > p_chiral_count + 1:
            score -= 0.3 # 手性中心爆增，绝对是瞎预测的
        elif r_chiral_count >= 1:
            score += 0.1 # 成功保留或关注到了手性特征，给予奖励
            
    return score

def eval_structural_alerts(r_mols):
    """捕捉化学上极其不稳定或现实中难以存在的高能/危险结构"""
    score = 0.0
    

    alert_patterns = [smarts for smarts in ALERT_PATTERNS]
    
    for m in r_mols:
        for pat in alert_patterns:
            if pat is not None and m.HasSubstructMatch(pat):
                score -= 0.8  # 一旦触发结构警报，遭受毁灭性扣分
                
    return score

def eval_murcko_scaffold_overlap(main_r_mol, p_mol):
    """强制要求核心宏观拓扑架构(去掉侧链后的核心)的遗传与继承"""
    score = 0.0
    try:
        # 提取产物和主要反应物的核心 Murcko 骨架
        p_scaffold = MurckoScaffold.GetScaffoldForMol(p_mol)
        r_scaffold = MurckoScaffold.GetScaffoldForMol(main_r_mol)
        
        # 将骨架转化为 SMARTS 或 SMILES 进行精确结构比较
        p_scaf_smi = Chem.MolToSmiles(p_scaffold)
        r_scaf_smi = Chem.MolToSmiles(r_scaffold)
        
        if p_scaf_smi == r_scaf_smi and p_scaf_smi != "":
            # 骨架完全一致，说明模型非常聪明地只在侧链(官能团)上做了单步修饰
            score += 0.3 
        else:
            # 如果骨架不一致，看产物骨架是否是反应物骨架的子结构 (比如反应是合环或开环)
            if p_scaffold.GetNumHeavyAtoms() > 0 and r_scaffold.GetNumHeavyAtoms() > 0:
                if p_scaffold.HasSubstructMatch(r_scaffold) or r_scaffold.HasSubstructMatch(p_scaffold):
                    score += 0.1 # 有明显的子结构关系，合理
                else:
                    score -= 0.2 # 宏观主骨架发生了天翻地覆的断裂变形，警惕！
    except Exception:
        pass
        
    return score

def eval_element_balance(r_mols, p_mol):
    """
    评估反应物与产物之间的元素整体守恒关系。
    重点：
    1. 严惩产物中的元素在反应物中无来源
    2. 轻惩反应物中出现过多无意义冗余元素
    """
    score = 0.0
    tracked_elements = {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si'}

    p_counts = Counter(atom.GetSymbol() for atom in p_mol.GetAtoms() if atom.GetSymbol() in tracked_elements)
    r_counts = Counter()
    for m in r_mols:
        r_counts.update(atom.GetSymbol() for atom in m.GetAtoms() if atom.GetSymbol() in tracked_elements)

    # 不同元素的权重可区别对待
    high_penalty_elements = {'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'S'}
    medium_penalty_elements = {'N', 'O'}
    low_penalty_elements = {'C'}

    for elem in tracked_elements:
        diff = r_counts[elem] - p_counts[elem]

        # 反应物元素数不足，说明“无中生有”
        if diff < 0:
            missing = abs(diff)
            if elem in high_penalty_elements:
                score -= 0.25 * missing
            elif elem in medium_penalty_elements:
                score -= 0.15 * missing
            else:
                score -= 0.08 * missing

        # 反应物中该元素多太多，说明可能乱塞无关组分
        elif diff > 3:
            extra = diff - 3
            if elem in high_penalty_elements:
                score -= 0.10 * extra
            elif elem in medium_penalty_elements:
                score -= 0.06 * extra
            else:
                score -= 0.03 * extra

    return score


def eval_descriptor_delta(main_r_mol, p_mol):
    """
    评估主要反应物与产物在整体分子描述符上的差异。
    差异过大，通常意味着不是一个合理的单步逆合成。
    """
    score = 0.0
    try:
        p_num_rings = rdMolDescriptors.CalcNumRings(p_mol)
        r_num_rings = rdMolDescriptors.CalcNumRings(main_r_mol)

        p_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(p_mol)
        r_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(main_r_mol)

        p_hetero = rdMolDescriptors.CalcNumHeteroatoms(p_mol)
        r_hetero = rdMolDescriptors.CalcNumHeteroatoms(main_r_mol)

        p_rot = rdMolDescriptors.CalcNumRotatableBonds(p_mol)
        r_rot = rdMolDescriptors.CalcNumRotatableBonds(main_r_mol)

        p_hba = rdMolDescriptors.CalcNumHBA(p_mol)
        r_hba = rdMolDescriptors.CalcNumHBA(main_r_mol)

        p_hbd = rdMolDescriptors.CalcNumHBD(p_mol)
        r_hbd = rdMolDescriptors.CalcNumHBD(main_r_mol)

        score -= 0.08 * abs(p_num_rings - r_num_rings)
        score -= 0.08 * abs(p_aromatic_rings - r_aromatic_rings)
        score -= 0.05 * abs(p_hetero - r_hetero)
        score -= 0.03 * abs(p_rot - r_rot)
        score -= 0.04 * abs(p_hba - r_hba)
        score -= 0.04 * abs(p_hbd - r_hbd)

    except Exception:
        pass

    return score

def eval_functional_group_transform(r_mols, p_mol):
    """
    基于常见官能团的逆合成逻辑，判断前体类型是否合理。
    当前覆盖：
    - 酯
    - 酰胺
    - 醚
    - 芳基硼酸/硼酸酯相关偶联前体
    """
    score = 0.0

    def has_any(mols, key):
        pat = FG_PATTERNS[key]
        return any(m.HasSubstructMatch(pat) for m in mols if pat is not None)

    try:
        # ---------- 酯 ----------
        if p_mol.HasSubstructMatch(FG_PATTERNS['ester']):
            has_acid = has_any(r_mols, 'carboxylic_acid')
            has_alcohol = has_any(r_mols, 'alcohol')
            has_acyl_halide = has_any(r_mols, 'acyl_halide')

            if (has_acid and has_alcohol) or (has_acyl_halide and has_alcohol):
                score += 0.30
            else:
                score -= 0.20

        # ---------- 酰胺 ----------
        if p_mol.HasSubstructMatch(FG_PATTERNS['amide']):
            has_amine = has_any(r_mols, 'amine')
            has_acid = has_any(r_mols, 'carboxylic_acid')
            has_acyl_halide = has_any(r_mols, 'acyl_halide')

            if has_amine and (has_acid or has_acyl_halide):
                score += 0.30
            else:
                score -= 0.20

        # ---------- 醚 ----------
        if p_mol.HasSubstructMatch(FG_PATTERNS['ether']):
            has_alcohol = has_any(r_mols, 'alcohol')
            has_leaving_partner = has_any(r_mols, 'sulfonate') or has_any(r_mols, 'aryl_halide')

            if has_alcohol and has_leaving_partner:
                score += 0.20

        # ---------- 偶联逻辑：产物含芳香体系时，若前体中有卤代芳烃+硼酸/硼酸酯，给奖励 ----------
        p_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(p_mol)
        if p_aromatic_rings >= 1:
            has_aryl_halide = has_any(r_mols, 'aryl_halide')
            has_boron_partner = has_any(r_mols, 'boronic_acid_or_ester')
            if has_aryl_halide and has_boron_partner:
                score += 0.25

    except Exception:
        pass

    return score

def eval_protecting_group_reasonableness(r_mols, p_mol):
    """
    对常见保护基/活化基进行识别：
    如果反应物中出现这些结构，而产物中没有，不扣分，甚至给微弱奖励。
    适合逆合成场景。
    """
    score = 0.0

    try:
        for _, pat in PROTECTING_GROUPS_SMARTS.items():
            if pat is None:
                continue

            p_has = p_mol.HasSubstructMatch(pat)
            r_has = any(m.HasSubstructMatch(pat) for m in r_mols)

            # 反应物有、产物没有：典型“前体保护/活化”行为
            if r_has and not p_has:
                score += 0.10

    except Exception:
        pass

    return score

def eval_pseudo_retro_copy(main_r_mol, r_mols, p_mol):
    """
    惩罚“看起来不是直接复制，但本质上只是产物微调/拼接一个无意义碎片”的伪逆合成。
    """
    score = 0.0
    try:
        p_fp = AllChem.GetMorganFingerprintAsBitVect(p_mol, 2, nBits=1024)
        r_fp = AllChem.GetMorganFingerprintAsBitVect(main_r_mol, 2, nBits=1024)
        sim = DataStructs.TanimotoSimilarity(p_fp, r_fp)

        # 除主反应物外的其他组分
        main_smi = Chem.MolToSmiles(main_r_mol, canonical=True)
        other_mols = [m for m in r_mols if Chem.MolToSmiles(m, canonical=True) != main_smi]

        if sim > 0.90:
            if len(other_mols) == 0:
                score -= 0.40
            else:
                # 其他组分都特别小，且看起来不像合理拆分
                if all(m.GetNumHeavyAtoms() <= 2 for m in other_mols):
                    score -= 0.30

    except Exception:
        pass

    return score

def eval_brics_consistency(r_mols, p_mol):
    """
    利用 BRICS 分解思想，检查反应物是否与产物的可拆片段存在一定一致性。
    """
    score = 0.0
    try:
        brics_fragments = BRICS.BRICSDecompose(p_mol)
        if not brics_fragments:
            return 0.0

        reactant_smis = set(Chem.MolToSmiles(m, canonical=True) for m in r_mols)

        overlap = 0
        for frag in brics_fragments:
            frag_clean = frag.replace('[*]', '')
            if len(frag_clean) < 2:
                continue
            for rsmi in reactant_smis:
                if frag_clean and frag_clean in rsmi:
                    overlap += 1
                    break

        if overlap >= 2:
            score += 0.20
        elif overlap == 1:
            score += 0.08
        else:
            score -= 0.10

    except Exception:
        pass

    return score

def eval_leaving_group_quality(r_mols, p_mol):
    """
    评估反应物中的离去基团质量。
    好离去基有助于说明逆合成前体设计合理。
    """
    score = 0.0



    try:
        has_good = False

        for m in r_mols:
            if m.HasSubstructMatch(LEAVE_PATTERNS['alkyl_or_aryl_br']):
                score += 0.15
                has_good = True
            if m.HasSubstructMatch(LEAVE_PATTERNS['alkyl_or_aryl_i']):
                score += 0.18
                has_good = True
            if m.HasSubstructMatch(LEAVE_PATTERNS['tosylate']):
                score += 0.20
                has_good = True
            if m.HasSubstructMatch(LEAVE_PATTERNS['mesylate']):
                score += 0.18
                has_good = True
            if m.HasSubstructMatch(LEAVE_PATTERNS['acyl_halide']):
                score += 0.20
                has_good = True

            # Cl 可以作为离去基，但能力中等
            if m.HasSubstructMatch(LEAVE_PATTERNS['alkyl_or_aryl_cl']):
                score += 0.08

            # F 作为普通离去基通常很差（除非特殊 SNAr，这里只轻罚）
            if m.HasSubstructMatch(LEAVE_PATTERNS['alkyl_or_aryl_f']):
                score -= 0.08

        # 如果产物含醚/酰胺/酯等明显需要活化前体，但反应物完全没有好离去基，可轻微扣分
        p_has_activated_bond = (
            p_mol.HasSubstructMatch(Chem.MolFromSmarts('[CX3](=O)[OX2][#6]')) or
            p_mol.HasSubstructMatch(Chem.MolFromSmarts('[CX3](=O)[NX3]')) or
            p_mol.HasSubstructMatch(Chem.MolFromSmarts('[OD2]([#6])[#6]'))
        )
        if p_has_activated_bond and not has_good:
            score -= 0.10

    except Exception:
        pass

    return score


def smooth_rl_reward(raw_score, center=0.45, pos_temp=1.6, neg_temp=1.1, clip_range=(-6.0, 6.0)):
    """
    将原始启发式分数平滑映射到 (-1, 1)。

    设计原则：
    1. 以 center 作为中性分界点
    2. 正向奖励平滑，避免稍微合理就过早饱和
    3. 负向惩罚更敏感，强化对明显错误/作弊预测的打压
    4. clip 防止极端原始分数影响数值稳定性
    """
    shifted = raw_score - center
    shifted = max(clip_range[0], min(clip_range[1], shifted))

    if shifted >= 0:
        scaled_score = math.tanh(shifted / pos_temp)
    else:
        scaled_score = math.tanh(shifted / neg_temp)

    return scaled_score

def _calc_molecule_aosi(mol):
    """内部辅助函数：计算单个分子的表观氧化态指数(AOSI)"""
    aosi = 0
    # 定义通常比碳电负性高、导致碳氧化态升高的杂原子
    electronegative_heteroatoms = {'O', 'N', 'F', 'Cl', 'Br', 'I', 'S'}
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            hs = atom.GetTotalNumHs()
            hetero_bonds = 0
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() in electronegative_heteroatoms:
                    bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                    # 依据键的类型进行加权（单键=1, 双键=2, 三键=3）
                    order = bond.GetBondTypeAsDouble()
                    hetero_bonds += int(order)
            # 氧化态变化：连杂原子+1，连氢-1
            aosi += (hetero_bonds - hs)
    return aosi

def eval_redox_balance(r_mols, p_mol):
    """
    评估碳骨架氧化还原态平衡。
    严惩在没有明确氧化还原试剂供体情况下发生的氧化态骤变。
    """
    score = 0.0
    try:
        p_aosi = _calc_molecule_aosi(p_mol)
        r_aosi = sum(_calc_molecule_aosi(m) for m in r_mols)
        
        delta = abs(p_aosi - r_aosi)
        
        # 氧化态的变化往往跨度为 2 (例如醇脱氢变酮，-2H)。
        # 允许极小的容错(delta=1)，如果偏差大于等于 2，开始扣分
        if delta >= 2:
            # 偏差越大，意味着违背物理化学常识越严重
            score -= 0.15 * (delta // 2)
            
    except Exception:
        pass
        
    return score

def eval_chemoselectivity(r_mols, p_mol):
    """
    基于竞争位点的活性启发式规则，评价化学选择性。
    此版本着重于抓取最经典的错误：游离且活泼的脂肪胺未反应，而较迟钝的醇反应了。
    """
    score = 0.0
    
    # 定义基础官能团 SMARTS
    amine_pat = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')        # 活泼的脂肪胺（伯胺/仲胺）
    alcohol_pat = Chem.MolFromSmarts('[OX2H][CX4]')                # 脂肪醇
    ester_pat = Chem.MolFromSmarts('[CX3](=O)[OX2][#6]')           # 酯基
    amide_pat = Chem.MolFromSmarts('[CX3](=O)[NX3]')               # 酰胺基
    
    try:
        r_has_amine = any(m.HasSubstructMatch(amine_pat) for m in r_mols)
        p_has_amine = p_mol.HasSubstructMatch(amine_pat)
        p_has_ester = p_mol.HasSubstructMatch(ester_pat)
        
        # 核心逻辑：如果产物中形成了酯键，并且产物中保留了一个未被保护的活泼脂肪胺；
        # 同时反应物中也存在这个脂肪胺。
        # 这意味着模型预测了：在氨基醇与酸(或活化酸)的反应中，酸绕过了亲核性强得多的胺，去跟醇成酯了。
        # 这种没有任何保护手段的选择性在现实中是不能自发发生的！
        if p_has_ester and p_has_amine and r_has_amine:
            score -= 0.45  # 严重扣分（违背热力学/动力学常识）
            
    except Exception:
        pass
        
    return score

def evaluate_expert_reward(reactants_smi, product_smi):
    """
    综合评估逆合成反应物预测质量的专家奖励函数

    评分思想：
    1. 基础合法性：SMILES 非法直接判负
    2. 硬约束：严惩明显作弊/炼金术/危险结构
    3. 化学合理性：奖励骨架继承、官能团转换、元素守恒
    4. 策略合理性：奖励更易合成、保护基/BRICS 等逆合成特征
    """
    # 1. 解析与基础否决
    r_mols, p_mol, main_r_mol = parse_and_validate_molecules(reactants_smi, product_smi)
    if r_mols is None:
        return -1.0

    # 2. 所有子评分只计算一次
    # ---------------- A. 硬约束层 ----------------
    s_close_loop = eval_close_loop_consistency(r_mols, p_mol)
    s_pseudo_copy = eval_pseudo_retro_copy(main_r_mol, r_mols, p_mol)
    s_struct_alert = eval_structural_alerts(r_mols)
    s_heavy_hetero = eval_heavy_heteroatom_match(r_mols, p_mol)
    s_element_balance = eval_element_balance(r_mols, p_mol)
    s_redox_balance = eval_redox_balance(r_mols, p_mol)
    # ---------------- B. 化学合理性层 ----------------
    s_struct_sim = eval_structural_similarity(main_r_mol, p_mol)
    s_scaffold = eval_murcko_scaffold_overlap(main_r_mol, p_mol)
    s_fg_transform = eval_functional_group_transform(r_mols, p_mol)
    s_chem_stability = eval_chemical_stability(r_mols, p_mol)
    s_descriptor = eval_descriptor_delta(main_r_mol, p_mol)
    s_ring = eval_ring_stability(r_mols, p_mol)
    s_stereo = eval_stereochemistry_sanity(r_mols, p_mol)
    s_frag_size = eval_fragmentation_and_size(r_mols, p_mol)
    s_leaving_group = eval_leaving_group_quality(r_mols, p_mol)
    s_chemoselectivity = eval_chemoselectivity(r_mols, p_mol)
    # ---------------- C. 策略层 ----------------
    s_sa = eval_synthetic_accessibility(main_r_mol, p_mol)
    s_protect = eval_protecting_group_reasonableness(r_mols, p_mol)
    # s_brics = eval_brics_consistency(r_mols, p_mol)

    # 3. 基础分
    raw_score = 0.35

    # 4. 加权汇总
    raw_score += 1.8 * s_close_loop
    raw_score += 1.4 * s_pseudo_copy
    raw_score += 1.2 * s_struct_alert
    raw_score += 1.2 * s_heavy_hetero
    raw_score += 1.0 * s_element_balance
    raw_score += 1.2 * s_redox_balance

    raw_score += 1.3 * s_struct_sim
    raw_score += 1.0 * s_scaffold
    raw_score += 1.1 * s_fg_transform
    raw_score += 1.0 * s_chem_stability
    raw_score += 0.7 * s_descriptor
    raw_score += 0.6 * s_ring
    raw_score += 0.6 * s_stereo
    raw_score += 0.5 * s_frag_size
    raw_score += 0.5 * s_leaving_group
    raw_score += 1.2 * s_chemoselectivity

    raw_score += 0.8 * s_sa
    raw_score += 0.4 * s_protect
    # raw_score += 0.4 * s_brics

    # 5. 极端错误兜底：直接复用缓存结果，不重复调用函数
    severe_penalty = 0
    if s_close_loop < -0.5: severe_penalty += 1
    if s_struct_alert < -0.5: severe_penalty += 1
    if s_heavy_hetero < -0.5: severe_penalty += 1
    if s_element_balance < -0.5: severe_penalty += 1
    if s_pseudo_copy < -0.2: severe_penalty += 1
    if s_redox_balance < -0.4: severe_penalty += 1         
    if s_chemoselectivity < -0.4: severe_penalty += 1      

    if severe_penalty >= 3:
        raw_score -= 1.0
    elif severe_penalty == 2:
        raw_score -= 0.4

    # 6. 平滑映射
    final_reward = smooth_rl_reward(raw_score)
    return final_reward



def check_reaction_feasibility(pred_reactants_smi, target_product_smi):
    """
    增强版正向反应可行性验证器
    """
    # ==========================================
    # 1. 基础解析与碎片数量检查
    # ==========================================
    # 如果预测出的反应物多于 3 个 (包含超过 2 个 '.' )，极大概率是幻觉
    if pred_reactants_smi.count('.') > 2:
        return False
        
    try:
        reactants = Chem.MolFromSmiles(pred_reactants_smi)
        product = Chem.MolFromSmiles(target_product_smi)
    except:
        return False
        
    if reactants is None or product is None:
        return False

    # ==========================================
    # 2. 自由基检查 (Radical Check)
    # ==========================================
    # 除非产物本身是自由基，否则单步逆合成不应该凭空产生自由基反应物
    prod_radicals = sum([atom.GetNumRadicalElectrons() for atom in product.GetAtoms()])
    rxn_radicals = sum([atom.GetNumRadicalElectrons() for atom in reactants.GetAtoms()])
    if rxn_radicals > prod_radicals:
        return False

    # ==========================================
    # 3. 电荷守恒 (Charge Conservation)
    # ==========================================
    rxn_charge = sum([atom.GetFormalCharge() for atom in reactants.GetAtoms()])
    prod_charge = sum([atom.GetFormalCharge() for atom in product.GetAtoms()])
    if rxn_charge != prod_charge:
        return False

    # ==========================================
    # 4. 严格的核心元素守恒 (Strict Element Conservation)
    # ==========================================
    def get_element_counts(mol):
        return Counter([atom.GetSymbol() for atom in mol.GetAtoms()])
        
    rxn_counts = get_element_counts(reactants)
    prod_counts = get_element_counts(product)
    
    # 反应物必须能提供产物所需的所有骨架元素
    core_elements = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    for elem in core_elements:
        if prod_counts.get(elem, 0) > rxn_counts.get(elem, 0):
            return False

    # ==========================================
    # 5. 环数量巨变检查 (Ring Sanity)
    # ==========================================
    # 比如：产物有3个环，反应物只有0个环。这种单步反应是不现实的。
    rxn_rings = reactants.GetRingInfo().NumRings()
    prod_rings = product.GetRingInfo().NumRings()
    
    # 允许开环/关环反应（差异在1-2个之间），但如果差异过大则判定为错误
    if abs(rxn_rings - prod_rings) > 2:
        return False

    # ==========================================
    # 7. 重原子数量骤增限制
    # ==========================================
    rxn_heavy = reactants.GetNumHeavyAtoms()
    prod_heavy = product.GetNumHeavyAtoms()
    
    if rxn_heavy > prod_heavy + 10: # 比之前更严格，通常不会多出10个重原子
        return False

    # 如果通过了这 7 道极其严苛的关卡，说明它绝对是一个在化学上高度合理的候选路线
    return True


# --- 测试示例 ---
if __name__ == "__main__":
    # 正常反应 (酯水解)
    p = "CCOC(=O)C"
    r = "CC(=O)O.CCO"
    print(f"合理反应得分: {evaluate_expert_reward(r, p):.3f}")

    # 异常反应 (复制粘贴)
    print(f"复制粘贴得分: {evaluate_expert_reward(p, p):.3f}")

    # 异常反应 (炼金术: 碳变氟)
    print(f"变出氟原子得分: {evaluate_expert_reward('F', 'c1ccccc1'):.3f}")