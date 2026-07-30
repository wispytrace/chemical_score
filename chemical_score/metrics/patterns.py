"""SMARTS queries compiled once and shared by all metrics."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


def compile_smarts(smarts: str) -> Chem.Mol:
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        raise RuntimeError(f"invalid built-in SMARTS pattern: {smarts}")
    return pattern


FUNCTIONAL_GROUPS = {
    "ester": compile_smarts("[CX3](=O)[OX2][#6]"),
    "carboxylic_acid": compile_smarts("[CX3](=O)[OX2H1]"),
    "alcohol": compile_smarts("[OX2H][#6]"),
    "acyl_halide": compile_smarts("[CX3](=O)[Cl,Br]"),
    "amide": compile_smarts("[CX3](=O)[NX3]"),
    "amine": compile_smarts("[NX3;H2,H1;!$(NC=O)]"),
    "ether": compile_smarts("[OD2]([#6])[#6]"),
    "aryl_halide": compile_smarts("[c][Cl,Br,I]"),
    "boronic_acid_or_ester": compile_smarts("[B]([O])[O]"),
    "sulfonate": compile_smarts("[O]S(=O)(=O)[#6]"),
}


PROTECTING_GROUPS = {
    "Boc": compile_smarts("CC(C)(C)OC(=O)"),
    "Cbz": compile_smarts("O=C(OCc1ccccc1)"),
    "tosyl": compile_smarts("S(=O)(=O)c1ccc(C)cc1"),
    "mesyl": compile_smarts("CS(=O)(=O)"),
    "Bpin_like": compile_smarts("B1OC(C)(C)C(C)(C)O1"),
    "silyl_like": compile_smarts("[Si]([C])([C])[C]"),
}


LEAVING_GROUPS = {
    "bromide": compile_smarts("[C,c]-Br"),
    "iodide": compile_smarts("[C,c]-I"),
    "chloride": compile_smarts("[C,c]-Cl"),
    "fluoride": compile_smarts("[C,c]-F"),
    "tosylate": compile_smarts("[O]S(=O)(=O)c1ccc(C)cc1"),
    "mesylate": compile_smarts("[O]S(=O)(=O)C"),
    "acyl_halide": compile_smarts("[CX3](=O)[Cl,Br]"),
}


@dataclass(frozen=True, slots=True)
class AlertPattern:
    id: str
    name: str
    smarts: str
    penalty: float
    pattern: Chem.Mol


def _alert(id_: str, name: str, smarts: str, penalty: float) -> AlertPattern:
    return AlertPattern(id_, name, smarts, penalty, compile_smarts(smarts))


STRUCTURAL_ALERTS = (
    _alert("adjacent_anions", "Adjacent anions", "[*-]-[*-]", 60),
    _alert("peroxide", "Peroxide bond", "[OX2]-[OX2]", 25),
    _alert("hetero_chain", "Long O/S heteroatom chain", "[O,S]-[O,S]-[O,S]", 35),
    _alert("azide", "Azide-like group", "[$([N-]=[N+]=N),$([N]#[N+][N-])]", 25),
    _alert("nitrogen_chain", "Long nitrogen chain", "[N]-[N]-[N]-[N]", 35),
    _alert(
        "strained_alkyne", "Highly strained small-ring alkyne", "[#6]1-[#6]-[#6]1#*", 45
    ),
)
