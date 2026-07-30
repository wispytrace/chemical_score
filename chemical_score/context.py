"""Strict reaction parsing and cached RDKit features."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from functools import cached_property
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


class ReactionInputError(ValueError):
    """Raised when a reaction cannot be parsed without losing input data."""


class ReactionContext:
    """Parsed reaction plus lazily cached molecular features.

    Metrics share one context, preventing repeated fingerprints, scaffolds and
    descriptor calculations during a single evaluation.
    """

    _fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048
    )

    def __init__(
        self,
        reactants_smiles: str,
        product_smiles: str,
        agents_smiles: str | None = None,
        resources: dict[str, object] | None = None,
    ) -> None:
        self.input_reactants_smiles = reactants_smiles
        self.input_product_smiles = product_smiles
        self.input_agents_smiles = agents_smiles or ""
        self.reactant_mols = self._parse_components(reactants_smiles, "reactants")
        self.product_mols = self._parse_components(product_smiles, "products")
        self.agent_mols = (
            self._parse_components(agents_smiles, "agents") if agents_smiles else []
        )
        self.resources = dict(resources or {})
        self.product_mol = max(
            self.product_mols,
            key=lambda mol: (mol.GetNumHeavyAtoms(), Descriptors.MolWt(mol)),
        )
        self._fingerprints: dict[int, object] = {}
        self._scaffolds: dict[int, Chem.Mol] = {}
        self._descriptors: dict[int, dict[str, float]] = {}
        self._computed: dict[str, Any] = {}

    def get_resource(self, name: str) -> object | None:
        """Return an evaluator-provided shared resource, if configured."""

        return self.resources.get(name)

    def memoize(self, key: str, factory: Callable[[], Any]) -> Any:
        """Compute an expensive cross-metric value at most once per reaction."""

        if key not in self._computed:
            self._computed[key] = factory()
        return self._computed[key]

    @staticmethod
    def _parse_components(smiles: str | None, field: str) -> list[Chem.Mol]:
        if not smiles or not smiles.strip():
            raise ReactionInputError(f"{field} SMILES must not be empty")
        components = [item.strip() for item in smiles.split(".")]
        if any(not item for item in components):
            raise ReactionInputError(f"{field} SMILES contains an empty component")
        molecules: list[Chem.Mol] = []
        for index, component in enumerate(components):
            molecule = Chem.MolFromSmiles(component)
            if molecule is None:
                raise ReactionInputError(
                    f"invalid {field} SMILES component at index {index}: {component!r}"
                )
            molecules.append(molecule)
        return molecules

    @cached_property
    def main_reactant(self) -> Chem.Mol:
        product_fp = self.fingerprint(self.product_mol)
        product_scaffold = self.scaffold(self.product_mol)
        best: tuple[float, Chem.Mol] | None = None
        for molecule in self.reactant_mols:
            similarity = DataStructs.TanimotoSimilarity(
                self.fingerprint(molecule), product_fp
            )
            score = 1.5 * similarity + 0.02 * min(molecule.GetNumHeavyAtoms(), 30)
            scaffold = self.scaffold(molecule)
            if (
                scaffold.GetNumHeavyAtoms()
                and product_scaffold.GetNumHeavyAtoms()
                and (
                    scaffold.HasSubstructMatch(product_scaffold)
                    or product_scaffold.HasSubstructMatch(scaffold)
                )
            ):
                score += 0.2
            if best is None or score > best[0]:
                best = (score, molecule)
        assert best is not None
        return best[1]

    @cached_property
    def canonical_reactants(self) -> list[str]:
        return [self.chemical_smiles(mol) for mol in self.reactant_mols]

    @cached_property
    def canonical_products(self) -> list[str]:
        return [self.chemical_smiles(mol) for mol in self.product_mols]

    @cached_property
    def canonical_agents(self) -> list[str]:
        return [self.chemical_smiles(mol) for mol in self.agent_mols]

    @cached_property
    def canonical_main_product(self) -> str:
        return self.chemical_smiles(self.product_mol)

    @cached_property
    def canonical_main_reactant(self) -> str:
        return self.chemical_smiles(self.main_reactant)

    @staticmethod
    def chemical_smiles(molecule: Chem.Mol) -> str:
        """Canonical SMILES without atom-map annotations used only as metadata."""

        copy = Chem.Mol(molecule)
        for atom in copy.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(copy, canonical=True)

    @cached_property
    def reactant_atom_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for molecule in self.reactant_mols:
            counts.update(atom.GetSymbol() for atom in molecule.GetAtoms())
        return counts

    @cached_property
    def product_atom_counts(self) -> Counter[str]:
        return Counter(atom.GetSymbol() for atom in self.product_mol.GetAtoms())

    @cached_property
    def main_similarity(self) -> float:
        return float(
            DataStructs.TanimotoSimilarity(
                self.fingerprint(self.main_reactant),
                self.fingerprint(self.product_mol),
            )
        )

    def fingerprint(self, molecule: Chem.Mol):
        key = id(molecule)
        if key not in self._fingerprints:
            self._fingerprints[key] = self._fingerprint_generator.GetFingerprint(
                molecule
            )
        return self._fingerprints[key]

    def scaffold(self, molecule: Chem.Mol) -> Chem.Mol:
        key = id(molecule)
        if key not in self._scaffolds:
            self._scaffolds[key] = MurckoScaffold.GetScaffoldForMol(molecule)
        return self._scaffolds[key]

    def descriptors(self, molecule: Chem.Mol) -> dict[str, float]:
        key = id(molecule)
        if key not in self._descriptors:
            self._descriptors[key] = {
                "rings": float(rdMolDescriptors.CalcNumRings(molecule)),
                "aromatic_rings": float(
                    rdMolDescriptors.CalcNumAromaticRings(molecule)
                ),
                "heteroatoms": float(rdMolDescriptors.CalcNumHeteroatoms(molecule)),
                "rotatable_bonds": float(
                    rdMolDescriptors.CalcNumRotatableBonds(molecule)
                ),
                "hba": float(rdMolDescriptors.CalcNumHBA(molecule)),
                "hbd": float(rdMolDescriptors.CalcNumHBD(molecule)),
                "molecular_weight": float(Descriptors.MolWt(molecule)),
                "heavy_atoms": float(molecule.GetNumHeavyAtoms()),
            }
        return self._descriptors[key]

    def reaction_dict(self) -> dict[str, object]:
        warnings: list[str] = []
        if len(self.product_mols) > 1:
            warnings.append(
                "multiple product components supplied; metrics use the largest component"
            )
        if self.agent_mols:
            warnings.append(
                "agents are screened for structural alerts but excluded from element "
                "conservation and material-efficiency estimates"
            )
        return {
            "reactants_smiles": ".".join(self.canonical_reactants),
            "agents_smiles": ".".join(self.canonical_agents) or None,
            "product_smiles": ".".join(self.canonical_products),
            "main_reactant_smiles": self.canonical_main_reactant,
            "main_product_smiles": self.canonical_main_product,
            "warnings": warnings,
        }
