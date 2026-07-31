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

    @cached_property
    def mapping_analysis(self) -> dict[str, Any]:
        """Summarize optional atom mapping and exact mapped bond changes.

        Unmapped reaction SMILES remain fully valid. Mapping quality is reported
        separately so missing metadata is never mistaken for bad chemistry.
        """

        reactant_atoms = [
            atom
            for molecule in self.reactant_mols
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() > 1
        ]
        product_atoms = [
            atom for atom in self.product_mol.GetAtoms() if atom.GetAtomicNum() > 1
        ]
        reactant_maps = Counter(
            atom.GetAtomMapNum() for atom in reactant_atoms if atom.GetAtomMapNum() > 0
        )
        product_maps = Counter(
            atom.GetAtomMapNum() for atom in product_atoms if atom.GetAtomMapNum() > 0
        )
        present = bool(reactant_maps or product_maps)
        duplicate_maps = sorted(
            map_number
            for map_number, count in (reactant_maps + product_maps).items()
            if reactant_maps[map_number] > 1 or product_maps[map_number] > 1
        )
        reactant_map_set = set(reactant_maps)
        product_map_set = set(product_maps)
        reactant_elements = {
            atom.GetAtomMapNum(): atom.GetSymbol()
            for atom in reactant_atoms
            if atom.GetAtomMapNum() > 0 and reactant_maps[atom.GetAtomMapNum()] == 1
        }
        product_elements = {
            atom.GetAtomMapNum(): atom.GetSymbol()
            for atom in product_atoms
            if atom.GetAtomMapNum() > 0 and product_maps[atom.GetAtomMapNum()] == 1
        }
        element_mismatches = [
            {
                "map_number": map_number,
                "reactant_element": reactant_elements[map_number],
                "product_element": product_elements[map_number],
            }
            for map_number in sorted(reactant_map_set & product_map_set)
            if map_number in reactant_elements
            and map_number in product_elements
            and reactant_elements[map_number] != product_elements[map_number]
        ]
        traceable_product_atoms = sum(
            1
            for atom in product_atoms
            if atom.GetAtomMapNum() > 0
            and atom.GetAtomMapNum() in reactant_map_set
            and reactant_maps[atom.GetAtomMapNum()] == 1
        )
        result: dict[str, Any] = {
            "present": present,
            "reactant_coverage": (
                len(reactant_maps) / len(reactant_atoms) if reactant_atoms else 0.0
            ),
            "product_coverage": (
                len(product_maps) / len(product_atoms) if product_atoms else 0.0
            ),
            "traceable_product_fraction": (
                traceable_product_atoms / len(product_atoms) if product_atoms else 0.0
            ),
            "duplicate_map_numbers": duplicate_maps,
            "element_mismatches": element_mismatches,
            "product_maps_missing_from_reactants": sorted(
                product_map_set - reactant_map_set
            ),
            "bond_changes": None,
        }
        if not present or duplicate_maps or element_mismatches:
            return result

        def mapped_bonds(molecules: list[Chem.Mol]):
            bonds: dict[tuple[int, int], tuple[float, str]] = {}
            for molecule in molecules:
                for bond in molecule.GetBonds():
                    begin_atom = bond.GetBeginAtom()
                    end_atom = bond.GetEndAtom()
                    begin = begin_atom.GetAtomMapNum()
                    end = end_atom.GetAtomMapNum()
                    if (
                        begin <= 0
                        or end <= 0
                        or begin_atom.GetAtomicNum() <= 1
                        or end_atom.GetAtomicNum() <= 1
                    ):
                        continue
                    elements = "-".join(
                        sorted((begin_atom.GetSymbol(), end_atom.GetSymbol()))
                    )
                    bonds[tuple(sorted((begin, end)))] = (
                        float(bond.GetBondTypeAsDouble()),
                        elements,
                    )
            return bonds

        before = mapped_bonds(self.reactant_mols)
        after = mapped_bonds([self.product_mol])
        changes: list[dict[str, Any]] = []
        for atom_pair in sorted(set(before) | set(after)):
            left = before.get(atom_pair)
            right = after.get(atom_pair)
            if left is None and right is not None:
                changes.append(
                    {
                        "type": "formed",
                        "atom_maps": list(atom_pair),
                        "elements": right[1],
                        "before_order": None,
                        "after_order": right[0],
                    }
                )
            elif left is not None and right is None:
                changes.append(
                    {
                        "type": "broken",
                        "atom_maps": list(atom_pair),
                        "elements": left[1],
                        "before_order": left[0],
                        "after_order": None,
                    }
                )
            elif left is not None and right is not None and left[0] != right[0]:
                changes.append(
                    {
                        "type": "order_changed",
                        "atom_maps": list(atom_pair),
                        "elements": left[1],
                        "before_order": left[0],
                        "after_order": right[0],
                    }
                )
        result["bond_changes"] = changes
        return result

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
            "input_quality": {
                "atom_mapping": {
                    key: value
                    for key, value in self.mapping_analysis.items()
                    if key != "bond_changes"
                },
                "reactant_component_count": len(self.reactant_mols),
                "product_component_count": len(self.product_mols),
                "agents_separated": bool(self.agent_mols),
            },
            "warnings": warnings,
        }
