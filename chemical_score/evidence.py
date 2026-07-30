"""Historical-reaction evidence index and reaction-to-corpus comparison.

The index intentionally uses transparent RDKit reaction fingerprints rather than
a trained model.  Scores therefore describe support in the supplied corpus, not
an intrinsic probability that a reaction will work.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdChemReactions

from chemical_score.context import ReactionContext, ReactionInputError


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One historical reaction and its optional observed outcome."""

    reaction_smiles: str
    identifier: str | None = None
    success: bool | None = None
    yield_percent: float | None = None
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reaction_smiles.strip():
            raise ValueError("evidence reaction_smiles must not be empty")
        if self.yield_percent is not None and not 0 <= self.yield_percent <= 100:
            raise ValueError("yield_percent must be between 0 and 100")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvidenceRecord:
        identifier = value.get("identifier", value.get("id"))
        yield_value = value.get("yield_percent", value.get("yield"))
        success = value.get("success")
        if success is not None and not isinstance(success, bool):
            raise ValueError("success must be true, false, or null")
        metadata = value.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be an object")
        return cls(
            reaction_smiles=str(value.get("reaction_smiles", "")),
            identifier=str(identifier) if identifier is not None else None,
            success=success,
            yield_percent=float(yield_value) if yield_value is not None else None,
            source=str(value["source"]) if value.get("source") is not None else None,
            metadata=dict(metadata),
        )


@dataclass(frozen=True, slots=True)
class EvidenceNeighbor:
    identifier: str | None
    reaction_smiles: str
    similarity: float
    success: bool | None
    yield_percent: float | None
    source: str | None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "identifier": self.identifier,
            "reaction_smiles": self.reaction_smiles,
            "similarity": round(self.similarity, 4),
        }
        if self.success is not None:
            result["success"] = self.success
        if self.yield_percent is not None:
            result["yield_percent"] = self.yield_percent
        if self.source is not None:
            result["source"] = self.source
        return result


@dataclass(frozen=True, slots=True)
class EvidenceComparison:
    index_size: int
    top_similarity: float
    neighbor_count: int
    exact_count: int
    transformation_signature: str | None
    transformation_count: int | None
    outcome_score: float | None
    outcome_count: int
    outcome_basis: dict[str, int]
    similarity_threshold: float
    neighbors: tuple[EvidenceNeighbor, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "index_size": self.index_size,
            "top_similarity": round(self.top_similarity, 4),
            "neighbor_count": self.neighbor_count,
            "exact_count": self.exact_count,
            "transformation_signature": self.transformation_signature,
            "transformation_count": self.transformation_count,
            "outcome_score": (
                round(self.outcome_score, 4) if self.outcome_score is not None else None
            ),
            "outcome_count": self.outcome_count,
            "outcome_basis": self.outcome_basis,
            "similarity_threshold": self.similarity_threshold,
            "neighbors": [neighbor.to_dict() for neighbor in self.neighbors],
        }


@dataclass(frozen=True, slots=True)
class _IndexedRecord:
    record: EvidenceRecord
    fingerprint: Any
    canonical_key: str
    transformation_signature: str | None


class EvidenceIndex:
    """Immutable, reusable index of historical reaction evidence."""

    def __init__(
        self,
        records: Iterable[EvidenceRecord | Mapping[str, Any]],
        *,
        similarity_threshold: float = 0.5,
        top_k: int = 5,
        strict: bool = True,
    ) -> None:
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.similarity_threshold = float(similarity_threshold)
        self.top_k = int(top_k)
        indexed: list[_IndexedRecord] = []
        rejected: list[str] = []
        for position, value in enumerate(records):
            try:
                if isinstance(value, EvidenceRecord):
                    record = value
                elif isinstance(value, Mapping):
                    record = EvidenceRecord.from_mapping(value)
                else:
                    raise TypeError("record must be an EvidenceRecord or object")
                indexed.append(self._prepare_record(record))
            except (TypeError, ValueError, ReactionInputError) as exc:
                message = f"record {position}: {exc}"
                if strict:
                    raise ValueError(message) from exc
                rejected.append(message)
        self._records = tuple(indexed)
        self._fingerprints = tuple(item.fingerprint for item in indexed)
        self._exact_counts = Counter(item.canonical_key for item in indexed)
        self._transformation_counts = Counter(
            item.transformation_signature
            for item in indexed
            if item.transformation_signature is not None
        )
        self._rejected = tuple(rejected)

    def __len__(self) -> int:
        return len(self._records)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        similarity_threshold: float = 0.5,
        top_k: int = 5,
        strict: bool = True,
    ) -> EvidenceIndex:
        """Load a JSON array/object or newline-delimited JSON evidence file."""

        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8")
        if file_path.suffix.lower() == ".jsonl":
            records = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        else:
            payload = json.loads(text)
            records = (
                payload.get("records", []) if isinstance(payload, dict) else payload
            )
        if not isinstance(records, list):
            raise TypeError("evidence file must contain a JSON array or records array")
        return cls(
            records,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            strict=strict,
        )

    def status(self) -> dict[str, Any]:
        return {
            "configured": True,
            "record_count": len(self),
            "rejected_count": len(self._rejected),
            "rejected": list(self._rejected),
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k,
            "fingerprint": "RDKit difference fingerprint",
        }

    def compare(self, context: ReactionContext) -> EvidenceComparison:
        if not self._records:
            raise ValueError("evidence index is empty")
        fingerprint = _reaction_fingerprint(context)
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint, list(self._fingerprints)
        )
        nearest = heapq.nlargest(
            min(self.top_k, len(similarities)),
            enumerate(similarities),
            key=lambda item: item[1],
        )
        neighbors = tuple(
            EvidenceNeighbor(
                identifier=self._records[index].record.identifier,
                reaction_smiles=self._records[index].record.reaction_smiles,
                similarity=float(similarity),
                success=self._records[index].record.success,
                yield_percent=self._records[index].record.yield_percent,
                source=self._records[index].record.source,
            )
            for index, similarity in nearest
        )
        qualifying = [
            (self._records[index].record, float(similarity))
            for index, similarity in enumerate(similarities)
            if similarity >= self.similarity_threshold
        ]
        outcome_values: list[tuple[float, float, str]] = []
        for record, similarity in qualifying:
            if record.yield_percent is not None:
                outcome_values.append((record.yield_percent, similarity, "yield"))
            elif record.success is not None:
                outcome_values.append(
                    (100.0 if record.success else 0.0, similarity, "success")
                )
        total_outcome_weight = sum(similarity for _, similarity, _ in outcome_values)
        outcome_score = (
            sum(value * similarity for value, similarity, _ in outcome_values)
            / total_outcome_weight
            if total_outcome_weight > 0
            else None
        )
        basis = Counter(kind for _, _, kind in outcome_values)
        signature = transformation_signature(
            context.reactant_mols, context.product_mols
        )
        return EvidenceComparison(
            index_size=len(self),
            top_similarity=float(nearest[0][1]),
            neighbor_count=len(qualifying),
            exact_count=self._exact_counts[canonical_reaction_key(context)],
            transformation_signature=signature,
            transformation_count=(
                self._transformation_counts[signature]
                if signature is not None
                else None
            ),
            outcome_score=outcome_score,
            outcome_count=len(outcome_values),
            outcome_basis=dict(basis),
            similarity_threshold=self.similarity_threshold,
            neighbors=neighbors,
        )

    @staticmethod
    def _prepare_record(record: EvidenceRecord) -> _IndexedRecord:
        reactants, products, agents = _split_reaction_smiles(record.reaction_smiles)
        context = ReactionContext(reactants, products, agents)
        return _IndexedRecord(
            record=record,
            fingerprint=_reaction_fingerprint(context),
            canonical_key=canonical_reaction_key(context),
            transformation_signature=transformation_signature(
                context.reactant_mols, context.product_mols
            ),
        )


def canonical_reaction_key(context: ReactionContext) -> str:
    """Canonical key insensitive to component order and atom-map numbers."""

    reactants = ".".join(sorted(context.canonical_reactants))
    products = ".".join(sorted(context.canonical_products))
    return f"{reactants}>>{products}"


def transformation_signature(
    reactants: Iterable[Chem.Mol], products: Iterable[Chem.Mol]
) -> str | None:
    """Return a map-number-independent signature of mapped bond edits."""

    left = _mapped_bonds(reactants)
    right = _mapped_bonds(products)
    if left is None or right is None:
        return None
    edits: list[str] = []
    for atom_pair in sorted(set(left) | set(right)):
        before = left.get(atom_pair)
        after = right.get(atom_pair)
        if before is None and after is not None:
            edits.append(f"formed:{after[0]}:{_format_order(after[1])}")
        elif before is not None and after is None:
            edits.append(f"broken:{before[0]}:{_format_order(before[1])}")
        elif before is not None and after is not None and before[1] != after[1]:
            elements = before[0] if before[0] == after[0] else f"{before[0]}>{after[0]}"
            edits.append(
                f"changed:{elements}:{_format_order(before[1])}>"
                f"{_format_order(after[1])}"
            )
    return "|".join(sorted(edits)) or None


def _mapped_bonds(
    molecules: Iterable[Chem.Mol],
) -> dict[tuple[int, int], tuple[str, float]] | None:
    atom_elements: dict[int, str] = {}
    molecule_list = list(molecules)
    for molecule in molecule_list:
        for atom in molecule.GetAtoms():
            map_number = atom.GetAtomMapNum()
            if map_number <= 0:
                continue
            if map_number in atom_elements:
                return None
            atom_elements[map_number] = atom.GetSymbol()
    if not atom_elements:
        return None
    bonds: dict[tuple[int, int], tuple[str, float]] = {}
    for molecule in molecule_list:
        for bond in molecule.GetBonds():
            begin = bond.GetBeginAtom().GetAtomMapNum()
            end = bond.GetEndAtom().GetAtomMapNum()
            if begin <= 0 or end <= 0:
                continue
            key = tuple(sorted((begin, end)))
            elements = "-".join(sorted((atom_elements[begin], atom_elements[end])))
            bonds[key] = (elements, float(bond.GetBondTypeAsDouble()))
    return bonds


def _format_order(order: float) -> str:
    return str(int(order)) if order.is_integer() else str(order)


def _reaction_fingerprint(context: ReactionContext):
    reaction_smiles = (
        f"{'.'.join(context.canonical_reactants)}>>"
        f"{'.'.join(context.canonical_products)}"
    )
    reaction = rdChemReactions.ReactionFromSmarts(reaction_smiles, useSmiles=True)
    if reaction is None:
        raise ValueError("RDKit could not build a reaction fingerprint")
    return rdChemReactions.CreateDifferenceFingerprintForReaction(reaction)


def _split_reaction_smiles(value: str) -> tuple[str, str, str | None]:
    parts = value.split(">")
    if len(parts) != 3:
        raise ReactionInputError(
            "reaction_smiles must use 'reactants>agents>products' format"
        )
    reactants, agents, products = (part.strip() for part in parts)
    if not reactants or not products:
        raise ReactionInputError("reaction_smiles requires reactants and products")
    return reactants, products, agents or None
