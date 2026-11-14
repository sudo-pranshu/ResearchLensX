"""Knowledge graph builder. STATUS: not implemented.

Spec: build(documents, verifications) -> contracts.KnowledgeGraph
  nodes: paper (doc_id), section (f"{doc_id}::s{i}"), entity (f"ent::{type}::{normalised name}"), claim (claim_id)
  edges: has_section, mentions (section->entity, weight=count), shares_entity (paper<->paper, weight=#shared),
         supports/contradicts (claim->paper, from VerificationResult evidence with label != INSUFFICIENT)
"""
from researchlens.contracts import Document, KnowledgeGraph, VerificationResult


class GraphBuilder:
    def build(self, documents: list[Document], verifications: list[VerificationResult] | None = None) -> KnowledgeGraph:
        raise NotImplementedError("G4: implement per docstring spec")
