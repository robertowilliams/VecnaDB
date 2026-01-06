"""
VecnaDB Answer Grounding

Provides complete provenance tracking for RAG answers.
Every claim in an answer must be traceable to source entities in the knowledge graph.

Grounding Process:
1. Parse answer into atomic claims
2. Match each claim to source entities
3. Trace claim to specific properties/relations
4. Build provenance graph
5. Generate citations
6. Create human-readable explanations

Principle: Every claim must have a traceable path to authoritative sources.
"""

import time
from typing import List, Optional, Dict, Any, Set, Tuple
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field

from vecnadb.infrastructure.engine.models.KnowledgeEntity import KnowledgeEntity
from vecnadb.infrastructure.storage.VecnaDBStorageInterface import (
    VecnaDBStorageInterface,
    Relation,
)
from vecnadb.modules.ontology.models.OntologySchema import OntologySchema


class CitationStyle(str, Enum):
    """Citation format style"""
    INLINE = "inline"  # [1], [2], etc.
    SUPERSCRIPT = "superscript"  # ^1, ^2, etc.
    FOOTNOTE = "footnote"  # Footnote markers
    PARENTHETICAL = "parenthetical"  # (Source: X)


class ProvenanceType(str, Enum):
    """Type of provenance"""
    DIRECT_QUOTE = "direct_quote"  # Direct quote from entity
    PARAPHRASE = "paraphrase"  # Paraphrased from entity
    SYNTHESIS = "synthesis"  # Synthesized from multiple entities
    INFERENCE = "inference"  # Inferred via reasoning
    AGGREGATION = "aggregation"  # Aggregated from multiple sources


class ProvenanceLink(BaseModel):
    """Link from claim to source entity"""
    source_entity_id: UUID
    property_name: Optional[str] = None  # Specific property used
    relation_id: Optional[UUID] = None  # Specific relation used
    provenance_type: ProvenanceType
    confidence: float  # 0.0-1.0
    explanation: str


class GroundedClaim(BaseModel):
    """A claim with full provenance"""
    claim_id: str  # Unique claim ID
    claim_text: str
    start_pos: int
    end_pos: int
    provenance_links: List[ProvenanceLink]
    is_grounded: bool
    confidence: float
    citation_ids: List[str] = Field(default_factory=list)


class Citation(BaseModel):
    """A citation for a source entity"""
    citation_id: str
    entity_id: UUID
    entity_type: str
    display_text: str  # How to display this citation
    full_reference: str  # Full reference information
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProvenanceGraph(BaseModel):
    """Complete provenance graph for an answer"""
    answer_text: str
    grounded_claims: List[GroundedClaim]
    citations: List[Citation]
    provenance_coverage: float  # % of answer that is grounded
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_ungrounded_claims(self) -> List[GroundedClaim]:
        """Get claims without provenance"""
        return [c for c in self.grounded_claims if not c.is_grounded]

    def get_sources_used(self) -> List[UUID]:
        """Get all source entity IDs"""
        sources = set()
        for claim in self.grounded_claims:
            for link in claim.provenance_links:
                sources.add(link.source_entity_id)
        return list(sources)


class GroundedAnswer(BaseModel):
    """Answer with complete grounding and citations"""
    original_answer: str
    annotated_answer: str  # Answer with citation markers
    provenance_graph: ProvenanceGraph
    citations_text: str  # Formatted citations
    citation_style: CitationStyle
    fully_grounded: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnswerGrounding:
    """
    Provides complete provenance tracking for RAG answers.

    Ensures every claim is traceable to source entities with full explanation
    of how the claim was derived.
    """

    def __init__(
        self,
        storage: VecnaDBStorageInterface,
        ontology: OntologySchema
    ):
        """
        Initialize answer grounding.

        Args:
            storage: VecnaDB storage interface
            ontology: Ontology schema
        """
        self.storage = storage
        self.ontology = ontology

    async def ground_answer(
        self,
        answer_text: str,
        context_entities: List[KnowledgeEntity],
        citation_style: CitationStyle = CitationStyle.INLINE
    ) -> GroundedAnswer:
        """
        Ground answer with full provenance and citations.

        Args:
            answer_text: Generated answer text
            context_entities: Context entities used
            citation_style: Citation format style

        Returns:
            GroundedAnswer with provenance and citations
        """
        start_time = time.time()

        # 1. Parse answer into claims
        claims = self._parse_into_claims(answer_text)

        # 2. Ground each claim
        grounded_claims = []
        for i, claim_text in enumerate(claims):
            grounded_claim = await self._ground_claim(
                claim_id=f"claim_{i}",
                claim_text=claim_text,
                context_entities=context_entities
            )
            grounded_claims.append(grounded_claim)

        # 3. Build citations
        citations = self._build_citations(grounded_claims, context_entities)

        # 4. Calculate provenance coverage
        coverage = self._calculate_coverage(grounded_claims)

        # 5. Build provenance graph
        provenance_graph = ProvenanceGraph(
            answer_text=answer_text,
            grounded_claims=grounded_claims,
            citations=citations,
            provenance_coverage=coverage,
            metadata={
                "total_claims": len(grounded_claims),
                "grounded_claims": sum(1 for c in grounded_claims if c.is_grounded),
                "context_entities": len(context_entities)
            }
        )

        # 6. Annotate answer with citations
        annotated_answer = self._annotate_with_citations(
            answer_text,
            grounded_claims,
            citation_style
        )

        # 7. Format citations
        citations_text = self._format_citations(citations, citation_style)

        # 8. Determine if fully grounded
        fully_grounded = all(c.is_grounded for c in grounded_claims)

        grounded_answer = GroundedAnswer(
            original_answer=answer_text,
            annotated_answer=annotated_answer,
            provenance_graph=provenance_graph,
            citations_text=citations_text,
            citation_style=citation_style,
            fully_grounded=fully_grounded,
            metadata={
                "grounding_time_ms": (time.time() - start_time) * 1000,
                "provenance_coverage": coverage
            }
        )

        return grounded_answer

    def _parse_into_claims(self, text: str) -> List[str]:
        """
        Parse answer text into atomic claims.

        Simple implementation: split by sentences.
        Real implementation would use NLP for claim extraction.

        Args:
            text: Answer text

        Returns:
            List of claim strings
        """
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    async def _ground_claim(
        self,
        claim_id: str,
        claim_text: str,
        context_entities: List[KnowledgeEntity]
    ) -> GroundedClaim:
        """
        Ground a single claim in context entities.

        Args:
            claim_id: Unique claim ID
            claim_text: Claim text
            context_entities: Available context

        Returns:
            GroundedClaim with provenance links
        """
        provenance_links = []

        # Try to match claim to each context entity
        for entity in context_entities:
            links = await self._find_provenance_links(claim_text, entity)
            provenance_links.extend(links)

        # Calculate claim grounding
        is_grounded = len(provenance_links) > 0

        confidence = (
            max(link.confidence for link in provenance_links)
            if provenance_links else 0.0
        )

        # Assign citation IDs
        citation_ids = [
            f"cite_{link.source_entity_id}"
            for link in provenance_links
        ]

        grounded_claim = GroundedClaim(
            claim_id=claim_id,
            claim_text=claim_text,
            start_pos=0,  # Would calculate actual position
            end_pos=len(claim_text),
            provenance_links=provenance_links,
            is_grounded=is_grounded,
            confidence=confidence,
            citation_ids=list(set(citation_ids))  # Unique IDs
        )

        return grounded_claim

    async def _find_provenance_links(
        self,
        claim_text: str,
        entity: KnowledgeEntity
    ) -> List[ProvenanceLink]:
        """
        Find provenance links from claim to entity.

        Args:
            claim_text: Claim text
            entity: Potential source entity

        Returns:
            List of provenance links
        """
        links = []
        claim_lower = claim_text.lower()

        # Check content property
        content = entity.properties.get('content', '')
        if isinstance(content, str) and content:
            content_lower = content.lower()

            # Check for direct quote
            if claim_text in content or content in claim_text:
                links.append(ProvenanceLink(
                    source_entity_id=entity.id,
                    property_name="content",
                    provenance_type=ProvenanceType.DIRECT_QUOTE,
                    confidence=0.95,
                    explanation=f"Claim is a direct quote from entity content"
                ))

            # Check for paraphrase (word overlap)
            elif self._has_significant_overlap(claim_lower, content_lower):
                links.append(ProvenanceLink(
                    source_entity_id=entity.id,
                    property_name="content",
                    provenance_type=ProvenanceType.PARAPHRASE,
                    confidence=0.75,
                    explanation=f"Claim paraphrases entity content"
                ))

        # Check name/title mention
        name = entity.properties.get('name', '')
        if isinstance(name, str) and name.lower() in claim_lower:
            links.append(ProvenanceLink(
                source_entity_id=entity.id,
                property_name="name",
                provenance_type=ProvenanceType.PARAPHRASE,
                confidence=0.6,
                explanation=f"Claim mentions entity '{name}'"
            ))

        return links

    def _has_significant_overlap(self, text1: str, text2: str) -> bool:
        """Check if two texts have significant word overlap"""
        words1 = set(text1.split())
        words2 = set(text2.split())

        overlap = words1.intersection(words2)

        # Require >40% overlap
        return len(overlap) > max(len(words1), len(words2)) * 0.4

    def _build_citations(
        self,
        grounded_claims: List[GroundedClaim],
        context_entities: List[KnowledgeEntity]
    ) -> List[Citation]:
        """
        Build citations for all sources used.

        Args:
            grounded_claims: All grounded claims
            context_entities: Context entities

        Returns:
            List of citations
        """
        # Collect all unique source entities
        source_ids = set()
        for claim in grounded_claims:
            for link in claim.provenance_links:
                source_ids.add(link.source_entity_id)

        # Build entity lookup
        entity_lookup = {entity.id: entity for entity in context_entities}

        # Create citations
        citations = []
        for i, source_id in enumerate(sorted(source_ids, key=str)):
            if source_id not in entity_lookup:
                continue

            entity = entity_lookup[source_id]

            # Build display text
            name = entity.properties.get('name', str(entity.id))
            entity_type = entity.ontology_type

            display_text = f"{name} ({entity_type})"

            # Build full reference
            full_reference = self._build_full_reference(entity)

            citation = Citation(
                citation_id=f"cite_{source_id}",
                entity_id=source_id,
                entity_type=entity_type,
                display_text=display_text,
                full_reference=full_reference,
                metadata={
                    "citation_number": i + 1
                }
            )

            citations.append(citation)

        return citations

    def _build_full_reference(self, entity: KnowledgeEntity) -> str:
        """Build full reference for an entity"""
        parts = []

        # Name/Title
        name = entity.properties.get('name') or entity.properties.get('title')
        if name:
            parts.append(name)

        # Type
        parts.append(f"({entity.ontology_type})")

        # ID
        parts.append(f"ID: {entity.id}")

        # Created date
        if entity.created_at:
            parts.append(f"Created: {entity.created_at.strftime('%Y-%m-%d')}")

        return " | ".join(parts)

    def _calculate_coverage(self, grounded_claims: List[GroundedClaim]) -> float:
        """Calculate what percentage of answer is grounded"""
        if not grounded_claims:
            return 0.0

        grounded_count = sum(1 for c in grounded_claims if c.is_grounded)
        return grounded_count / len(grounded_claims)

    def _annotate_with_citations(
        self,
        answer_text: str,
        grounded_claims: List[GroundedClaim],
        citation_style: CitationStyle
    ) -> str:
        """
        Annotate answer with citation markers.

        Args:
            answer_text: Original answer
            grounded_claims: Grounded claims
            citation_style: Citation style

        Returns:
            Annotated answer text
        """
        # Simple implementation: add citations at end of each claim
        # Real implementation would do inline insertion

        annotated = answer_text

        for claim in grounded_claims:
            if not claim.is_grounded or not claim.citation_ids:
                continue

            # Get citation numbers
            citation_nums = []
            for cit_id in claim.citation_ids:
                # Extract number from cite_ID format
                # This is simplified
                citation_nums.append("1")  # Would extract actual number

            # Build citation marker
            if citation_style == CitationStyle.INLINE:
                marker = f"[{','.join(citation_nums)}]"
            elif citation_style == CitationStyle.SUPERSCRIPT:
                marker = f"^{','.join(citation_nums)}"
            elif citation_style == CitationStyle.PARENTHETICAL:
                marker = f" (Sources: {','.join(citation_nums)})"
            else:
                marker = f"[{','.join(citation_nums)}]"

            # Add marker after claim (simple replacement)
            annotated = annotated.replace(
                claim.claim_text,
                claim.claim_text + marker,
                1  # Only first occurrence
            )

        return annotated

    def _format_citations(
        self,
        citations: List[Citation],
        citation_style: CitationStyle
    ) -> str:
        """
        Format citations as text.

        Args:
            citations: Citations to format
            citation_style: Citation style

        Returns:
            Formatted citation text
        """
        if not citations:
            return ""

        lines = ["\n## Sources\n"]

        for citation in citations:
            num = citation.metadata.get("citation_number", "?")

            if citation_style == CitationStyle.INLINE:
                line = f"[{num}] {citation.full_reference}"
            elif citation_style == CitationStyle.SUPERSCRIPT:
                line = f"^{num} {citation.full_reference}"
            else:
                line = f"{num}. {citation.full_reference}"

            lines.append(line)

        return "\n".join(lines)

    async def explain_provenance(
        self,
        grounded_answer: GroundedAnswer,
        claim_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate detailed provenance explanation.

        Args:
            grounded_answer: Grounded answer
            claim_id: Optional specific claim to explain

        Returns:
            Provenance explanation
        """
        if claim_id:
            # Explain specific claim
            claim = next(
                (c for c in grounded_answer.provenance_graph.grounded_claims
                 if c.claim_id == claim_id),
                None
            )

            if not claim:
                return {"error": f"Claim {claim_id} not found"}

            return {
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "is_grounded": claim.is_grounded,
                "confidence": claim.confidence,
                "provenance_links": [
                    {
                        "source_entity": str(link.source_entity_id),
                        "provenance_type": link.provenance_type,
                        "confidence": link.confidence,
                        "explanation": link.explanation,
                        "property": link.property_name
                    }
                    for link in claim.provenance_links
                ]
            }
        else:
            # Explain entire answer
            graph = grounded_answer.provenance_graph

            return {
                "answer": graph.answer_text,
                "fully_grounded": grounded_answer.fully_grounded,
                "provenance_coverage": graph.provenance_coverage,
                "total_claims": len(graph.grounded_claims),
                "grounded_claims": sum(1 for c in graph.grounded_claims if c.is_grounded),
                "ungrounded_claims": len(graph.get_ungrounded_claims()),
                "sources_used": len(graph.get_sources_used()),
                "citations": [
                    {
                        "id": cit.citation_id,
                        "entity": str(cit.entity_id),
                        "display": cit.display_text
                    }
                    for cit in graph.citations
                ]
            }


# Convenience functions
async def ground_answer(
    answer_text: str,
    context_entities: List[KnowledgeEntity],
    storage: VecnaDBStorageInterface,
    ontology: OntologySchema,
    citation_style: CitationStyle = CitationStyle.INLINE
) -> GroundedAnswer:
    """
    Ground answer with provenance and citations.

    Args:
        answer_text: Answer text
        context_entities: Context used
        storage: Storage interface
        ontology: Ontology schema
        citation_style: Citation style

    Returns:
        GroundedAnswer with provenance
    """
    grounding = AnswerGrounding(storage, ontology)
    return await grounding.ground_answer(answer_text, context_entities, citation_style)


async def trace_claim(
    claim_text: str,
    context_entities: List[KnowledgeEntity],
    storage: VecnaDBStorageInterface,
    ontology: OntologySchema
) -> List[ProvenanceLink]:
    """
    Trace a claim to source entities.

    Args:
        claim_text: Claim to trace
        context_entities: Context to search
        storage: Storage interface
        ontology: Ontology schema

    Returns:
        List of provenance links
    """
    grounding = AnswerGrounding(storage, ontology)
    grounded_claim = await grounding._ground_claim("claim", claim_text, context_entities)
    return grounded_claim.provenance_links
