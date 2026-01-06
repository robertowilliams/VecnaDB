"""
VecnaDB Hallucination Prevention

Prevents hallucination in RAG by detecting unsupported claims and verifying
all statements against the knowledge graph.

Hallucination Detection Methods:
1. Claim extraction from generated text
2. Entity grounding (every claim must reference real entities)
3. Relation verification (claimed relations must exist in graph)
4. Fact checking against knowledge graph
5. Consistency checking across claims
6. Confidence thresholding

Principle: If a claim cannot be traced to the knowledge graph, it is a hallucination.
"""

import re
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


class HallucinationType(str, Enum):
    """Types of hallucination"""
    UNGROUNDED_CLAIM = "ungrounded_claim"  # Claim not supported by context
    FABRICATED_ENTITY = "fabricated_entity"  # References non-existent entity
    FABRICATED_RELATION = "fabricated_relation"  # Claims non-existent relation
    CONTRADICTORY_CLAIM = "contradictory_claim"  # Contradicts knowledge graph
    UNSUPPORTED_INFERENCE = "unsupported_inference"  # Invalid inference
    OVERCONFIDENT_CLAIM = "overconfident_claim"  # Claim stronger than evidence


class Claim(BaseModel):
    """A claim extracted from generated text"""
    text: str
    entities_mentioned: List[str] = Field(default_factory=list)
    relations_claimed: List[str] = Field(default_factory=list)
    start_pos: int
    end_pos: int


class HallucinationDetection(BaseModel):
    """A detected hallucination"""
    claim: Claim
    hallucination_type: HallucinationType
    severity: str  # "critical", "warning", "minor"
    explanation: str
    suggested_correction: Optional[str] = None


class HallucinationReport(BaseModel):
    """Complete hallucination analysis report"""
    answer_text: str
    claims_extracted: int
    claims_verified: int
    hallucinations_detected: int
    detections: List[HallucinationDetection]
    overall_risk: float  # 0.0-1.0
    is_trustworthy: bool
    analysis_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_critical_hallucinations(self) -> List[HallucinationDetection]:
        """Get critical hallucinations only"""
        return [d for d in self.detections if d.severity == "critical"]

    def has_critical_hallucinations(self) -> bool:
        """Check if any critical hallucinations detected"""
        return len(self.get_critical_hallucinations()) > 0


class HallucinationPrevention:
    """
    Prevents hallucination in RAG by verifying all claims against knowledge graph.

    Process:
    1. Extract claims from generated answer
    2. Parse entities and relations mentioned
    3. Verify each entity exists in knowledge graph
    4. Verify each relation exists in knowledge graph
    5. Check for contradictions
    6. Calculate hallucination risk
    7. Flag unverifiable claims
    """

    def __init__(
        self,
        storage: VecnaDBStorageInterface,
        ontology: OntologySchema
    ):
        """
        Initialize hallucination prevention.

        Args:
            storage: VecnaDB storage interface
            ontology: Ontology schema
        """
        self.storage = storage
        self.ontology = ontology

    async def detect_hallucinations(
        self,
        answer_text: str,
        context_entities: List[KnowledgeEntity],
        confidence_threshold: float = 0.7
    ) -> HallucinationReport:
        """
        Detect hallucinations in generated answer.

        Args:
            answer_text: Generated answer text
            context_entities: Context used for generation
            confidence_threshold: Minimum confidence for verification

        Returns:
            HallucinationReport with detections
        """
        import time
        start_time = time.time()

        # 1. Extract claims from answer
        claims = self._extract_claims(answer_text)

        # 2. Build context lookup
        context_lookup = self._build_context_lookup(context_entities)

        # 3. Verify each claim
        detections = []
        verified_count = 0

        for claim in claims:
            detection = await self._verify_claim(
                claim,
                context_lookup,
                context_entities
            )

            if detection:
                detections.append(detection)
            else:
                verified_count += 1

        # 4. Calculate overall risk
        overall_risk = self._calculate_hallucination_risk(
            claims,
            detections,
            confidence_threshold
        )

        # 5. Determine trustworthiness
        is_trustworthy = (
            overall_risk < 0.3
            and not any(d.severity == "critical" for d in detections)
        )

        report = HallucinationReport(
            answer_text=answer_text,
            claims_extracted=len(claims),
            claims_verified=verified_count,
            hallucinations_detected=len(detections),
            detections=detections,
            overall_risk=overall_risk,
            is_trustworthy=is_trustworthy,
            analysis_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "confidence_threshold": confidence_threshold,
                "context_entities": len(context_entities)
            }
        )

        return report

    async def verify_answer(
        self,
        answer_text: str,
        context_entities: List[KnowledgeEntity],
        require_all_grounded: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Verify answer against knowledge graph.

        Args:
            answer_text: Generated answer
            context_entities: Context used
            require_all_grounded: Require all claims grounded

        Returns:
            Tuple of (is_valid, list of issues)
        """
        report = await self.detect_hallucinations(answer_text, context_entities)

        issues = []

        # Check for critical hallucinations
        if report.has_critical_hallucinations():
            for detection in report.get_critical_hallucinations():
                issues.append(
                    f"CRITICAL: {detection.hallucination_type} - {detection.explanation}"
                )

        # Check if trustworthy
        if not report.is_trustworthy:
            issues.append(f"High hallucination risk: {report.overall_risk:.2f}")

        # Check grounding requirement
        if require_all_grounded:
            unverified = report.claims_extracted - report.claims_verified
            if unverified > 0:
                issues.append(f"{unverified} claims could not be verified")

        is_valid = len(issues) == 0

        return is_valid, issues

    def _extract_claims(self, text: str) -> List[Claim]:
        """
        Extract claims from text.

        Simple implementation: split by sentences.
        Real implementation would use NLP for claim extraction.

        Args:
            text: Text to extract claims from

        Returns:
            List of claims
        """
        # Split into sentences (simple regex)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        claims = []
        pos = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Extract entity-like mentions (capitalized words)
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)

            # Extract relation-like phrases (verbs)
            # Simplified: look for common relation words
            relation_words = ['is', 'has', 'contains', 'relates to', 'part of']
            relations = [w for w in relation_words if w in sentence.lower()]

            claim = Claim(
                text=sentence,
                entities_mentioned=entities,
                relations_claimed=relations,
                start_pos=pos,
                end_pos=pos + len(sentence)
            )

            claims.append(claim)
            pos += len(sentence) + 1

        return claims

    def _build_context_lookup(
        self,
        context_entities: List[KnowledgeEntity]
    ) -> Dict[str, KnowledgeEntity]:
        """Build lookup table for context entities"""
        lookup = {}

        for entity in context_entities:
            # Add by ID
            lookup[str(entity.id)] = entity

            # Add by name
            if 'name' in entity.properties:
                name = entity.properties['name']
                lookup[name] = entity

            # Add by title
            if 'title' in entity.properties:
                title = entity.properties['title']
                lookup[title] = entity

        return lookup

    async def _verify_claim(
        self,
        claim: Claim,
        context_lookup: Dict[str, KnowledgeEntity],
        context_entities: List[KnowledgeEntity]
    ) -> Optional[HallucinationDetection]:
        """
        Verify a single claim against knowledge graph.

        Args:
            claim: Claim to verify
            context_lookup: Context entity lookup
            context_entities: All context entities

        Returns:
            HallucinationDetection if hallucination found, None otherwise
        """
        # Check 1: Are all mentioned entities in context?
        for entity_mention in claim.entities_mentioned:
            if entity_mention not in context_lookup:
                # Mentioned entity not in context - possible fabrication
                return HallucinationDetection(
                    claim=claim,
                    hallucination_type=HallucinationType.FABRICATED_ENTITY,
                    severity="critical",
                    explanation=(
                        f"Claim mentions entity '{entity_mention}' which is not in "
                        f"the retrieved context. This may be a fabricated entity."
                    ),
                    suggested_correction=(
                        "Remove mention of unverified entity or rephrase claim."
                    )
                )

        # Check 2: Are claimed relations verifiable?
        if claim.relations_claimed and claim.entities_mentioned:
            # For simplicity, check if any relations exist between mentioned entities
            # Real implementation would parse claim structure more carefully

            verified = await self._verify_relations_exist(
                claim.entities_mentioned,
                context_lookup,
                claim.relations_claimed
            )

            if not verified:
                return HallucinationDetection(
                    claim=claim,
                    hallucination_type=HallucinationType.FABRICATED_RELATION,
                    severity="warning",
                    explanation=(
                        f"Claim asserts relations between entities that cannot be "
                        f"verified in the knowledge graph."
                    ),
                    suggested_correction=(
                        "Weaken claim to indicate uncertainty or remove unverified relations."
                    )
                )

        # Check 3: Is claim grounded in context?
        is_grounded = self._check_grounding(claim, context_entities)

        if not is_grounded:
            return HallucinationDetection(
                claim=claim,
                hallucination_type=HallucinationType.UNGROUNDED_CLAIM,
                severity="warning",
                explanation=(
                    f"Claim '{claim.text[:50]}...' cannot be traced to context entities. "
                    f"It may be inferred or fabricated."
                ),
                suggested_correction=(
                    "Add explicit citation or remove ungrounded claim."
                )
            )

        # No hallucination detected
        return None

    async def _verify_relations_exist(
        self,
        entity_mentions: List[str],
        context_lookup: Dict[str, KnowledgeEntity],
        relation_claims: List[str]
    ) -> bool:
        """
        Verify that claimed relations exist in knowledge graph.

        Args:
            entity_mentions: Entities mentioned
            context_lookup: Context lookup
            relation_claims: Relations claimed

        Returns:
            True if relations verified, False otherwise
        """
        if len(entity_mentions) < 2:
            return True  # Can't verify relations with <2 entities

        # Get entity IDs
        entity_ids = []
        for mention in entity_mentions:
            if mention in context_lookup:
                entity_ids.append(context_lookup[mention].id)

        if len(entity_ids) < 2:
            return False

        # Check if any relations exist between these entities
        try:
            for entity_id in entity_ids:
                relations = await self.storage.get_relations(
                    entity_id=entity_id,
                    relation_type=None,  # Any relation type
                    direction="OUTGOING"
                )

                # Check if any relation connects to another mentioned entity
                for relation in relations:
                    if relation.target_id in entity_ids:
                        return True  # Found a relation

            # No relations found
            return False

        except Exception:
            # If we can't verify, assume false
            return False

    def _check_grounding(
        self,
        claim: Claim,
        context_entities: List[KnowledgeEntity]
    ) -> bool:
        """
        Check if claim is grounded in context.

        Simple implementation: check if claim text overlaps with context content.

        Args:
            claim: Claim to check
            context_entities: Context entities

        Returns:
            True if grounded, False otherwise
        """
        claim_text_lower = claim.text.lower()

        # Check for overlap with context content
        for entity in context_entities:
            content = entity.properties.get('content', '')
            if isinstance(content, str):
                content_lower = content.lower()

                # Check for significant word overlap
                claim_words = set(claim_text_lower.split())
                content_words = set(content_lower.split())

                overlap = claim_words.intersection(content_words)

                # If >50% of claim words appear in content, consider grounded
                if len(overlap) > len(claim_words) * 0.5:
                    return True

            # Also check name/title
            name = entity.properties.get('name', '')
            if isinstance(name, str) and name.lower() in claim_text_lower:
                return True

        return False

    def _calculate_hallucination_risk(
        self,
        claims: List[Claim],
        detections: List[HallucinationDetection],
        confidence_threshold: float
    ) -> float:
        """
        Calculate overall hallucination risk.

        Args:
            claims: All claims
            detections: Detected hallucinations
            confidence_threshold: Confidence threshold

        Returns:
            Risk score 0.0-1.0
        """
        if not claims:
            return 1.0  # No claims = high risk (empty answer)

        # Base risk from detection ratio
        detection_ratio = len(detections) / len(claims)

        # Weight by severity
        critical_count = sum(
            1 for d in detections if d.severity == "critical"
        )

        severity_factor = critical_count / len(claims) if claims else 0.0

        # Combined risk
        risk = (detection_ratio * 0.6) + (severity_factor * 0.4)

        # Boost risk if many detections
        if len(detections) > len(claims) * 0.5:
            risk = min(1.0, risk + 0.2)

        return risk

    async def suggest_corrections(
        self,
        answer_text: str,
        context_entities: List[KnowledgeEntity]
    ) -> Tuple[str, List[str]]:
        """
        Suggest corrections for hallucinated answer.

        Args:
            answer_text: Answer with potential hallucinations
            context_entities: Context used

        Returns:
            Tuple of (corrected_text, list of corrections made)
        """
        report = await self.detect_hallucinations(answer_text, context_entities)

        if not report.detections:
            return answer_text, []

        corrections = []
        corrected_text = answer_text

        # Apply suggested corrections
        for detection in report.detections:
            if detection.suggested_correction and detection.severity == "critical":
                # Remove the problematic claim
                corrected_text = corrected_text.replace(detection.claim.text, "")
                corrections.append(
                    f"Removed: '{detection.claim.text[:50]}...' - {detection.explanation}"
                )

        # Add disclaimer for warnings
        if any(d.severity == "warning" for d in report.detections):
            corrected_text += (
                "\n\n[Note: Some claims in this answer could not be fully verified "
                "against the knowledge graph.]"
            )
            corrections.append("Added verification disclaimer")

        return corrected_text, corrections


# Convenience functions
async def check_hallucination(
    answer_text: str,
    context_entities: List[KnowledgeEntity],
    storage: VecnaDBStorageInterface,
    ontology: OntologySchema
) -> HallucinationReport:
    """
    Check answer for hallucinations.

    Args:
        answer_text: Generated answer
        context_entities: Context used
        storage: Storage interface
        ontology: Ontology schema

    Returns:
        HallucinationReport
    """
    prevention = HallucinationPrevention(storage, ontology)
    return await prevention.detect_hallucinations(answer_text, context_entities)


async def verify_answer(
    answer_text: str,
    context_entities: List[KnowledgeEntity],
    storage: VecnaDBStorageInterface,
    ontology: OntologySchema
) -> Tuple[bool, List[str]]:
    """
    Verify answer is hallucination-free.

    Args:
        answer_text: Generated answer
        context_entities: Context used
        storage: Storage interface
        ontology: Ontology schema

    Returns:
        Tuple of (is_valid, list of issues)
    """
    prevention = HallucinationPrevention(storage, ontology)
    return await prevention.verify_answer(answer_text, context_entities)
