from typing import TypedDict, Optional

from vecnadb.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from vecnadb.modules.ontology.matching_strategies import MatchingStrategy


class OntologyConfig(TypedDict, total=False):
    """Configuration containing ontology resolver.

    Attributes:
        ontology_resolver: The ontology resolver instance to use
    """

    ontology_resolver: Optional[BaseOntologyResolver]


class Config(TypedDict, total=False):
    """Top-level configuration dictionary.

    Attributes:
        ontology_config: Configuration containing ontology resolver
    """

    ontology_config: Optional[OntologyConfig]
