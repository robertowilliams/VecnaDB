"""
VecnaDB Ontology Loader

This module provides functionality to load ontology schemas from various formats
(primarily YAML) and convert them into OntologySchema objects.

Supported formats:
- YAML (primary)
- JSON (future)
- Python dict (programmatic)
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from uuid import uuid4
from datetime import datetime, timezone

from vecnadb.modules.ontology.models.OntologySchema import (
    OntologySchema,
    EntityTypeDefinition,
    RelationTypeDefinition,
    PropertyDefinition,
    Constraint,
    EmbeddingRequirements,
    PropertyType,
    ConstraintType,
    Cardinality,
)


class OntologyLoader:
    """
    Loads ontology schemas from files or dictionaries.

    Handles parsing and validation of ontology definitions.
    """

    @staticmethod
    def load_from_yaml(file_path: str) -> OntologySchema:
        """
        Load ontology from a YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            OntologySchema instance

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is invalid
            ValueError: If ontology structure is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Ontology file not found: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return OntologyLoader.load_from_dict(data)

    @staticmethod
    def load_from_json(file_path: str) -> OntologySchema:
        """
        Load ontology from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            OntologySchema instance
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Ontology file not found: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return OntologyLoader.load_from_dict(data)

    @staticmethod
    def load_from_dict(data: Dict[str, Any]) -> OntologySchema:
        """
        Load ontology from a dictionary.

        Args:
            data: Dictionary containing ontology definition

        Returns:
            OntologySchema instance

        Raises:
            ValueError: If ontology structure is invalid
        """
        # Extract basic metadata
        name = data.get('name')
        if not name:
            raise ValueError("Ontology must have a 'name' field")

        version = data.get('version', '1.0.0')
        description = data.get('description', '')

        # Parse entity types
        entity_types = {}
        entity_types_data = data.get('entity_types', {})

        for type_name, type_data in entity_types_data.items():
            entity_types[type_name] = OntologyLoader._parse_entity_type(
                type_name, type_data
            )

        # Parse relation types
        relation_types = {}
        relation_types_data = data.get('relation_types', {})

        for type_name, type_data in relation_types_data.items():
            relation_types[type_name] = OntologyLoader._parse_relation_type(
                type_name, type_data
            )

        # Parse global constraints
        global_constraints = []
        global_constraints_data = data.get('global_constraints', [])
        for constraint_data in global_constraints_data:
            global_constraints.append(
                OntologyLoader._parse_constraint(constraint_data)
            )

        # Create ontology
        ontology = OntologySchema(
            id=uuid4(),
            name=name,
            version=version,
            description=description,
            entity_types=entity_types,
            relation_types=relation_types,
            global_constraints=global_constraints,
            metadata=data.get('metadata', {})
        )

        return ontology

    @staticmethod
    def _parse_entity_type(
        type_name: str,
        data: Dict[str, Any]
    ) -> EntityTypeDefinition:
        """Parse entity type definition from dictionary"""

        # Parse properties
        properties = {}
        properties_data = data.get('properties', {})
        for prop_name, prop_data in properties_data.items():
            properties[prop_name] = OntologyLoader._parse_property(
                prop_name, prop_data
            )

        # Parse constraints
        constraints = []
        constraints_data = data.get('constraints', [])
        for constraint_data in constraints_data:
            constraints.append(
                OntologyLoader._parse_constraint(constraint_data)
            )

        # Parse embedding requirements
        emb_req_data = data.get('embedding_requirements', {})
        embedding_requirements = EmbeddingRequirements(
            required_types=emb_req_data.get('required_types', ['content']),
            optional_types=emb_req_data.get('optional_types', []),
            embeddable_properties=emb_req_data.get('embeddable_properties', []),
            min_embeddings=emb_req_data.get('min_embeddings', 1)
        )

        return EntityTypeDefinition(
            name=type_name,
            description=data.get('description', ''),
            properties=properties,
            required_properties=data.get('required_properties', []),
            inherits_from=data.get('inherits_from'),
            constraints=constraints,
            embedding_requirements=embedding_requirements,
            abstract=data.get('abstract', False)
        )

    @staticmethod
    def _parse_relation_type(
        type_name: str,
        data: Dict[str, Any]
    ) -> RelationTypeDefinition:
        """Parse relation type definition from dictionary"""

        # Parse properties
        properties = {}
        properties_data = data.get('properties', {})
        for prop_name, prop_data in properties_data.items():
            properties[prop_name] = OntologyLoader._parse_property(
                prop_name, prop_data
            )

        # Parse constraints
        constraints = []
        constraints_data = data.get('constraints', [])
        for constraint_data in constraints_data:
            constraints.append(
                OntologyLoader._parse_constraint(constraint_data)
            )

        # Parse cardinality
        source_card = data.get('source_cardinality', '0..*')
        target_card = data.get('target_cardinality', '0..*')

        return RelationTypeDefinition(
            name=type_name,
            description=data.get('description', ''),
            allowed_source_types=data.get('allowed_source_types', []),
            allowed_target_types=data.get('allowed_target_types', []),
            source_cardinality=Cardinality(source_card),
            target_cardinality=Cardinality(target_card),
            is_directed=data.get('is_directed', True),
            symmetric=data.get('symmetric', False),
            transitive=data.get('transitive', False),
            inverse_of=data.get('inverse_of'),
            properties=properties,
            required_properties=data.get('required_properties', []),
            constraints=constraints
        )

    @staticmethod
    def _parse_property(
        prop_name: str,
        data: Dict[str, Any]
    ) -> PropertyDefinition:
        """Parse property definition from dictionary"""

        # Parse property type
        type_str = data.get('type', 'string')
        try:
            prop_type = PropertyType(type_str.lower())
        except ValueError:
            raise ValueError(f"Invalid property type: {type_str}")

        # Parse constraints
        constraints = []
        constraints_data = data.get('constraints', [])
        for constraint_data in constraints_data:
            constraints.append(
                OntologyLoader._parse_constraint(constraint_data)
            )

        return PropertyDefinition(
            name=prop_name,
            type=prop_type,
            required=data.get('required', False),
            default=data.get('default'),
            constraints=constraints,
            indexed=data.get('indexed', False),
            embeddable=data.get('embeddable', False),
            description=data.get('description', '')
        )

    @staticmethod
    def _parse_constraint(data: Dict[str, Any]) -> Constraint:
        """Parse constraint from dictionary"""

        # Parse constraint type
        type_str = data.get('type', 'custom')
        try:
            constraint_type = ConstraintType(type_str.lower())
        except ValueError:
            raise ValueError(f"Invalid constraint type: {type_str}")

        return Constraint(
            type=constraint_type,
            parameters=data.get('parameters', {}),
            error_message=data.get('error_message', 'Constraint validation failed')
        )

    @staticmethod
    def save_to_yaml(ontology: OntologySchema, file_path: str):
        """
        Save ontology to a YAML file.

        Args:
            ontology: The OntologySchema to save
            file_path: Path where to save the YAML file
        """
        data = OntologyLoader.ontology_to_dict(ontology)

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True
            )

    @staticmethod
    def save_to_json(ontology: OntologySchema, file_path: str):
        """
        Save ontology to a JSON file.

        Args:
            ontology: The OntologySchema to save
            file_path: Path where to save the JSON file
        """
        data = OntologyLoader.ontology_to_dict(ontology)

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def ontology_to_dict(ontology: OntologySchema) -> Dict[str, Any]:
        """
        Convert OntologySchema to a dictionary suitable for serialization.

        Args:
            ontology: The OntologySchema to convert

        Returns:
            Dictionary representation
        """
        data = {
            'name': ontology.name,
            'version': ontology.version,
            'description': ontology.description,
            'entity_types': {},
            'relation_types': {},
            'global_constraints': [],
            'metadata': ontology.metadata
        }

        # Convert entity types
        for type_name, type_def in ontology.entity_types.items():
            data['entity_types'][type_name] = {
                'description': type_def.description,
                'abstract': type_def.abstract,
                'inherits_from': type_def.inherits_from,
                'properties': {
                    name: {
                        'type': prop.type.value,
                        'required': prop.required,
                        'default': prop.default,
                        'indexed': prop.indexed,
                        'embeddable': prop.embeddable,
                        'description': prop.description,
                        'constraints': [
                            {
                                'type': c.type.value,
                                'parameters': c.parameters,
                                'error_message': c.error_message
                            }
                            for c in prop.constraints
                        ]
                    }
                    for name, prop in type_def.properties.items()
                },
                'required_properties': type_def.required_properties,
                'embedding_requirements': {
                    'required_types': type_def.embedding_requirements.required_types,
                    'optional_types': type_def.embedding_requirements.optional_types,
                    'embeddable_properties': type_def.embedding_requirements.embeddable_properties,
                    'min_embeddings': type_def.embedding_requirements.min_embeddings
                }
            }

        # Convert relation types
        for type_name, type_def in ontology.relation_types.items():
            data['relation_types'][type_name] = {
                'description': type_def.description,
                'is_directed': type_def.is_directed,
                'symmetric': type_def.symmetric,
                'transitive': type_def.transitive,
                'inverse_of': type_def.inverse_of,
                'allowed_source_types': type_def.allowed_source_types,
                'allowed_target_types': type_def.allowed_target_types,
                'source_cardinality': type_def.source_cardinality.value,
                'target_cardinality': type_def.target_cardinality.value,
                'properties': {
                    name: {
                        'type': prop.type.value,
                        'required': prop.required,
                        'default': prop.default,
                        'description': prop.description
                    }
                    for name, prop in type_def.properties.items()
                },
                'required_properties': type_def.required_properties
            }

        # Convert global constraints
        for constraint in ontology.global_constraints:
            data['global_constraints'].append({
                'type': constraint.type.value,
                'parameters': constraint.parameters,
                'error_message': constraint.error_message
            })

        return data


# Convenience function to load the core ontology
def load_core_ontology() -> OntologySchema:
    """
    Load the VecnaDB core ontology.

    Returns:
        The core OntologySchema

    Raises:
        FileNotFoundError: If core ontology file is not found
    """
    import os
    from pathlib import Path

    # Find the core ontology file
    current_dir = Path(__file__).parent.parent.parent  # vecnadb/modules/ontology -> vecnadb
    core_ontology_path = current_dir / "ontologies" / "core.yaml"

    if not core_ontology_path.exists():
        raise FileNotFoundError(
            f"Core ontology not found at: {core_ontology_path}"
        )

    return OntologyLoader.load_from_yaml(str(core_ontology_path))
