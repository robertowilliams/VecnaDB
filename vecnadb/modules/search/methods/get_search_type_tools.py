import os
from typing import Callable, List, Optional, Type

from vecnadb.modules.engine.models.node_set import NodeSet
from vecnadb.modules.retrieval.triplet_retriever import TripletRetriever
from vecnadb.modules.search.types import SearchType
from vecnadb.modules.search.operations import select_search_type
from vecnadb.modules.search.exceptions import UnsupportedSearchTypeError

# Retrievers
from vecnadb.modules.retrieval.user_qa_feedback import UserQAFeedback
from vecnadb.modules.retrieval.chunks_retriever import ChunksRetriever
from vecnadb.modules.retrieval.summaries_retriever import SummariesRetriever
from vecnadb.modules.retrieval.completion_retriever import CompletionRetriever
from vecnadb.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from vecnadb.modules.retrieval.temporal_retriever import TemporalRetriever
from vecnadb.modules.retrieval.coding_rules_retriever import CodingRulesRetriever
from vecnadb.modules.retrieval.jaccard_retrival import JaccardChunksRetriever
from vecnadb.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)
from vecnadb.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from vecnadb.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from vecnadb.modules.retrieval.cypher_search_retriever import CypherSearchRetriever
from vecnadb.modules.retrieval.natural_language_retriever import NaturalLanguageRetriever


async def get_search_type_tools(
    query_type: SearchType,
    query_text: str,
    system_prompt_path: str = "answer_simple_question.txt",
    system_prompt: Optional[str] = None,
    top_k: int = 10,
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    save_interaction: bool = False,
    last_k: Optional[int] = None,
    wide_search_top_k: Optional[int] = 100,
    triplet_distance_penalty: Optional[float] = 3.5,
) -> list:
    search_tasks: dict[SearchType, List[Callable]] = {
        SearchType.SUMMARIES: [
            SummariesRetriever(top_k=top_k).get_completion,
            SummariesRetriever(top_k=top_k).get_context,
        ],
        SearchType.CHUNKS: [
            ChunksRetriever(top_k=top_k).get_completion,
            ChunksRetriever(top_k=top_k).get_context,
        ],
        SearchType.RAG_COMPLETION: [
            CompletionRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                system_prompt=system_prompt,
            ).get_completion,
            CompletionRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                system_prompt=system_prompt,
            ).get_context,
        ],
        SearchType.TRIPLET_COMPLETION: [
            TripletRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                system_prompt=system_prompt,
            ).get_completion,
            TripletRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                system_prompt=system_prompt,
            ).get_context,
        ],
        SearchType.GRAPH_COMPLETION: [
            GraphCompletionRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                system_prompt=system_prompt,
                wide_search_top_k=wide_search_top_k,
                triplet_distance_penalty=triplet_distance_penalty,
            ).get_completion,
            GraphCompletionRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                system_prompt=system_prompt,
                wide_search_top_k=wide_search_top_k,
                triplet_distance_penalty=triplet_distance_penalty,
            ).get_context,
        ],
        SearchType.GRAPH_COMPLETION_COT: [
            GraphCompletionCotRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                system_prompt=system_prompt,
                wide_search_top_k=wide_search_top_k,
                triplet_distance_penalty=triplet_distance_penalty,
            ).get_completion,
            GraphCompletionCotRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                system_prompt=system_prompt,
                wide_search_top_k=wide_search_top_k,
                triplet_distance_penalty=triplet_distance_penalty,
            ).get_context,
        ],
        SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION: [
            GraphCompletionContextExtensionRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                system_prompt=system_prompt,
                wide_search_top_k=wide_search_top_k,
                triplet_distance_penalty=triplet_distance_penalty,
            ).get_completion,
            GraphCompletionContextExtensionRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                system_prompt=system_prompt,
                wide_search_top_k=wide_search_top_k,
                triplet_distance_penalty=triplet_distance_penalty,
            ).get_context,
        ],
        SearchType.GRAPH_SUMMARY_COMPLETION: [
            GraphSummaryCompletionRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                system_prompt=system_prompt,
                wide_search_top_k=wide_search_top_k,
                triplet_distance_penalty=triplet_distance_penalty,
            ).get_completion,
            GraphSummaryCompletionRetriever(
                system_prompt_path=system_prompt_path,
                top_k=top_k,
                node_type=node_type,
                node_name=node_name,
                save_interaction=save_interaction,
                system_prompt=system_prompt,
                wide_search_top_k=wide_search_top_k,
                triplet_distance_penalty=triplet_distance_penalty,
            ).get_context,
        ],
        SearchType.CYPHER: [
            CypherSearchRetriever().get_completion,
            CypherSearchRetriever().get_context,
        ],
        SearchType.NATURAL_LANGUAGE: [
            NaturalLanguageRetriever().get_completion,
            NaturalLanguageRetriever().get_context,
        ],
        SearchType.FEEDBACK: [UserQAFeedback(last_k=last_k).add_feedback],
        SearchType.TEMPORAL: [
            TemporalRetriever(
                top_k=top_k,
                wide_search_top_k=wide_search_top_k,
                triplet_distance_penalty=triplet_distance_penalty,
            ).get_completion,
            TemporalRetriever(
                top_k=top_k,
                wide_search_top_k=wide_search_top_k,
                triplet_distance_penalty=triplet_distance_penalty,
            ).get_context,
        ],
        SearchType.CHUNKS_LEXICAL: (
            lambda _r=JaccardChunksRetriever(top_k=top_k): [
                _r.get_completion,
                _r.get_context,
            ]
        )(),
        SearchType.CODING_RULES: [
            CodingRulesRetriever(rules_nodeset_name=node_name).get_existing_rules,
        ],
    }

    # If the query type is FEELING_LUCKY, select the search type intelligently
    if query_type is SearchType.FEELING_LUCKY:
        query_type = await select_search_type(query_text)

    if (
        query_type in [SearchType.CYPHER, SearchType.NATURAL_LANGUAGE]
        and os.getenv("ALLOW_CYPHER_QUERY", "true").lower() == "false"
    ):
        raise UnsupportedSearchTypeError("Cypher query search types are disabled.")

    from vecnadb.modules.retrieval.registered_community_retrievers import (
        registered_community_retrievers,
    )

    if query_type in registered_community_retrievers:
        retriever = registered_community_retrievers[query_type]
        retriever_instance = retriever(top_k=top_k)
        search_type_tools = [
            retriever_instance.get_completion,
            retriever_instance.get_context,
        ]
    else:
        search_type_tools = search_tasks.get(query_type)

    if not search_type_tools:
        raise UnsupportedSearchTypeError(str(query_type))

    return search_type_tools
