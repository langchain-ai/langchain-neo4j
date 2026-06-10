import pytest
from langchain_core.documents import Document

from langchain_neo4j.graph_transformers.llm import (
    LLMGraphTransformer,
    _format_nodes,
    _format_relationships,
    _parse_and_clean_json,
    create_simple_model,
    create_unstructured_prompt,
    format_property_key,
    get_default_prompt,
    validate_and_get_relationship_type,
)
from langchain_neo4j.graphs.graph_document import Node, Relationship
from tests.llms.fake_llm import FakeLLM, FakeStructuredChatModel

# The schema the transformer would build at runtime when no node/relationship
# constraints are provided. Reusing it here keeps the fake LLM response in sync
# with the real shape map_to_base_node / map_to_base_relationship expect.
SimpleGraph = create_simple_model()


# ---------- Pure helpers ----------


def test_format_property_key_camel_cases_multi_word() -> None:
    assert format_property_key("first name") == "firstName"
    assert format_property_key("Date Of Birth") == "dateOfBirth"
    assert format_property_key("single") == "single"


def test_format_nodes_normalises_id_and_type() -> None:
    formatted = _format_nodes([Node(id="alice smith", type="person")])
    assert formatted[0].id == "Alice Smith"
    assert formatted[0].type == "Person"


def test_format_relationships_uppercases_type_with_underscores() -> None:
    rel = Relationship(
        source=Node(id="alice", type="person"),
        target=Node(id="acme", type="company"),
        type="works at",
    )
    formatted = _format_relationships([rel])[0]
    assert formatted.type == "WORKS_AT"
    assert formatted.source.id == "Alice"
    assert formatted.target.type == "Company"


def test_parse_and_clean_json_skips_invalid_entries_and_infers_types() -> None:
    payload = {
        "nodes": [
            {"id": "Alice", "type": "Person"},
            {"id": "Acme", "type": "Company"},
            {"id": "", "type": "Person"},  # missing id -> skipped
        ],
        "relationships": [
            # full relationship -> kept
            {
                "source_node_id": "Alice",
                "target_node_id": "Acme",
                "type": "WORKS_AT",
            },
            # missing type -> skipped
            {"source_node_id": "Alice", "target_node_id": "Acme"},
            # missing target -> skipped
            {"source_node_id": "Alice", "type": "KNOWS"},
        ],
    }
    nodes, rels = _parse_and_clean_json(payload)
    assert [n.id for n in nodes] == ["Alice", "Acme"]
    assert len(rels) == 1
    # source / target types inferred from the node list
    assert rels[0].source.type == "Person"
    assert rels[0].target.type == "Company"


def test_validate_and_get_relationship_type_string() -> None:
    assert (
        validate_and_get_relationship_type(["KNOWS", "WORKS_AT"], allowed_nodes=None)
        == "string"
    )


def test_validate_and_get_relationship_type_tuple_ok() -> None:
    allowed = [("Person", "WORKS_AT", "Company")]
    assert (
        validate_and_get_relationship_type(allowed, allowed_nodes=["Person", "Company"])
        == "tuple"
    )


def test_validate_and_get_relationship_type_tuple_rejects_unknown_node() -> None:
    allowed = [("Person", "WORKS_AT", "Company")]
    with pytest.raises(ValueError):
        validate_and_get_relationship_type(allowed, allowed_nodes=["Person"])


def test_create_simple_model_rejects_id_node_property() -> None:
    with pytest.raises(ValueError, match="'id' is reserved"):
        create_simple_model(node_properties=["id", "name"])


def test_prompts_include_constructor_arguments() -> None:
    """Guards against regressions where additional_instructions or allowed_*
    silently stop being passed into the rendered prompt."""
    default = get_default_prompt(additional_instructions="ALWAYS_USE_TITLE_CASE")
    rendered_default = default.format(input="some text")
    assert "ALWAYS_USE_TITLE_CASE" in rendered_default

    unstructured = create_unstructured_prompt(
        node_labels=["Person", "Company"],
        rel_types=["WORKS_AT"],
        additional_instructions="ALWAYS_USE_TITLE_CASE",
    )
    rendered_unstructured = unstructured.format(input="some text")
    assert "Person" in rendered_unstructured
    assert "Company" in rendered_unstructured
    assert "WORKS_AT" in rendered_unstructured
    assert "ALWAYS_USE_TITLE_CASE" in rendered_unstructured


# ---------- End-to-end happy paths ----------


def test_convert_to_graph_documents_function_calling_path() -> None:
    """Function-calling path: with_structured_output yields a parsed object,
    process_response -> _convert_to_graph_document -> _format_nodes/relationships."""
    parsed = SimpleGraph(
        nodes=[
            {"id": "alice", "type": "person"},
            {"id": "acme", "type": "company"},
        ],
        relationships=[
            {
                "source_node_id": "alice",
                "source_node_type": "person",
                "target_node_id": "acme",
                "target_node_type": "company",
                "type": "works at",
            }
        ],
    )
    llm = FakeStructuredChatModel(response=parsed)
    transformer = LLMGraphTransformer(llm=llm)

    docs = transformer.convert_to_graph_documents(
        [Document(page_content="Alice works at Acme")]
    )

    assert len(docs) == 1
    graph_doc = docs[0]
    assert {(n.id, n.type) for n in graph_doc.nodes} == {
        ("Alice", "Person"),
        ("Acme", "Company"),
    }
    assert len(graph_doc.relationships) == 1
    rel = graph_doc.relationships[0]
    assert rel.type == "WORKS_AT"
    assert rel.source.id == "Alice"
    assert rel.target.id == "Acme"
    assert graph_doc.source is not None
    assert graph_doc.source.page_content == "Alice works at Acme"


async def test_aconvert_to_graph_documents_prompt_based_path() -> None:
    """Prompt-based path: FakeLLM extends LLM, so with_structured_output raises
    NotImplementedError and the transformer falls back to JSON-repair parsing."""
    response = (
        '[{"head": "Alice", "head_type": "Person", '
        '"relation": "WORKS_AT", '
        '"tail": "Acme", "tail_type": "Company"}]'
    )
    # FakeLLM ignores prompt keys when queries lookup fails; use sequential mode
    # so any rendered prompt receives the same canned response.
    llm = FakeLLM(queries={"_": response}, sequential_responses=True)
    transformer = LLMGraphTransformer(llm=llm)

    docs = await transformer.aconvert_to_graph_documents(
        [Document(page_content="Alice works at Acme")]
    )

    assert len(docs) == 1
    graph_doc = docs[0]
    assert {(n.id, n.type) for n in graph_doc.nodes} == {
        ("Alice", "Person"),
        ("Acme", "Company"),
    }
    assert len(graph_doc.relationships) == 1
    assert graph_doc.relationships[0].type == "WORKS_AT"


# ---------- Strict-mode filtering ----------


def test_strict_mode_filters_disallowed_nodes_and_relationships() -> None:
    parsed = SimpleGraph(
        nodes=[
            {"id": "alice", "type": "person"},
            {"id": "acme", "type": "company"},
            {"id": "rover", "type": "dog"},  # disallowed node type
        ],
        relationships=[
            {
                "source_node_id": "alice",
                "source_node_type": "person",
                "target_node_id": "acme",
                "target_node_type": "company",
                "type": "works at",
            },
            {
                "source_node_id": "alice",
                "source_node_type": "person",
                "target_node_id": "rover",
                "target_node_type": "dog",
                "type": "owns",  # disallowed relationship + disallowed target
            },
        ],
    )
    llm = FakeStructuredChatModel(response=parsed)
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Person", "Company"],
        allowed_relationships=["WORKS_AT"],
    )

    docs = transformer.convert_to_graph_documents(
        [Document(page_content="Alice works at Acme; Alice owns Rover")]
    )

    graph_doc = docs[0]
    assert {n.type for n in graph_doc.nodes} == {"Person", "Company"}
    assert [r.type for r in graph_doc.relationships] == ["WORKS_AT"]


def test_strict_mode_tuple_filter_drops_wrong_direction() -> None:
    """Tuple-form allowed_relationships filters by the full
    (source_type, rel_type, target_type) triple, not by rel_type alone.
    A (Person)-[WORKS_AT]->(Company) edge is kept, but the reverse direction
    (Company)-[WORKS_AT]->(Person) is dropped."""
    parsed = SimpleGraph(
        nodes=[
            {"id": "alice", "type": "person"},
            {"id": "acme", "type": "company"},
        ],
        relationships=[
            {
                "source_node_id": "alice",
                "source_node_type": "person",
                "target_node_id": "acme",
                "target_node_type": "company",
                "type": "WORKS_AT",
            },
            {
                # same rel type, opposite direction -> not in allowed triples
                "source_node_id": "acme",
                "source_node_type": "company",
                "target_node_id": "alice",
                "target_node_type": "person",
                "type": "WORKS_AT",
            },
        ],
    )
    llm = FakeStructuredChatModel(response=parsed)
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Person", "Company"],
        allowed_relationships=[("Person", "WORKS_AT", "Company")],
    )

    docs = transformer.convert_to_graph_documents(
        [Document(page_content="Alice works at Acme")]
    )

    rels = docs[0].relationships
    assert len(rels) == 1
    assert (rels[0].source.type, rels[0].type, rels[0].target.type) == (
        "Person",
        "WORKS_AT",
        "Company",
    )


def test_node_properties_round_trip() -> None:
    """With node_properties enabled, a node returned by the LLM with a
    properties list ends up as a dict on the resulting Node, with keys
    normalised through format_property_key."""
    graph_with_props = create_simple_model(node_properties=True)
    parsed = graph_with_props(
        nodes=[
            {
                "id": "alice",
                "type": "person",
                "properties": [
                    {"key": "first name", "value": "Alice"},
                    {"key": "age", "value": "30"},
                ],
            },
        ],
        relationships=[],
    )
    llm = FakeStructuredChatModel(response=parsed)
    transformer = LLMGraphTransformer(llm=llm, node_properties=True)

    docs = transformer.convert_to_graph_documents(
        [Document(page_content="Alice is 30")]
    )

    node = docs[0].nodes[0]
    assert node.properties == {"firstName": "Alice", "age": "30"}
