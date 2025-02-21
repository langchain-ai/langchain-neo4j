"""Test Graph Database Chain."""

from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseLanguageModel

from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph
from tests.integration_tests.utils import Neo4jCredentials
from tests.llms.fake_llm import FakeLLM


@pytest.mark.usefixtures("clear_neo4j_database")
def test_cypher_generating_run(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that Cypher statement is correctly generated and executed."""
    graph = Neo4jGraph(**neo4j_credentials)
    # Create two nodes and a relationship
    graph.query(
        "CREATE (a:Actor {name:'Bruce Willis'})"
        "-[:ACTED_IN]->(:Movie {title: 'Pulp Fiction'})"
    )
    # Refresh schema information
    graph.refresh_schema()

    query = (
        "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) "
        "WHERE m.title = 'Pulp Fiction' "
        "RETURN a.name"
    )
    llm = FakeLLM(
        queries={"query": query, "response": "Bruce Willis"}, sequential_responses=True
    )
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        validate_cypher=True,
        allow_dangerous_requests=True,
    )
    output = chain.run("Who starred in Pulp Fiction?")
    expected_output = "Bruce Willis"
    assert output == expected_output


@pytest.mark.usefixtures("clear_neo4j_database")
def test_cypher_top_k(neo4j_credentials: Neo4jCredentials) -> None:
    """Test top_k parameter correctly limits the number of results in the context."""
    TOP_K = 1
    graph = Neo4jGraph(**neo4j_credentials)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    graph.query(
        "CREATE (a:Actor {name:'Bruce Willis'})"
        "-[:ACTED_IN]->(:Movie {title: 'Pulp Fiction'})"
        "<-[:ACTED_IN]-(:Actor {name:'Foo'})"
    )
    # Refresh schema information
    graph.refresh_schema()

    query = (
        "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) "
        "WHERE m.title = 'Pulp Fiction' "
        "RETURN a.name"
    )
    llm = FakeLLM(queries={"query": query}, sequential_responses=True)
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        return_direct=True,
        top_k=TOP_K,
        allow_dangerous_requests=True,
    )
    output = chain.run("Who starred in Pulp Fiction?")
    assert len(output) == TOP_K


@pytest.mark.usefixtures("clear_neo4j_database")
def test_cypher_intermediate_steps(neo4j_credentials: Neo4jCredentials) -> None:
    """Test the returning of the intermediate steps."""
    graph = Neo4jGraph(**neo4j_credentials)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    graph.query(
        "CREATE (a:Actor {name:'Bruce Willis'})"
        "-[:ACTED_IN]->(:Movie {title: 'Pulp Fiction'})"
    )
    # Refresh schema information
    graph.refresh_schema()

    query = (
        "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) "
        "WHERE m.title = 'Pulp Fiction' "
        "RETURN a.name"
    )
    llm = FakeLLM(
        queries={"query": query, "response": "Bruce Willis"}, sequential_responses=True
    )
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
    )
    output = chain("Who starred in Pulp Fiction?")

    expected_output = "Bruce Willis"
    assert output["result"] == expected_output

    assert output["intermediate_steps"][0]["query"] == query

    context = output["intermediate_steps"][1]["context"]
    expected_context = [{"a.name": "Bruce Willis"}]
    assert context == expected_context


@pytest.mark.usefixtures("clear_neo4j_database")
def test_cypher_return_direct(neo4j_credentials: Neo4jCredentials) -> None:
    """Test that chain returns direct results."""
    graph = Neo4jGraph(**neo4j_credentials)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    graph.query(
        "CREATE (a:Actor {name:'Bruce Willis'})"
        "-[:ACTED_IN]->(:Movie {title: 'Pulp Fiction'})"
    )
    # Refresh schema information
    graph.refresh_schema()

    query = (
        "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) "
        "WHERE m.title = 'Pulp Fiction' "
        "RETURN a.name"
    )
    llm = FakeLLM(queries={"query": query}, sequential_responses=True)
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        return_direct=True,
        allow_dangerous_requests=True,
    )
    output = chain.run("Who starred in Pulp Fiction?")
    expected_output = [{"a.name": "Bruce Willis"}]
    assert output == expected_output


@pytest.mark.usefixtures("clear_neo4j_database")
def test_function_response(neo4j_credentials: Neo4jCredentials) -> None:
    """Test returning a function response."""
    graph = Neo4jGraph(**neo4j_credentials)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    graph.query(
        "CREATE (a:Actor {name:'Bruce Willis'})"
        "-[:ACTED_IN]->(:Movie {title: 'Pulp Fiction'})"
    )
    # Refresh schema information
    graph.refresh_schema()

    query = (
        "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) "
        "WHERE m.title = 'Pulp Fiction' "
        "RETURN a.name"
    )
    llm = FakeLLM(
        queries={"query": query, "response": "Bruce Willis"}, sequential_responses=True
    )
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        allow_dangerous_requests=True,
        use_function_response=True,
    )
    output = chain.run("Who starred in Pulp Fiction?")
    expected_output = "Bruce Willis"
    assert output == expected_output


@pytest.mark.usefixtures("clear_neo4j_database")
def test_exclude_types(neo4j_credentials: Neo4jCredentials) -> None:
    """Test exclude types from schema."""
    graph = Neo4jGraph(**neo4j_credentials)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    graph.query(
        "CREATE (a:Actor {name:'Bruce Willis'})"
        "-[:ACTED_IN]->(:Movie {title: 'Pulp Fiction'})"
        "<-[:DIRECTED]-(p:Person {name:'John'})"
    )
    # Refresh schema information
    graph.refresh_schema()

    llm = MagicMock(spec=BaseLanguageModel)
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        exclude_types=["Person", "DIRECTED"],
        allow_dangerous_requests=True,
    )
    expected_schema = (
        "Node properties:\n"
        "Actor {name: STRING}\n"
        "Movie {title: STRING}\n"
        "Relationship properties:\n\n"
        "The relationships:\n"
        "(:Actor)-[:ACTED_IN]->(:Movie)"
    )
    assert chain.graph_schema == expected_schema


@pytest.mark.usefixtures("clear_neo4j_database")
def test_include_types(neo4j_credentials: Neo4jCredentials) -> None:
    """Test include types from schema."""
    graph = Neo4jGraph(**neo4j_credentials)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    graph.query(
        "CREATE (a:Actor {name:'Bruce Willis'})"
        "-[:ACTED_IN]->(:Movie {title: 'Pulp Fiction'})"
        "<-[:DIRECTED]-(p:Person {name:'John'})"
    )
    # Refresh schema information
    graph.refresh_schema()

    llm = MagicMock(spec=BaseLanguageModel)
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        include_types=["Movie", "Actor", "ACTED_IN"],
        allow_dangerous_requests=True,
    )
    expected_schema = (
        "Node properties:\n"
        "Actor {name: STRING}\n"
        "Movie {title: STRING}\n"
        "Relationship properties:\n\n"
        "The relationships:\n"
        "(:Actor)-[:ACTED_IN]->(:Movie)"
    )

    assert chain.graph_schema == expected_schema


@pytest.mark.usefixtures("clear_neo4j_database")
def test_include_types2(neo4j_credentials: Neo4jCredentials) -> None:
    """Test include types from schema."""
    graph = Neo4jGraph(**neo4j_credentials)
    # Delete all nodes in the graph
    graph.query("MATCH (n) DETACH DELETE n")
    # Create two nodes and a relationship
    graph.query(
        "CREATE (a:Actor {name:'Bruce Willis'})"
        "-[:ACTED_IN]->(:Movie {title: 'Pulp Fiction'})"
        "<-[:DIRECTED]-(p:Person {name:'John'})"
    )
    # Refresh schema information
    graph.refresh_schema()

    llm = MagicMock(spec=BaseLanguageModel)
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        include_types=["Movie", "ACTED_IN"],
        allow_dangerous_requests=True,
    )
    expected_schema = (
        "Node properties:\n"
        "Movie {title: STRING}\n"
        "Relationship properties:\n\n"
        "The relationships:\n"
    )
    assert chain.graph_schema == expected_schema
