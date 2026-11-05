import pytest

from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema

schema = Schema(left_node="Actor", relation="ACTED_IN", right_node="Movie")
corrector = CypherQueryCorrector([schema])


@pytest.mark.parametrize(
    "description, query, expected",
    [
        (
            "Correct query",
            "MATCH (a:Actor)-[r]->(b:Movie)",
            "MATCH (a:Actor)-[r]->(b:Movie)",
        ),
        (
            "Reversed relation",
            "MATCH (a:Actor)<-[r]-(b:Movie)",
            "MATCH (a:Actor)-[r]->(b:Movie)",
        ),
        (
            "Repeated query",
            "MATCH (a)-[r]->(b) MATCH (a)-[r]->(b)",
            "MATCH (a)-[r]->(b) MATCH (a)-[r]->(b)",
        ),
        (
            "Query doesn't match schema returns empty",
            "MATCH (a:Director)-[r:ACTED_IN]->(b:Movie)",
            "",
        ),
    ],
)
def test_cypher_query_corrector(description: str, query: str, expected: str) -> None:
    assert corrector.correct_query(query) == expected, f"{description} failed"


geography_corrector = CypherQueryCorrector(
    [
        Schema(left_node="Country", relation="locatedIn", right_node="Continent"),
        Schema(left_node="River", relation="flowsThrough", right_node="Country"),
    ]
)


@pytest.mark.parametrize(
    "description, query, expected",
    [
        (
            "Unnamed node with parameter property map",
            "MATCH (:Country {name: $country_name})-[:locatedIn]->(c:Continent) "
            "RETURN c.name",
            "MATCH (:Country {name: $country_name})-[:locatedIn]->(c:Continent) "
            "RETURN c.name",
        ),
        (
            "Unnamed node with literal property map",
            "MATCH (:Country {name: 'Honduras'})-[:locatedIn]->(c:Continent) "
            "RETURN c.name",
            "MATCH (:Country {name: 'Honduras'})-[:locatedIn]->(c:Continent) "
            "RETURN c.name",
        ),
        (
            "Unnamed node with property map on the right",
            "MATCH (r:River)-[:flowsThrough]->(:Country {name: $n}) RETURN r.name",
            "MATCH (r:River)-[:flowsThrough]->(:Country {name: $n}) RETURN r.name",
        ),
        (
            "Reversed relation with unnamed node with property map",
            "MATCH (c:Continent)-[:locatedIn]->(:Country {name: $n}) RETURN c.name",
            "MATCH (c:Continent)<-[:locatedIn]-(:Country {name: $n}) RETURN c.name",
        ),
        (
            "Unnamed node with property map and relation not in schema",
            "MATCH (:Country {name: $n})-[:flowsThrough]->(c:Continent) RETURN c.name",
            "",
        ),
        (
            "Named node with property map",
            "MATCH (x:Country {name: $n})-[:locatedIn]->(c:Continent) RETURN c.name",
            "MATCH (x:Country {name: $n})-[:locatedIn]->(c:Continent) RETURN c.name",
        ),
        (
            "Unnamed node without property map",
            "MATCH (:Country)-[:locatedIn]->(c:Continent) RETURN c.name",
            "MATCH (:Country)-[:locatedIn]->(c:Continent) RETURN c.name",
        ),
    ],
)
def test_cypher_query_corrector_unnamed_node_with_properties(
    description: str, query: str, expected: str
) -> None:
    assert geography_corrector.correct_query(query) == expected, f"{description} failed"
