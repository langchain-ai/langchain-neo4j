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
