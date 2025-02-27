from typing import TypedDict


class Neo4jCredentials(TypedDict):
    url: str
    username: str
    password: str
