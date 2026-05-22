# 🦜️🔗 LangChain Neo4j

This package contains the LangChain integration with Neo4j.

## 📦 Installation

```bash
pip install -U langchain-neo4j
```

## 💻 Examples

### Neo4jGraph

The `Neo4jGraph` class is a wrapper around Neo4j's Python driver.
It provides a simple interface for interacting with a Neo4j database.

```python
from langchain_neo4j import Neo4jGraph

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
graph.query("MATCH (n) RETURN n LIMIT 1;")
```

### Neo4jChatMessageHistory

The `Neo4jChatMessageHistory` class is used to store chat message history in a Neo4j database.
It stores messages as nodes and creates relationships between them, allowing for easy querying of the conversation history.

```python
from langchain_neo4j import Neo4jChatMessageHistory

history = Neo4jChatMessageHistory(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password",
    session_id="session_id_1",
)
history.add_user_message("hi!")
history.add_ai_message("whats up?")
history.messages
```

### Neo4jVector

The `Neo4jVector` class provides functionality for managing a Neo4j vector store.
It enables you to create new vector indexes, add vectors to existing indexes, and perform queries on indexes.

```python
from langchain_classic.docstore.document import Document
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings

# Create a vector store from some documents and embeddings
docs = [
    Document(
        page_content=(
            "LangChain is a framework to build "
            "with LLMs by chaining interoperable components."
        ),
    )
]
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key="sk-...",  # Replace with your OpenAI API key
)
db = Neo4jVector.from_documents(
    docs,
    embeddings,
    url="bolt://localhost:7687",
    username="neo4j",
    password="password",
)
# Query the vector store for similar documents
docs_with_score = db.similarity_search_with_score("What is LangChain?", k=1)
```

### GraphCypherQAChain

The `CypherQAChain` class enables natural language interactions with a Neo4j database.
It uses an LLM and the database's schema to translate a user's question into a Cypher query, which is executed against the database.
The resulting data is then sent along with the user's question to the LLM to generate a natural language response.

```python
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(
    temperature=0,
    api_key="sk-...",  # Replace with your OpenAI API key
)
graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
chain = GraphCypherQAChain.from_llm(llm=llm, graph=graph, allow_dangerous_requests=True)
chain.invoke({"query": "Who starred in Top Gun?"})
```

### Neo4jSaver

The `Neo4jSaver` provides a synchronous Neo4j checkpoint saver for LangGraph.

This class implements the `BaseCheckpointSaver` interface using Neo4j
as the persistence backend with a proper graph model. It supports
storing checkpoints, channel states, and pending writes using
relationships for efficient traversal.

```python
from langchain_neo4j import Neo4jSaver
# Using from_conn_string (recommended)
with Neo4jSaver.from_conn_string(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
) as checkpointer:
    checkpointer.setup()  # Create indexes (run once)
    graph = builder.compile(checkpointer=checkpointer)
    result = graph.invoke({"messages": [...]}, config)

# Using existing driver
driver = GraphDatabase.driver(uri, auth=(user, password))
checkpointer = Neo4jSaver(driver)
checkpointer.setup()
```

### AsyncNeo4jSaver

The `AsyncNeo4jSaver` provides asynchronous Neo4j checkpoint saver for LangGraph.

This class implements the `BaseCheckpointSaver` interface using Neo4j
as the persistence backend with async support and a proper graph model.
It supports storing checkpoints, channel states, and pending writes
using relationships for efficient traversal.

```python
# Using from_conn_string (recommended)
async with await AsyncNeo4jSaver.from_conn_string(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
) as checkpointer:
    await checkpointer.setup()  # Create indexes (run once)
    graph = builder.compile(checkpointer=checkpointer)
    result = await graph.ainvoke({"messages": [...]}, config)

# Using existing async driver
driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
checkpointer = AsyncNeo4jSaver(driver)
await checkpointer.setup()
```

## 🧪 Tests

### Unit Tests

Run the unit tests using:

```bash
make tests
```

### Integration Tests

1. Start the Neo4j instance using Docker:

    ```bash
    cd tests/integration_tests/docker-compose
    docker-compose -f neo4j.yml up
    ```

2. Run the tests:

    ```bash
    make integration_tests
    ```

## 🧹 Code Formatting and Linting

To format your code, run:

```bash
make format
```

To lint it, run:

```bash
make lint
```

To format the `pyproject. ` file, run:

```bash
uv run pyprojectsort pyproject.toml
```
