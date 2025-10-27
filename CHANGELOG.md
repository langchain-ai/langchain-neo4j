# Changelog

## Next

## 0.6.0

### Added

- Added `langchain-classic` as a dependency.

### Changed

- Removed `langchain` as a dependency.
- Removed support for Python 3.9 as `langchain-classic` no longer supports it.

## 0.5.0

### Added

- Added support for specifying custom database names in `Neo4jChatMessageHistory` (not limited to the default `"neo4j"`).  
- Introduced an optional `mmr` dependency group for maximal marginal relevance search dependencies.
- Included `numpy` in the `mmr` dependency group.

### Changed

- Bumped `neo4j-graphrag` dependency version from `^1.5.0` to `^1.9.0`.  

### Fixed

- Prevented infinite loops in `CypherQueryCorrector.extract_paths` whenever there was a repeated path in a Cypher query.
- Replaced direct `result["text"]` indexing with `result.get("text")` in `Neo4jVector.similarity_search_with_score_by_vector`, so missing text fields are handled gracefully instead of raising errors.
- Changed an empty-list check in `CypherQueryCorrector.detect_node_variables` so that nodes without an explicit variable are properly skipped instead of misclassifying all labels under an empty variable.
- Stopped `Neo4jChatMessageHistory` from creating duplicate session nodes.
- Schema bug in `Neo4jGraph` when `enhance_schema=True` where relationships with boolean properties triggered a malformed `RETURN {} AS output` Cypher query and prevented graph initialization.
- Ensured custom DB names are passed through to `retrieve_vector_index_info` and `retrieve_existing_fts_index` in `Neo4jVector`.  

## 0.4.0

### Changed

- Renamed the `type` property to `role` on `Message` nodes in `Neo4jChatMessageHistory`.
- Updated `GraphCypherQAChain` to use the same schema format as `Neo4jGraph`.
- Replaced code used to query vector and full text indexes in the `vectorstores.neo4j_vector` module with equivalents from the `neo4j-graphrag` package.
- Replaced database schema retrieval code in the `graphs.neo4j_graph` module with equivalents from the `neo4j-graphrag` package.
- Replaced the Cypher queries in `Neo4jChatMessageHistory` with equivalents from the `neo4j-graphrag` package.
- Updated the `construct_schema` function used by the `GraphCypherQAChain` to use schema retrieval functions from the `neo4j-graphrag` package.
- Replaced the legacy `extract_cypher` function with the improved version from `neo4j-graphrag` package.

### Added

- Introduced a `delete_session_node` parameter to the `clear` method in `Neo4jChatMessageHistory` for optional deletion of the `Session` node.
- Added `neo4j-graphrag` to the dependencies.

## 0.3.0

### Added

- Optional parameter to specify embedding dimension in `Neo4jVector`, avoiding the need to query the embedding model.

### Changed

- Made the `source` parameter of `GraphDocument` optional and updated related methods to support this.
- Suppressed AggregationSkippedNull warnings raised by the Neo4j driver in the `Neo4jGraph` class when fetching the enhanced_schema.
- Modified the `Neo4jGraph` class's enhanced schema Cypher query to utilize the apoc.meta.graph procedure instead of apoc.meta.graphSample.
- Updated `GraphStore` to be a Protocol, enabling compatibility with `GraphCypherQAChain` without requiring inheritance.

### Fixed

- Resolved syntax errors in `GraphCypherQAChain` by ensuring node labels with spaces are correctly quoted in Cypher queries.
- Added missing Lucene special character '/' to the list of characters escaped in `remove_lucene_chars`.

## 0.2.0

### Added

- Enhanced Neo4j driver connection management with more robust error handling.
- Simplified connection state checking in Neo4jGraph.
- Introduced `effective_search_ratio` parameter in Neo4jVector to enhance query accuracy by adjusting the candidate pool size during similarity searches.

### Fixed

- Removed deprecated LLMChain from GraphCypherQAChain to resolve instantiation issues with the use_function_response parameter.
- Removed unnecessary # type: ignore comments, improving type safety and code clarity.

## 0.1.1

### Changed

- Removed dependency on LangChain Community package by integrating necessary components directly into the LangChain Neo4j codebase.

### Updated

- Fixed bugs in the Neo4jVector and GraphCypherQAChain classes preventing these classes from working with versions < 5.23 of Neo4j.

## 0.1.0

### Added

- Migrated all Neo4j-related code, including tests and integration tests, from the LangChain Community package to this package.
