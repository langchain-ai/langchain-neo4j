# Changelog

## Next

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
