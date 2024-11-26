# Changelog

## Next

### Changed

- Removed dependency on LangChain Community package by integrating necessary components directly into the LangChain Neo4j codebase.

### Updated

- Fixed bugs in the Neo4jVector and GraphCypherQAChain classes preventing these classes from working with versions < 5.23 of Neo4j.

## 0.1.0

### Added

- Migrated all Neo4j-related code, including tests and integration tests, from the LangChain Community package to this package.
