## 1.0.0

- Initial stable release of `dart_vector_index`.
- Pure Dart vector indexing and nearest-neighbor search engine.
- Support for `FlatIndex` (exact search) and `HnswIndex` (hierarchical navigable small world graph).
- Multiple distance metrics: `cosine`, `euclidean`, `dotProduct`, `manhattan`, `chebyshev`.
- Scalar Quantization (SQ8) for 4x memory reduction on mobile devices.
- Metadata storage and rich predicate filtering (`eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `inList`, `contains`, `and`, `or`, `not`).
- High-level `VectorStore` with atomic JSON serialization, deserialization, batch upsert, and deletion.
- SIMD and `Float32List` optimized distance math.
