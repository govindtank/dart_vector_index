import 'dart:convert';
import 'filter.dart';
import 'flat_index.dart';
import 'hnsw_index.dart';
import 'index.dart';
import 'metric.dart';
import 'persistence_stub.dart' if (dart.library.io) 'persistence_io.dart'
    as persistence;
import 'quantization.dart';
import 'record.dart';

/// Available vector indexing algorithm strategies.
enum IndexType {
  /// Hierarchical Navigable Small World graph for fast $O(\log N)$ approximate search.
  hnsw,

  /// Brute-force exhaustive search for exact 100% ground-truth recall.
  flat,
}

/// A developer-friendly on-device vector store and retrieval engine for RAG,
/// semantic similarity, and AI embeddings.
class VectorStore {
  final VectorIndex _index;
  final IndexType _indexType;
  final ScalarQuantizer? _quantizer;

  /// Creates a [VectorStore] instance.
  ///
  /// - [type]: Indexing strategy ([IndexType.hnsw] or [IndexType.flat]).
  /// - [metric]: Distance metric (defaults to [DistanceMetric.cosine]).
  /// - [m]: Number of bidirectional connections per node for HNSW (defaults to 16).
  /// - [efConstruction]: Size of dynamic candidate list during HNSW graph construction (defaults to 64).
  /// - [efSearch]: Size of dynamic candidate list during HNSW search queries (defaults to 32).
  /// - [useQuantization]: If true, applies 8-bit Scalar Quantization (SQ8) to compress vectors.
  /// - [randomSeed]: Optional random seed for deterministic graph builds.
  VectorStore({
    IndexType type = IndexType.hnsw,
    DistanceMetric metric = DistanceMetric.cosine,
    int m = 16,
    int efConstruction = 64,
    int efSearch = 32,
    bool useQuantization = false,
    int? randomSeed,
  })  : _indexType = type,
        _quantizer = useQuantization ? ScalarQuantizer() : null,
        _index = type == IndexType.hnsw
            ? HnswIndex(
                metric: metric,
                m: m,
                efConstruction: efConstruction,
                efSearch: efSearch,
                randomSeed: randomSeed,
              )
            : FlatIndex(metric: metric);

  VectorStore._({
    required VectorIndex index,
    required IndexType indexType,
    ScalarQuantizer? quantizer,
  })  : _index = index,
        _indexType = indexType,
        _quantizer = quantizer;

  /// Underlying vector index strategy.
  VectorIndex get index => _index;

  /// Index type strategy used by this store.
  IndexType get indexType => _indexType;

  /// Distance metric configured for this store.
  DistanceMetric get metric => _index.metric;

  /// Total number of vector records indexed.
  int get count => _index.count;

  /// Dimensionality of the vectors in this store.
  int get dimension => _index.dimension;

  /// Whether 8-bit Scalar Quantization is enabled.
  bool get isQuantized => _quantizer != null;

  /// Adds a single record to the store.
  void add({
    required String id,
    required List<num> vector,
    Map<String, dynamic>? metadata,
    String? tag,
  }) {
    final record = VectorRecord(
      id: id,
      vector: vector,
      metadata: metadata,
      tag: tag,
    );
    _index.insert(record);
  }

  /// Inserts a pre-constructed [VectorRecord].
  void addRecord(VectorRecord record) => _index.insert(record);

  /// Inserts multiple [records] in a single batch.
  void addBatch(Iterable<VectorRecord> records) => _index.insertAll(records);

  /// Retrieves a stored record by [id], or returns `null` if not present.
  VectorRecord? get(String id) => _index.get(id);

  /// Checks if a record with [id] exists in the store.
  bool contains(String id) => _index.contains(id);

  /// Removes a record by [id]. Returns `true` if removed.
  bool remove(String id) => _index.remove(id);

  /// Clears all records from the store.
  void clear() => _index.clear();

  /// Searches for the top [limit] most similar records to [query].
  ///
  /// - [query]: Raw floating point embedding vector.
  /// - [limit]: Maximum number of search results to return (defaults to 10).
  /// - [filter]: Optional [MetadataFilter] predicate.
  /// - [minScore]: Minimum normalized similarity score threshold in `[0.0, 1.0]`.
  /// - [maxDistance]: Maximum allowed distance threshold.
  /// - [includeVectors]: Whether to return raw embedding vectors in the results.
  List<SearchResult> search(
    List<num> query, {
    int limit = 10,
    MetadataFilter? filter,
    double? minScore,
    double? maxDistance,
    bool includeVectors = false,
  }) {
    return _index.search(
      query,
      topK: limit,
      filter: filter,
      minScore: minScore,
      maxDistance: maxDistance,
      includeVectors: includeVectors,
    );
  }

  /// Convenience alias for semantic similarity search.
  List<SearchResult> similaritySearch(
    List<num> query, {
    int limit = 10,
    MetadataFilter? filter,
    double? minScore,
  }) {
    return search(query, limit: limit, filter: filter, minScore: minScore);
  }

  /// Searches for nearest neighbors for multiple query vectors in batch.
  List<List<SearchResult>> batchSearch(
    List<List<num>> queries, {
    int limit = 10,
    MetadataFilter? filter,
    double? minScore,
    double? maxDistance,
    bool includeVectors = false,
  }) {
    return queries
        .map((q) => search(
              q,
              limit: limit,
              filter: filter,
              minScore: minScore,
              maxDistance: maxDistance,
              includeVectors: includeVectors,
            ))
        .toList();
  }

  /// Serializes the entire vector store into a JSON string.
  String saveToJsonString({bool pretty = false}) {
    final q = _quantizer;
    final map = <String, dynamic>{
      'version': '1.0.0',
      'indexType': _indexType.name,
      'isQuantized': isQuantized,
      if (q != null) 'quantizer': q.toJson(),
      'index': _index.toJson(),
    };

    if (pretty) {
      return const JsonEncoder.withIndent('  ').convert(map);
    }
    return jsonEncode(map);
  }

  /// Restores a [VectorStore] from a serialized [jsonString].
  static VectorStore loadFromJsonString(String jsonString) {
    final map = jsonDecode(jsonString) as Map<String, dynamic>;
    final indexTypeStr = map['indexType'] as String? ?? 'hnsw';
    final indexType = indexTypeStr == 'flat' ? IndexType.flat : IndexType.hnsw;

    final quantizerMap = map['quantizer'] as Map<String, dynamic>?;
    final quantizer =
        quantizerMap != null ? ScalarQuantizer.fromJson(quantizerMap) : null;

    final indexMap = map['index'] as Map<String, dynamic>;
    final index = indexType == IndexType.flat
        ? FlatIndex.fromJson(indexMap)
        : HnswIndex.fromJson(indexMap);

    return VectorStore._(
      index: index,
      indexType: indexType,
      quantizer: quantizer,
    );
  }

  /// Persists the vector store to a local [file] (or file path).
  ///
  /// On Native platforms (Android, iOS, Desktop), performs an atomic save via a temporary file.
  Future<void> saveToFile(dynamic file) async {
    final jsonStr = saveToJsonString();
    await persistence.saveStringToFile(file, jsonStr);
  }

  /// Loads and restores a [VectorStore] from a local [file] (or file path).
  static Future<VectorStore> loadFromFile(dynamic file) async {
    final content = await persistence.readStringFromFile(file);
    return loadFromJsonString(content);
  }
}
