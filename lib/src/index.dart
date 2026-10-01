import 'filter.dart';
import 'metric.dart';
import 'record.dart';

/// Abstract contract for high-performance vector search indexes.
abstract class VectorIndex {
  /// Default abstract constructor for [VectorIndex].
  const VectorIndex();

  /// Total number of vector records currently stored in this index.
  int get count;

  /// Dimensionality of the vectors indexed by this instance, or -1 if empty.
  int get dimension;

  /// The distance metric used for ranking similarity.
  DistanceMetric get metric;

  /// Inserts a single [record] into the index.
  ///
  /// If a record with the same [VectorRecord.id] already exists, it will be updated.
  void insert(VectorRecord record);

  /// Inserts multiple [records] in batch.
  void insertAll(Iterable<VectorRecord> records);

  /// Removes the record with identifier [id] from the index.
  ///
  /// Returns `true` if the record existed and was removed, `false` otherwise.
  bool remove(String id);

  /// Retrieves the [VectorRecord] associated with [id], or `null` if not found.
  VectorRecord? get(String id);

  /// Returns `true` if a record with [id] is present in the index.
  bool contains(String id);

  /// Searches the index for the [topK] nearest neighbors to [query].
  ///
  /// - [query]: Dense target query vector.
  /// - [topK]: Maximum number of nearest neighbors to return (default 10).
  /// - [filter]: Optional [MetadataFilter] to prune candidates.
  /// - [minScore]: Optional minimum similarity score threshold in range `[0.0, 1.0]`.
  /// - [maxDistance]: Optional maximum distance threshold.
  /// - [includeVectors]: Whether to include the raw embedding vectors in the returned [SearchResult]s.
  List<SearchResult> search(
    List<num> query, {
    int topK = 10,
    MetadataFilter? filter,
    double? minScore,
    double? maxDistance,
    bool includeVectors = false,
  });

  /// Removes all records and resets the index.
  void clear();

  /// Serializes the index to a JSON-compatible map.
  Map<String, dynamic> toJson();
}
