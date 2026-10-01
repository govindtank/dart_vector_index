import 'dart:typed_data';
import 'filter.dart';
import 'index.dart';
import 'metric.dart';
import 'record.dart';

/// Exact, brute-force vector index providing 100% recall.
///
/// Ideal for datasets up to 10,000 vectors where exact ground-truth precision is required.
class FlatIndex implements VectorIndex {
  final Map<String, VectorRecord> _records = <String, VectorRecord>{};
  final DistanceMetric _metric;
  int _dimension = -1;

  /// Creates a [FlatIndex] configured with [metric].
  FlatIndex({DistanceMetric metric = DistanceMetric.cosine}) : _metric = metric;

  @override
  int get count => _records.length;

  @override
  int get dimension => _dimension;

  @override
  DistanceMetric get metric => _metric;

  @override
  void insert(VectorRecord record) {
    if (_dimension == -1) {
      _dimension = record.dimension;
    } else if (record.dimension != _dimension) {
      throw ArgumentError(
        'Vector dimension mismatch: expected $_dimension, got ${record.dimension}.',
      );
    }
    _records[record.id] = record;
  }

  @override
  void insertAll(Iterable<VectorRecord> records) {
    for (final record in records) {
      insert(record);
    }
  }

  @override
  bool remove(String id) {
    final removed = _records.remove(id);
    if (_records.isEmpty) {
      _dimension = -1;
    }
    return removed != null;
  }

  @override
  VectorRecord? get(String id) => _records[id];

  @override
  bool contains(String id) => _records.containsKey(id);

  @override
  List<SearchResult> search(
    List<num> query, {
    int topK = 10,
    MetadataFilter? filter,
    double? minScore,
    double? maxDistance,
    bool includeVectors = false,
  }) {
    if (_records.isEmpty || topK <= 0) {
      return const <SearchResult>[];
    }

    final queryVec =
        query is Float32List ? query : VectorMath.toFloat32List(query);

    if (_dimension != -1 && queryVec.length != _dimension) {
      throw ArgumentError(
        'Query dimension (${queryVec.length}) does not match index dimension ($_dimension).',
      );
    }

    final matches = <SearchResult>[];

    for (final record in _records.values) {
      if (filter != null &&
          !filter.evaluate(record.metadata, tag: record.tag)) {
        continue;
      }

      final dist = VectorMath.distance(queryVec, record.vector, _metric);

      if (maxDistance != null && dist > maxDistance) {
        continue;
      }

      final score = VectorMath.distanceToScore(dist, _metric);

      if (minScore != null && score < minScore) {
        continue;
      }

      matches.add(
        SearchResult(
          id: record.id,
          score: score,
          distance: dist,
          metadata: record.metadata,
          vector: includeVectors ? record.vector : null,
          tag: record.tag,
        ),
      );
    }

    matches.sort();

    if (matches.length > topK) {
      return matches.sublist(0, topK);
    }
    return matches;
  }

  @override
  void clear() {
    _records.clear();
    _dimension = -1;
  }

  @override
  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'type': 'flat',
      'metric': _metric.name,
      'dimension': _dimension,
      'records': _records.values.map((r) => r.toJson()).toList(growable: false),
    };
  }

  /// Deserializes a [FlatIndex] from a JSON-compatible map.
  factory FlatIndex.fromJson(Map<String, dynamic> json) {
    final metricName = json['metric'] as String? ?? 'cosine';
    final metric = DistanceMetric.values.firstWhere(
      (m) => m.name == metricName,
      orElse: () => DistanceMetric.cosine,
    );

    final index = FlatIndex(metric: metric);
    final recordsList = json['records'] as List<dynamic>?;
    if (recordsList != null) {
      for (final item in recordsList) {
        final record = VectorRecord.fromJson(item as Map<String, dynamic>);
        index.insert(record);
      }
    }
    return index;
  }
}
