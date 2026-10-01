import 'dart:typed_data';
import 'metric.dart';

/// A record stored within the vector index containing its embedding and metadata.
class VectorRecord {
  /// Unique identifier for this vector record.
  final String id;

  /// High-dimensional dense embedding vector.
  final Float32List vector;

  /// Optional arbitrary metadata payload associated with this vector.
  final Map<String, dynamic> metadata;

  /// Optional categorical grouping tag for quick partition filtering.
  final String? tag;

  /// Creates a [VectorRecord].
  VectorRecord({
    required this.id,
    required List<num> vector,
    Map<String, dynamic>? metadata,
    this.tag,
  })  : vector = VectorMath.toFloat32List(vector),
        metadata = metadata == null
            ? const <String, dynamic>{}
            : Map<String, dynamic>.unmodifiable(metadata);

  /// Dimensionality of the embedding vector.
  int get dimension => vector.length;

  /// Serializes this record into a JSON-compatible Map.
  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'id': id,
      'vector': vector.toList(growable: false),
      'metadata': metadata,
      if (tag != null) 'tag': tag,
    };
  }

  /// Deserializes a [VectorRecord] from a JSON-compatible Map.
  factory VectorRecord.fromJson(Map<String, dynamic> json) {
    final rawVector = json['vector'] as List<dynamic>;
    final floatList = Float32List(rawVector.length);
    for (var i = 0; i < rawVector.length; i++) {
      floatList[i] = (rawVector[i] as num).toDouble();
    }

    return VectorRecord(
      id: json['id'] as String,
      vector: floatList,
      metadata: json['metadata'] != null
          ? Map<String, dynamic>.from(json['metadata'] as Map<dynamic, dynamic>)
          : const <String, dynamic>{},
      tag: json['tag'] as String?,
    );
  }

  @override
  String toString() =>
      'VectorRecord(id: $id, dim: $dimension, tag: $tag, metadata: $metadata)';
}

/// The result of a nearest-neighbor similarity search query.
class SearchResult implements Comparable<SearchResult> {
  /// Unique identifier of the matching record.
  final String id;

  /// Normalized similarity score in range `[0.0, 1.0]`. Higher is more similar.
  final double score;

  /// Raw computed distance value. Lower is closer.
  final double distance;

  /// Associated metadata payload.
  final Map<String, dynamic> metadata;

  /// The embedding vector if requested in search query, otherwise null.
  final Float32List? vector;

  /// Optional categorical tag.
  final String? tag;

  /// Creates a [SearchResult].
  const SearchResult({
    required this.id,
    required this.score,
    required this.distance,
    this.metadata = const <String, dynamic>{},
    this.vector,
    this.tag,
  });

  /// Serializes to a JSON-compatible Map.
  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'id': id,
      'score': score,
      'distance': distance,
      'metadata': metadata,
      if (vector != null) 'vector': vector!.toList(growable: false),
      if (tag != null) 'tag': tag,
    };
  }

  @override
  int compareTo(SearchResult other) {
    // Sort by descending score (higher score first), then ascending distance
    final scoreComp = other.score.compareTo(score);
    if (scoreComp != 0) {
      return scoreComp;
    }
    return distance.compareTo(other.distance);
  }

  @override
  String toString() =>
      'SearchResult(id: $id, score: ${score.toStringAsFixed(4)}, dist: ${distance.toStringAsFixed(4)}, metadata: $metadata)';
}
