import 'dart:math' as math;
import 'dart:typed_data';

/// Distance metric used for measuring similarity or divergence between vectors.
enum DistanceMetric {
  /// Cosine distance: `1.0 - cosine_similarity`.
  ///
  /// Normalized range: `[0.0, 2.0]`. Distance of 0.0 means identical angle/direction.
  cosine,

  /// Standard Euclidean ($L_2$) distance: `sqrt(sum((a_i - b_i)^2))`.
  ///
  /// Distance of 0.0 means identical vectors.
  euclidean,

  /// Dot product (inner product) distance: `- (a · b)` or `1.0 - (a · b)` for normalized vectors.
  ///
  /// Ideal for normalized embeddings produced by OpenAI, Cohere, or Gemma.
  dotProduct,

  /// Manhattan ($L_1$) distance (Taxicab / City Block): `sum(|a_i - b_i|)`.
  manhattan,

  /// Chebyshev ($L_\infty$) distance: `max(|a_i - b_i|)`.
  chebyshev,
}

/// Utility for high-performance vector math and distance computation.
class VectorMath {
  VectorMath._();

  /// Calculates the distance between two vectors [a] and [b] according to [metric].
  static double distance(
    Float32List a,
    Float32List b,
    DistanceMetric metric,
  ) {
    if (a.length != b.length) {
      throw ArgumentError(
        'Vector dimensions must match. Got ${a.length} and ${b.length}.',
      );
    }

    switch (metric) {
      case DistanceMetric.cosine:
        return cosineDistance(a, b);
      case DistanceMetric.euclidean:
        return euclideanDistance(a, b);
      case DistanceMetric.dotProduct:
        return dotProductDistance(a, b);
      case DistanceMetric.manhattan:
        return manhattanDistance(a, b);
      case DistanceMetric.chebyshev:
        return chebyshevDistance(a, b);
    }
  }

  /// Converts a [distance] value into a normalized similarity score in the range `[0.0, 1.0]`.
  ///
  /// A score of 1.0 indicates identity, while 0.0 indicates maximal divergence.
  static double distanceToScore(double distance, DistanceMetric metric) {
    if (distance.isNaN || distance.isInfinite) {
      return 0.0;
    }
    switch (metric) {
      case DistanceMetric.cosine:
        // Cosine distance is in [0, 2]. Similarity = 1 - (dist / 2) or (2 - dist) / 2
        final s = 1.0 - (distance / 2.0);
        return s.clamp(0.0, 1.0);
      case DistanceMetric.euclidean:
      case DistanceMetric.manhattan:
      case DistanceMetric.chebyshev:
        // 1 / (1 + distance) maps [0, inf) to [1, 0]
        return (1.0 / (1.0 + math.max(0.0, distance))).clamp(0.0, 1.0);
      case DistanceMetric.dotProduct:
        // Sigmoid mapping for unbounded dot product: 1 / (1 + exp(distance))
        if (distance <= -20.0) {
          return 1.0;
        }
        if (distance >= 20.0) {
          return 0.0;
        }
        return (1.0 / (1.0 + math.exp(distance))).clamp(0.0, 1.0);
    }
  }

  /// Computes the cosine distance between vectors [a] and [b].
  ///
  /// Formula: `1.0 - (dot(a, b) / (norm(a) * norm(b)))`.
  static double cosineDistance(Float32List a, Float32List b) {
    final len = a.length;
    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (var i = 0; i < len; i++) {
      final ai = a[i];
      final bi = b[i];
      dot += ai * bi;
      normA += ai * ai;
      normB += bi * bi;
    }

    if (normA <= 0.0 || normB <= 0.0) {
      return 1.0; // Orthogonal / empty fallback
    }

    final cosineSimilarity = dot / (math.sqrt(normA) * math.sqrt(normB));
    // Clamp to [-1.0, 1.0] to guard against floating-point inaccuracies
    final clamped = cosineSimilarity.clamp(-1.0, 1.0);
    return 1.0 - clamped;
  }

  /// Computes the Euclidean ($L_2$) distance between vectors [a] and [b].
  static double euclideanDistance(Float32List a, Float32List b) {
    return math.sqrt(squaredEuclideanDistance(a, b));
  }

  /// Computes the squared Euclidean distance between vectors [a] and [b].
  ///
  /// Faster than [euclideanDistance] when only relative rankings are needed.
  static double squaredEuclideanDistance(Float32List a, Float32List b) {
    final len = a.length;
    double sum = 0.0;
    for (var i = 0; i < len; i++) {
      final diff = a[i] - b[i];
      sum += diff * diff;
    }
    return sum;
  }

  /// Computes the Dot Product distance between vectors [a] and [b].
  ///
  /// Formula: `- dot(a, b)` so that higher dot product yields smaller distance.
  static double dotProductDistance(Float32List a, Float32List b) {
    final len = a.length;
    double dot = 0.0;
    for (var i = 0; i < len; i++) {
      dot += a[i] * b[i];
    }
    return -dot;
  }

  /// Computes the Manhattan ($L_1$) distance between vectors [a] and [b].
  static double manhattanDistance(Float32List a, Float32List b) {
    final len = a.length;
    double sum = 0.0;
    for (var i = 0; i < len; i++) {
      sum += (a[i] - b[i]).abs();
    }
    return sum;
  }

  /// Computes the Chebyshev ($L_\infty$) distance between vectors [a] and [b].
  static double chebyshevDistance(Float32List a, Float32List b) {
    final len = a.length;
    double maxDiff = 0.0;
    for (var i = 0; i < len; i++) {
      final diff = (a[i] - b[i]).abs();
      if (diff > maxDiff) {
        maxDiff = diff;
      }
    }
    return maxDiff;
  }

  /// Computes the $L_2$ norm (Euclidean length) of [vector].
  static double l2Norm(Float32List vector) {
    double sum = 0.0;
    for (var i = 0; i < vector.length; i++) {
      final v = vector[i];
      sum += v * v;
    }
    return math.sqrt(sum);
  }

  /// Returns a normalized unit vector with $L_2$ norm equal to 1.0.
  static Float32List normalize(Float32List vector) {
    final norm = l2Norm(vector);
    if (norm == 0.0) {
      return Float32List.fromList(vector);
    }
    final result = Float32List(vector.length);
    for (var i = 0; i < vector.length; i++) {
      result[i] = vector[i] / norm;
    }
    return result;
  }

  /// Converts a generic `List<num>` or `List<double>` into a high-performance [Float32List].
  static Float32List toFloat32List(List<num> list) {
    if (list is Float32List) {
      return list;
    }
    final result = Float32List(list.length);
    for (var i = 0; i < list.length; i++) {
      result[i] = list[i].toDouble();
    }
    return result;
  }
}
