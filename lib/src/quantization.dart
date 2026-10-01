import 'dart:typed_data';

/// 8-bit Scalar Quantization (SQ8) utility for compressing 32-bit floating point vectors
/// into 8-bit unsigned integers, cutting RAM usage by 75% (4x compression).
class ScalarQuantizer {
  /// Global or trained minimum value across all vector components.
  final double minVal;

  /// Global or trained maximum value across all vector components.
  final double maxVal;

  /// Scale factor: `255.0 / (maxVal - minVal)`.
  final double scale;

  /// Inverse scale factor for fast dequantization.
  final double invScale;

  /// Creates a [ScalarQuantizer] with fixed bounds [minVal] and [maxVal].
  ScalarQuantizer({this.minVal = -1.0, this.maxVal = 1.0})
      : assert(maxVal > minVal, 'maxVal must be strictly greater than minVal'),
        scale = 255.0 / (maxVal - minVal),
        invScale = (maxVal - minVal) / 255.0;

  /// Fits a [ScalarQuantizer] by computing the minimum and maximum component
  /// values observed across a sample batch of [vectors].
  factory ScalarQuantizer.fit(Iterable<Float32List> vectors) {
    if (vectors.isEmpty) {
      return ScalarQuantizer();
    }

    double globalMin = double.infinity;
    double globalMax = -double.infinity;

    for (final vec in vectors) {
      for (var i = 0; i < vec.length; i++) {
        final v = vec[i];
        if (v < globalMin) {
          globalMin = v;
        }
        if (v > globalMax) {
          globalMax = v;
        }
      }
    }

    if (globalMin == globalMax ||
        globalMin.isInfinite ||
        globalMax.isInfinite) {
      return ScalarQuantizer(minVal: -1.0, maxVal: 1.0);
    }

    // Add tiny margin to prevent boundary overflow
    final span = globalMax - globalMin;
    return ScalarQuantizer(
      minVal: globalMin - span * 0.01,
      maxVal: globalMax + span * 0.01,
    );
  }

  /// Quantizes a 32-bit floating point [vector] into a compact [Uint8List].
  Uint8List quantize(Float32List vector) {
    final len = vector.length;
    final quantized = Uint8List(len);

    for (var i = 0; i < len; i++) {
      final v = vector[i];
      final normalized = (v - minVal) * scale;
      final q = normalized.round().clamp(0, 255);
      quantized[i] = q;
    }

    return quantized;
  }

  /// Dequantizes an 8-bit [quantized] vector back into a [Float32List].
  Float32List dequantize(Uint8List quantized) {
    final len = quantized.length;
    final floatList = Float32List(len);

    for (var i = 0; i < len; i++) {
      floatList[i] = minVal + (quantized[i] * invScale);
    }

    return floatList;
  }

  /// Computes approximate squared Euclidean distance directly between two quantized vectors.
  double quantizedSquaredEuclidean(Uint8List a, Uint8List b) {
    final len = a.length;
    var sumDiffSq = 0;

    for (var i = 0; i < len; i++) {
      final diff = a[i] - b[i];
      sumDiffSq += diff * diff;
    }

    // Scale back to original domain
    return sumDiffSq * invScale * invScale;
  }

  /// Computes approximate dot product directly between two quantized vectors.
  double quantizedDotProduct(Uint8List a, Uint8List b) {
    final len = a.length;
    var rawSum = 0.0;

    for (var i = 0; i < len; i++) {
      final valA = minVal + (a[i] * invScale);
      final valB = minVal + (b[i] * invScale);
      rawSum += valA * valB;
    }

    return rawSum;
  }

  /// Serializes configuration to a JSON map.
  Map<String, dynamic> toJson() => {
        'minVal': minVal,
        'maxVal': maxVal,
      };

  /// Deserializes a [ScalarQuantizer] from a JSON map.
  factory ScalarQuantizer.fromJson(Map<String, dynamic> json) {
    return ScalarQuantizer(
      minVal: (json['minVal'] as num).toDouble(),
      maxVal: (json['maxVal'] as num).toDouble(),
    );
  }
}
