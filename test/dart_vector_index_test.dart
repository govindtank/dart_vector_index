import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:dart_vector_index/dart_vector_index.dart';
import 'package:test/test.dart';

void main() {
  group('VectorMath & Metrics', () {
    test('Cosine distance properties', () {
      final a = Float32List.fromList([1.0, 0.0, 0.0]);
      final b = Float32List.fromList([1.0, 0.0, 0.0]);
      final c = Float32List.fromList([0.0, 1.0, 0.0]);
      final d = Float32List.fromList([-1.0, 0.0, 0.0]);

      expect(VectorMath.cosineDistance(a, b), closeTo(0.0, 1e-6));
      expect(VectorMath.cosineDistance(a, c), closeTo(1.0, 1e-6));
      expect(VectorMath.cosineDistance(a, d), closeTo(2.0, 1e-6));
    });

    test('Euclidean distance properties', () {
      final a = Float32List.fromList([0.0, 0.0]);
      final b = Float32List.fromList([3.0, 4.0]);

      expect(VectorMath.euclideanDistance(a, b), closeTo(5.0, 1e-6));
      expect(VectorMath.squaredEuclideanDistance(a, b), closeTo(25.0, 1e-6));
    });

    test('Manhattan and Chebyshev distance', () {
      final a = Float32List.fromList([1.0, 2.0, 3.0]);
      final b = Float32List.fromList([4.0, 6.0, 3.0]);

      expect(VectorMath.manhattanDistance(a, b),
          closeTo(7.0, 1e-6)); // |3|+|4|+|0|
      expect(
          VectorMath.chebyshevDistance(a, b), closeTo(4.0, 1e-6)); // max(3,4,0)
    });

    test('Vector normalization and L2 norm', () {
      final v = Float32List.fromList([3.0, 4.0]);
      expect(VectorMath.l2Norm(v), closeTo(5.0, 1e-6));

      final norm = VectorMath.normalize(v);
      expect(norm[0], closeTo(0.6, 1e-6));
      expect(norm[1], closeTo(0.8, 1e-6));
      expect(VectorMath.l2Norm(norm), closeTo(1.0, 1e-6));
    });

    test('distanceToScore normalization', () {
      expect(
        VectorMath.distanceToScore(0.0, DistanceMetric.cosine),
        closeTo(1.0, 1e-6),
      );
      expect(
        VectorMath.distanceToScore(2.0, DistanceMetric.cosine),
        closeTo(0.0, 1e-6),
      );
      expect(
        VectorMath.distanceToScore(0.0, DistanceMetric.euclidean),
        closeTo(1.0, 1e-6),
      );
    });
  });

  group('MetadataFilter', () {
    final meta = <String, dynamic>{
      'category': 'technology',
      'rating': 4.8,
      'views': 1200,
      'tags': ['ai', 'flutter', 'dart'],
    };

    test('Comparison filters', () {
      expect(Filter.eq('category', 'technology').evaluate(meta), isTrue);
      expect(Filter.eq('category', 'sports').evaluate(meta), isFalse);
      expect(Filter.neq('category', 'sports').evaluate(meta), isTrue);

      expect(Filter.gt('rating', 4.0).evaluate(meta), isTrue);
      expect(Filter.gt('rating', 5.0).evaluate(meta), isFalse);
      expect(Filter.gte('rating', 4.8).evaluate(meta), isTrue);
      expect(Filter.lt('views', 2000).evaluate(meta), isTrue);
      expect(Filter.lte('views', 1200).evaluate(meta), isTrue);
    });

    test('InList and Contains filters', () {
      expect(
        Filter.inList('category', ['technology', 'science']).evaluate(meta),
        isTrue,
      );
      expect(
        Filter.inList('category', ['finance', 'sports']).evaluate(meta),
        isFalse,
      );
      expect(Filter.contains('tags', 'flutter').evaluate(meta), isTrue);
      expect(Filter.contains('tags', 'golang').evaluate(meta), isFalse);
    });

    test('Composite filters: and, or, not', () {
      final filter = Filter.and([
        Filter.gte('rating', 4.5),
        Filter.or([
          Filter.eq('category', 'technology'),
          Filter.eq('category', 'finance'),
        ]),
        Filter.not(Filter.lt('views', 1000)),
      ]);

      expect(filter.evaluate(meta), isTrue);
    });

    test('Tag filter', () {
      expect(Filter.tag('doc_v1').evaluate(meta, tag: 'doc_v1'), isTrue);
      expect(Filter.tag('doc_v1').evaluate(meta, tag: 'doc_v2'), isFalse);
    });

    test('Filter JSON serialization roundtrip', () {
      final filter = Filter.and([
        Filter.eq('lang', 'dart'),
        Filter.gt('score', 80),
        Filter.inList('status', ['active', 'verified']),
      ]);

      final json = filter.toJson();
      final restored = MetadataFilter.fromJson(json);

      final sampleMeta = <String, dynamic>{
        'lang': 'dart',
        'score': 95,
        'status': 'verified',
      };
      expect(restored.evaluate(sampleMeta), isTrue);
    });
  });

  group('ScalarQuantizer (SQ8)', () {
    test('Quantization & dequantization precision', () {
      final quantizer = ScalarQuantizer(minVal: -2.0, maxVal: 2.0);
      final original = Float32List.fromList([-1.5, -0.5, 0.0, 0.75, 1.8]);

      final quantized = quantizer.quantize(original);
      expect(quantized.length, equals(original.length));

      final dequantized = quantizer.dequantize(quantized);
      for (var i = 0; i < original.length; i++) {
        expect(dequantized[i], closeTo(original[i], 0.03));
      }
    });

    test('Fit quantizer on batch', () {
      final samples = [
        Float32List.fromList([-0.8, 0.2, 0.5]),
        Float32List.fromList([0.1, -0.9, 0.95]),
      ];
      final q = ScalarQuantizer.fit(samples);
      expect(q.minVal, lessThanOrEqualTo(-0.9));
      expect(q.maxVal, greaterThanOrEqualTo(0.95));
    });
  });

  group('FlatIndex', () {
    test('Basic CRUD and Exact Search', () {
      final index = FlatIndex(metric: DistanceMetric.cosine);
      index.insert(
        VectorRecord(
          id: 'doc1',
          vector: [1.0, 0.0, 0.0],
          metadata: {'title': 'Doc One'},
        ),
      );
      index.insert(
        VectorRecord(
          id: 'doc2',
          vector: [0.0, 1.0, 0.0],
          metadata: {'title': 'Doc Two'},
        ),
      );
      index.insert(
        VectorRecord(
          id: 'doc3',
          vector: [0.7071, 0.7071, 0.0],
          metadata: {'title': 'Doc Three'},
        ),
      );

      expect(index.count, equals(3));
      expect(index.dimension, equals(3));

      // Query vector [1, 0, 0] should match doc1 highest
      final results = index.search([1.0, 0.0, 0.0], topK: 2);
      expect(results.length, equals(2));
      expect(results[0].id, equals('doc1'));
      expect(results[0].score, closeTo(1.0, 1e-4));
      expect(results[1].id, equals('doc3'));

      // Remove doc1
      expect(index.remove('doc1'), isTrue);
      expect(index.count, equals(2));
      expect(index.get('doc1'), isNull);
    });

    test('Search with metadata filter', () {
      final index = FlatIndex();
      index.insert(
        VectorRecord(
          id: 'a',
          vector: [1.0, 0.0],
          metadata: {'tag': 'AI'},
        ),
      );
      index.insert(
        VectorRecord(
          id: 'b',
          vector: [0.99, 0.01],
          metadata: {'tag': 'Crypto'},
        ),
      );

      final results = index.search(
        [1.0, 0.0],
        filter: Filter.eq('tag', 'AI'),
      );
      expect(results.length, equals(1));
      expect(results[0].id, equals('a'));
    });

    test('FlatIndex JSON serialization', () {
      final index = FlatIndex(metric: DistanceMetric.euclidean);
      index.insert(
        VectorRecord(
          id: 'x',
          vector: [2.0, 3.0],
          metadata: {'k': 'v'},
        ),
      );

      final json = index.toJson();
      final restored = FlatIndex.fromJson(json);

      expect(restored.count, equals(1));
      expect(restored.dimension, equals(2));
      expect(restored.metric, equals(DistanceMetric.euclidean));
      expect(restored.get('x')?.metadata['k'], equals('v'));
    });
  });

  group('HnswIndex', () {
    test('HnswIndex construction and high recall vs FlatIndex', () {
      final hnsw = HnswIndex(
        metric: DistanceMetric.cosine,
        m: 16,
        efConstruction: 64,
        efSearch: 32,
        randomSeed: 42,
      );

      final flat = FlatIndex(metric: DistanceMetric.cosine);

      // Generate 100 random 32-dimensional normalized vectors
      final rng = math.Random(123);
      final records = <VectorRecord>[];
      for (var i = 0; i < 100; i++) {
        final raw = Float32List(32);
        for (var j = 0; j < 32; j++) {
          raw[j] = rng.nextDouble() * 2.0 - 1.0;
        }
        final normalized = VectorMath.normalize(raw);
        final rec = VectorRecord(
          id: 'vec_$i',
          vector: normalized,
          metadata: {'index': i, 'group': i % 3 == 0 ? 'A' : 'B'},
        );
        records.add(rec);
      }

      hnsw.insertAll(records);
      flat.insertAll(records);

      expect(hnsw.count, equals(100));
      expect(hnsw.dimension, equals(32));

      // Test 20 query vectors
      var totalTop1Matches = 0;
      for (var q = 0; q < 20; q++) {
        final queryRaw = Float32List(32);
        for (var j = 0; j < 32; j++) {
          queryRaw[j] = rng.nextDouble() * 2.0 - 1.0;
        }
        final query = VectorMath.normalize(queryRaw);

        final groundTruth = flat.search(query, topK: 5);
        final hnswResults = hnsw.search(query, topK: 5);

        expect(hnswResults.isNotEmpty, isTrue);
        if (hnswResults.first.id == groundTruth.first.id) {
          totalTop1Matches++;
        }
      }

      // High recall: At least 95% of top-1 results match exact ground truth
      expect(totalTop1Matches, greaterThanOrEqualTo(19));
    });

    test('HnswIndex deletion and re-election', () {
      final hnsw = HnswIndex(randomSeed: 99);
      hnsw.insert(VectorRecord(id: '1', vector: [1.0, 0.0]));
      hnsw.insert(VectorRecord(id: '2', vector: [0.0, 1.0]));
      hnsw.insert(VectorRecord(id: '3', vector: [0.5, 0.5]));

      expect(hnsw.count, equals(3));
      final epBefore = hnsw.entryPointId;
      expect(epBefore, isNotNull);

      expect(hnsw.remove('1'), isTrue);
      expect(hnsw.count, equals(2));
      expect(hnsw.get('1'), isNull);

      final search = hnsw.search([1.0, 0.0], topK: 1);
      expect(search.first.id, equals('3'));
    });

    test('HnswIndex JSON serialization', () {
      final hnsw = HnswIndex(metric: DistanceMetric.cosine, randomSeed: 1);
      hnsw.insert(
        VectorRecord(
          id: 'item1',
          vector: [0.6, 0.8],
          metadata: {'category': 'books'},
        ),
      );
      hnsw.insert(
        VectorRecord(
          id: 'item2',
          vector: [0.8, 0.6],
          metadata: {'category': 'tech'},
        ),
      );

      final json = hnsw.toJson();
      final restored = HnswIndex.fromJson(json);

      expect(restored.count, equals(2));
      expect(restored.dimension, equals(2));
      final res = restored.search([0.6, 0.8], topK: 1);
      expect(res.first.id, equals('item1'));
    });
  });

  group('VectorStore & Persistence', () {
    test('VectorStore high-level CRUD and search', () {
      final store = VectorStore(type: IndexType.hnsw, randomSeed: 7);
      store.add(
        id: 'r1',
        vector: [1.0, 0.0, 0.0],
        metadata: {'title': 'Dart Vector Index'},
      );
      store.add(
        id: 'r2',
        vector: [0.0, 1.0, 0.0],
        metadata: {'title': 'Flutter Voice Orb'},
      );

      expect(store.count, equals(2));
      expect(store.dimension, equals(3));
      expect(store.contains('r1'), isTrue);

      final results = store.similaritySearch([1.0, 0.1, 0.0], limit: 1);
      expect(results.first.id, equals('r1'));
      expect(results.first.metadata['title'], equals('Dart Vector Index'));
    });

    test('Atomic file persistence roundtrip', () async {
      final tempDir =
          await Directory.systemTemp.createTemp('vector_store_test');
      final file = File('${tempDir.path}/test_store.json');

      final store =
          VectorStore(type: IndexType.flat, metric: DistanceMetric.cosine);
      store.add(
        id: 'p1',
        vector: [0.1, 0.2, 0.3],
        metadata: {'author': 'Govind Tank'},
      );
      store.add(
        id: 'p2',
        vector: [0.4, 0.5, 0.6],
        metadata: {'pkg': 'dart_vector_index'},
      );

      await store.saveToFile(file);
      expect(await file.exists(), isTrue);

      final loaded = await VectorStore.loadFromFile(file);
      expect(loaded.count, equals(2));
      expect(loaded.dimension, equals(3));
      expect(loaded.get('p1')?.metadata['author'], equals('Govind Tank'));

      await tempDir.delete(recursive: true);
    });
  });
}
