import 'package:dart_vector_index/dart_vector_index.dart';

void main() {
  print('=== dart_vector_index: On-Device RAG & Vector Search Demo ===\n');

  // 1. Initialize a fast HNSW vector store with Cosine Distance
  final store = VectorStore(
    type: IndexType.hnsw,
    metric: DistanceMetric.cosine,
    m: 16,
    efConstruction: 64,
    efSearch: 32,
    randomSeed: 42,
  );

  // 2. Index sample document embeddings (e.g. from Gemma, MiniLM, or OpenAI)
  print('Indexing knowledge base articles...');
  store.addBatch([
    VectorRecord(
      id: 'doc_1',
      vector: [0.95, 0.20, 0.05, 0.10],
      metadata: {
        'title': 'Introduction to Flutter On-Device AI',
        'category': 'ai',
        'author': 'Govind Tank',
        'views': 4500,
      },
      tag: 'flutter',
    ),
    VectorRecord(
      id: 'doc_2',
      vector: [0.88, 0.35, 0.12, 0.08],
      metadata: {
        'title': 'Building Realtime Voice Agents with Neural Orbs',
        'category': 'ai',
        'author': 'Govind Tank',
        'views': 3200,
      },
      tag: 'flutter',
    ),
    VectorRecord(
      id: 'doc_3',
      vector: [0.10, 0.85, 0.90, 0.25],
      metadata: {
        'title': 'Darvas Box Trading Strategy & Technical Analysis',
        'category': 'finance',
        'author': 'Govind Tank',
        'views': 8900,
      },
      tag: 'trading',
    ),
    VectorRecord(
      id: 'doc_4',
      vector: [0.15, 0.78, 0.88, 0.30],
      metadata: {
        'title': 'Indian Stock Market Paper Trading Bot Guide',
        'category': 'finance',
        'author': 'Govind Tank',
        'views': 6100,
      },
      tag: 'trading',
    ),
  ]);

  print('Total indexed records: ${store.count} (dim: ${store.dimension})\n');

  // 3. Perform a semantic similarity query for AI & Flutter
  print('--- Semantic Search Query: "Flutter AI voice assistant" ---');
  final aiQueryVector = [0.92, 0.25, 0.08, 0.12];

  final results = store.similaritySearch(
    aiQueryVector,
    limit: 2,
  );

  for (final hit in results) {
    print(
      '-> Match [${hit.id}] | Score: ${(hit.score * 100).toStringAsFixed(1)}% | Dist: ${hit.distance.toStringAsFixed(4)}',
    );
    print('   Title: ${hit.metadata['title']}');
    print(
        '   Category: ${hit.metadata['category']} | Views: ${hit.metadata['views']}');
  }

  // 4. Perform filtered search: Only finance docs with views >= 7000
  print('\n--- Filtered Search: category == "finance" AND views >= 7000 ---');
  final financeQueryVector = [0.12, 0.82, 0.89, 0.28];

  final filteredResults = store.search(
    financeQueryVector,
    limit: 5,
    filter: Filter.and([
      Filter.eq('category', 'finance'),
      Filter.gte('views', 7000),
    ]),
  );

  for (final hit in filteredResults) {
    print(
      '-> Filtered [${hit.id}] | Score: ${(hit.score * 100).toStringAsFixed(1)}%',
    );
    print(
        '   Title: ${hit.metadata['title']} | Views: ${hit.metadata['views']}');
  }

  // 5. Demonstrate 8-bit Scalar Quantization (SQ8)
  print('\n--- 8-Bit Scalar Quantization (SQ8) ---');
  final quantizer = ScalarQuantizer(minVal: -1.0, maxVal: 1.0);
  final sampleFloatVec = VectorMath.toFloat32List([0.95, -0.42, 0.0, 0.78]);
  final compressed = quantizer.quantize(sampleFloatVec);
  final decompressed = quantizer.dequantize(compressed);

  print('Original float32: $sampleFloatVec');
  print('Quantized uint8:  $compressed (4x RAM reduction)');
  print('Decompressed:     $decompressed');

  // 6. JSON Export & Backup
  final jsonDump = store.saveToJsonString(pretty: true);
  print('\n--- Serialized VectorStore JSON (snippet) ---');
  print('${jsonDump.split('\n').take(12).join('\n')}\n...');

  print('\n=== Demo Completed Successfully ===');
}
