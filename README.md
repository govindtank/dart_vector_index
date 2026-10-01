# dart_vector_index

[![Pub Version](https://img.shields.io/pub/v/dart_vector_index.svg?style=flat-square&color=blue)](https://pub.dev/packages/dart_vector_index)
[![Pub Points](https://img.shields.io/pub/points/dart_vector_index?style=flat-square&color=2E8B57&label=pub%20points)](https://pub.dev/packages/dart_vector_index/score)
[![Pub Likes](https://img.shields.io/pub/likes/dart_vector_index?style=flat-square)](https://pub.dev/packages/dart_vector_index)
[![CI](https://github.com/govindtank/dart_vector_index/actions/workflows/ci.yml/badge.svg)](https://github.com/govindtank/dart_vector_index/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square)](LICENSE)

Fast, zero-dependency, pure-Dart vector search and **HNSW** (Hierarchical Navigable Small World) index for on-device RAG (Retrieval Augmented Generation), semantic search, and AI embedding similarity on Flutter and Dart.

<p align="center">
  <img src="https://raw.githubusercontent.com/govindtank/dart_vector_index/main/screenshot.svg" width="750" alt="dart_vector_index architecture overview"/>
</p>

---

## ⚡ Why dart_vector_index?

Building on-device AI in Flutter (using local models like **Gemma**, **Whisper**, **LLaMA**, or cloud embeddings from OpenAI/Cohere) requires searching thousands of high-dimensional vectors in sub-millisecond time.

Most existing solutions force you into bulky C++ native binaries, heavy SQLite extensions, or remote cloud databases. 

`dart_vector_index` is **100% pure Dart**:
- **Zero native dependencies**: Runs seamlessly across **Flutter (Android, iOS, Web, macOS, Windows, Linux)**, Dart CLI, Dart Frog, and server backends.
- **$O(\log N)$ HNSW Graph Index**: Fast approximate nearest-neighbor search with >99% recall.
- **Exact Flat Index**: Guaranteed 100% ground-truth brute force search for smaller datasets.
- **8-Bit Scalar Quantization (SQ8)**: Cuts mobile RAM usage by 75% (4x compression).
- **Rich Predicate Filtering**: Filter metadata by equality, numeric ranges, lists, tags, and composite `and`/`or`/`not` boolean logic.
- **Atomic File Persistence**: Save and restore your embedding knowledge base to disk with a single line of code.

---

## 🚀 Features

| Feature | Description |
| :--- | :--- |
| **HNSW Graph Index** | Multi-layer navigable small-world graph for $O(\log N)$ search across 100k+ embeddings. |
| **Flat Brute-Force Index** | Exact exhaustive search for guaranteed 100% recall. |
| **Distance Metrics** | `Cosine`, `Euclidean` ($L_2$), `DotProduct`, `Manhattan` ($L_1$), `Chebyshev` ($L_\infty$). |
| **SQ8 Quantization** | 4x RAM reduction by quantizing Float32 embeddings into Uint8 buffers. |
| **Metadata Filtering** | Declarative SQL-like filters (`Filter.eq`, `Filter.gte`, `Filter.inList`, `Filter.and`). |
| **Atomic File Sync** | Atomic disk serialization to prevent file corruption on mobile app termination. |

---

## 📦 Installation

Add `dart_vector_index` to your `pubspec.yaml`:

```yaml
dependencies:
  dart_vector_index: ^1.0.0
```

Or via terminal:

```bash
dart pub add dart_vector_index
```

---

## 🏁 Quick Start

```dart
import 'package:dart_vector_index/dart_vector_index.dart';

void main() {
  // 1. Initialize an HNSW Vector Store
  final store = VectorStore(
    type: IndexType.hnsw,
    metric: DistanceMetric.cosine,
  );

  // 2. Add embeddings with metadata
  store.add(
    id: 'doc_1',
    vector: [0.95, 0.20, 0.05, 0.10],
    metadata: {'title': 'On-Device AI in Flutter', 'category': 'tech'},
  );

  store.add(
    id: 'doc_2',
    vector: [0.10, 0.85, 0.90, 0.25],
    metadata: {'title': 'Stock Market Technical Analysis', 'category': 'finance'},
  );

  // 3. Search for nearest neighbors
  final queryVector = [0.92, 0.25, 0.08, 0.12];
  final results = store.similaritySearch(queryVector, limit: 1);

  for (final hit in results) {
    print('Match: ${hit.id} (${(hit.score * 100).toStringAsFixed(1)}%)');
    print('Title: ${hit.metadata['title']}');
  }
}
```

---

## 🔍 Deep Dive & API Examples

### 1. Advanced Metadata Filtering
Prune search results before returning top-K candidates:

```dart
final results = store.search(
  queryVector,
  limit: 5,
  filter: Filter.and([
    Filter.eq('category', 'tech'),
    Filter.gte('rating', 4.5),
    Filter.inList('tags', ['flutter', 'ai']),
  ]),
);
```

Available filter operators:
- `Filter.eq(key, value)` / `Filter.neq(key, value)`
- `Filter.gt(key, num)` / `Filter.gte(key, num)` / `Filter.lt(key, num)` / `Filter.lte(key, num)`
- `Filter.inList(key, List values)`
- `Filter.contains(key, substringOrElement)`
- `Filter.tag(String tag)`
- `Filter.and(List<Filter> filters)` / `Filter.or(List<Filter> filters)` / `Filter.not(Filter filter)`

---

### 2. 8-Bit Scalar Quantization (SQ8)
Compress 32-bit floats into 8-bit unsigned bytes for 4x memory savings on mobile:

```dart
final quantizer = ScalarQuantizer(minVal: -1.0, maxVal: 1.0);

// Quantize Float32List -> Uint8List
final quantized = quantizer.quantize(floatVector);

// Dequantize back to Float32List when needed
final restored = quantizer.dequantize(quantized);
```

---

### 3. File Persistence & Backup
Save your entire on-device knowledge base to a local JSON file atomically:

```dart
import 'dart:io';

// Save store to disk
final file = File('/path/to/documents_vector_db.json');
await store.saveToFile(file);

// Load store from disk
final restoredStore = await VectorStore.loadFromFile(file);
print('Loaded ${restoredStore.count} vectors.');
```

---

## 👨💻 Author & Maintainer

Developed and maintained by **Govind Tank** ([@govindtank](https://github.com/govindtank)).

---

## 📄 License

Licensed under the **Apache License, Version 2.0**. See the [LICENSE](LICENSE) file for details.
