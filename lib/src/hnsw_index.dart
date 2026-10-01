import 'dart:math' as math;
import 'dart:typed_data';
import 'filter.dart';
import 'index.dart';
import 'metric.dart';
import 'record.dart';

/// Hierarchical Navigable Small World (HNSW) graph index for fast $O(\log N)$
/// approximate nearest neighbor search over high-dimensional vector embeddings.
class HnswIndex implements VectorIndex {
  /// Maximum number of bi-directional links per node on higher layers.
  final int m;

  /// Maximum number of bi-directional links on the ground layer (layer 0).
  final int maxM0;

  /// Size of the dynamic candidate list during graph construction.
  final int efConstruction;

  /// Default size of the dynamic candidate list during search queries.
  final int efSearch;

  /// Level generation multiplier: `1.0 / ln(m)`.
  final double mL;

  final DistanceMetric _metric;
  final math.Random _random;

  final Map<String, _HnswNode> _nodes = <String, _HnswNode>{};
  String? _entryPointId;
  int _maxLayer = -1;
  int _dimension = -1;

  /// Creates an [HnswIndex].
  ///
  /// - [metric]: Distance metric (defaults to [DistanceMetric.cosine]).
  /// - [m]: Number of bidirectional connections per element (defaults to 16).
  /// - [efConstruction]: Trade-off between build speed and graph quality (defaults to 64).
  /// - [efSearch]: Trade-off between query speed and search recall (defaults to 32).
  /// - [randomSeed]: Optional seed for deterministic graph generation in tests.
  HnswIndex({
    DistanceMetric metric = DistanceMetric.cosine,
    this.m = 16,
    this.efConstruction = 64,
    this.efSearch = 32,
    int? randomSeed,
  })  : assert(m >= 2, 'm must be at least 2'),
        assert(
          efConstruction >= m,
          'efConstruction must be greater than or equal to m',
        ),
        assert(efSearch >= 1, 'efSearch must be at least 1'),
        maxM0 = 2 * m,
        mL = 1.0 / math.log(m.toDouble()),
        _metric = metric,
        _random = randomSeed != null ? math.Random(randomSeed) : math.Random();

  @override
  int get count => _nodes.length;

  @override
  int get dimension => _dimension;

  @override
  DistanceMetric get metric => _metric;

  /// Current highest layer level in the graph.
  int get maxLayer => _maxLayer;

  /// Identifier of the top-layer entry point node, if any.
  String? get entryPointId => _entryPointId;

  @override
  void insert(VectorRecord record) {
    if (_dimension == -1) {
      _dimension = record.dimension;
    } else if (record.dimension != _dimension) {
      throw ArgumentError(
        'Vector dimension mismatch: expected $_dimension, got ${record.dimension}.',
      );
    }

    // If node exists, remove and re-insert to update
    if (_nodes.containsKey(record.id)) {
      remove(record.id);
    }

    final targetLevel = _sampleLevel();
    final newNode = _HnswNode(
      id: record.id,
      record: record,
      level: targetLevel,
    );
    _nodes[record.id] = newNode;

    // First element in graph
    if (_entryPointId == null) {
      _entryPointId = record.id;
      _maxLayer = targetLevel;
      return;
    }

    var currObj = _entryPointId!;
    var currDist = VectorMath.distance(
      record.vector,
      _nodes[currObj]!.record.vector,
      _metric,
    );

    // 1. Traverse top layers down to (targetLevel + 1) greedily
    for (var level = _maxLayer; level > targetLevel; level--) {
      var changed = true;
      while (changed) {
        changed = false;
        final neighbors = _nodes[currObj]!.neighborsAt(level);
        for (final neighborId in neighbors) {
          final neighborNode = _nodes[neighborId];
          if (neighborNode == null) {
            continue;
          }
          final d = VectorMath.distance(
            record.vector,
            neighborNode.record.vector,
            _metric,
          );
          if (d < currDist) {
            currDist = d;
            currObj = neighborId;
            changed = true;
          }
        }
      }
    }

    // 2. From min(targetLevel, maxLayer) down to layer 0, search and link
    var enterPoints = <_DistItem>[_DistItem(currObj, currDist)];
    final topSearchLevel = math.min(targetLevel, _maxLayer);

    for (var level = topSearchLevel; level >= 0; level--) {
      final candidates = _searchLayer(
        record.vector,
        enterPoints,
        efConstruction,
        level,
      );

      final maxM = level == 0 ? maxM0 : m;
      final selectedNeighbors = _selectNeighbors(candidates, maxM);

      for (final neighbor in selectedNeighbors) {
        newNode.addNeighbor(level, neighbor.id);
        final neighborNode = _nodes[neighbor.id];
        if (neighborNode != null) {
          neighborNode.addNeighbor(level, record.id);
          // Prune neighbor's links if exceeding max degree
          if (neighborNode.neighborCountAt(level) > maxM) {
            _shrinkNeighbors(neighborNode, level, maxM);
          }
        }
      }

      enterPoints = candidates;
    }

    // 3. Update entry point if new node has higher level
    if (targetLevel > _maxLayer) {
      _maxLayer = targetLevel;
      _entryPointId = record.id;
    }
  }

  @override
  void insertAll(Iterable<VectorRecord> records) {
    for (final record in records) {
      insert(record);
    }
  }

  @override
  bool remove(String id) {
    final targetNode = _nodes.remove(id);
    if (targetNode == null) {
      return false;
    }

    // Unlink from all neighbors across all layers
    for (var level = 0; level <= targetNode.level; level++) {
      final neighbors = targetNode.neighborsAt(level);
      for (final neighborId in neighbors) {
        final neighborNode = _nodes[neighborId];
        if (neighborNode != null) {
          neighborNode.removeNeighbor(level, id);
        }
      }
    }

    // If removed node was entry point, elect a new one
    if (_entryPointId == id) {
      if (_nodes.isEmpty) {
        _entryPointId = null;
        _maxLayer = -1;
        _dimension = -1;
      } else {
        // Find node with highest level
        var highestLevel = -1;
        String? newEp;
        for (final node in _nodes.values) {
          if (node.level > highestLevel) {
            highestLevel = node.level;
            newEp = node.id;
          }
        }
        _entryPointId = newEp;
        _maxLayer = highestLevel;
      }
    }

    return true;
  }

  @override
  VectorRecord? get(String id) => _nodes[id]?.record;

  @override
  bool contains(String id) => _nodes.containsKey(id);

  @override
  List<SearchResult> search(
    List<num> query, {
    int topK = 10,
    MetadataFilter? filter,
    double? minScore,
    double? maxDistance,
    bool includeVectors = false,
    int? customEfSearch,
  }) {
    if (_nodes.isEmpty || topK <= 0 || _entryPointId == null) {
      return const <SearchResult>[];
    }

    final queryVec =
        query is Float32List ? query : VectorMath.toFloat32List(query);

    if (_dimension != -1 && queryVec.length != _dimension) {
      throw ArgumentError(
        'Query dimension (${queryVec.length}) does not match index dimension ($_dimension).',
      );
    }

    var currObj = _entryPointId!;
    var currDist = VectorMath.distance(
      queryVec,
      _nodes[currObj]!.record.vector,
      _metric,
    );

    // 1. Greedy routing down through upper layers
    for (var level = _maxLayer; level > 0; level--) {
      var changed = true;
      while (changed) {
        changed = false;
        final neighbors = _nodes[currObj]!.neighborsAt(level);
        for (final neighborId in neighbors) {
          final neighborNode = _nodes[neighborId];
          if (neighborNode == null) {
            continue;
          }
          final d = VectorMath.distance(
            queryVec,
            neighborNode.record.vector,
            _metric,
          );
          if (d < currDist) {
            currDist = d;
            currObj = neighborId;
            changed = true;
          }
        }
      }
    }

    // 2. Beam search on ground layer (layer 0)
    final ef = math.max(customEfSearch ?? efSearch, topK);
    final candidates = _searchLayer(
      queryVec,
      [_DistItem(currObj, currDist)],
      ef,
      0,
    );

    // 3. Filter, score, and sort
    final results = <SearchResult>[];

    for (final item in candidates) {
      final node = _nodes[item.id];
      if (node == null) {
        continue;
      }

      if (filter != null &&
          !filter.evaluate(node.record.metadata, tag: node.record.tag)) {
        continue;
      }

      if (maxDistance != null && item.distance > maxDistance) {
        continue;
      }

      final score = VectorMath.distanceToScore(item.distance, _metric);

      if (minScore != null && score < minScore) {
        continue;
      }

      results.add(
        SearchResult(
          id: node.id,
          score: score,
          distance: item.distance,
          metadata: node.record.metadata,
          vector: includeVectors ? node.record.vector : null,
          tag: node.record.tag,
        ),
      );
    }

    results.sort();

    if (results.length > topK) {
      return results.sublist(0, topK);
    }
    return results;
  }

  @override
  void clear() {
    _nodes.clear();
    _entryPointId = null;
    _maxLayer = -1;
    _dimension = -1;
  }

  int _sampleLevel() {
    final r = _random.nextDouble();
    if (r == 0.0) {
      return 0;
    }
    final level = (-math.log(r) * mL).floor();
    return level;
  }

  List<_DistItem> _searchLayer(
    Float32List queryVec,
    List<_DistItem> enterPoints,
    int ef,
    int level,
  ) {
    final visited = <String>{};
    final candidates = <_DistItem>[]; // Min-priority for unvisited exploration
    final w = <_DistItem>[]; // Dynamic result set (tracked sorted by distance)

    for (final ep in enterPoints) {
      visited.add(ep.id);
      candidates.add(ep);
      w.add(ep);
    }

    while (candidates.isNotEmpty) {
      // Extract candidate closest to query
      candidates.sort((a, b) => a.distance.compareTo(b.distance));
      final current = candidates.removeAt(0);

      // Get furthest element in current best set w
      w.sort((a, b) => a.distance.compareTo(b.distance));
      final furthestInW = w.last;

      if (current.distance > furthestInW.distance && w.length >= ef) {
        break;
      }

      final currNode = _nodes[current.id];
      if (currNode == null) {
        continue;
      }

      final neighbors = currNode.neighborsAt(level);
      for (final neighborId in neighbors) {
        if (visited.contains(neighborId)) {
          continue;
        }
        visited.add(neighborId);

        final neighborNode = _nodes[neighborId];
        if (neighborNode == null) {
          continue;
        }

        final d = VectorMath.distance(
          queryVec,
          neighborNode.record.vector,
          _metric,
        );

        if (d < furthestInW.distance || w.length < ef) {
          final item = _DistItem(neighborId, d);
          candidates.add(item);
          w.add(item);
          w.sort((a, b) => a.distance.compareTo(b.distance));
          if (w.length > ef) {
            w.removeLast();
          }
        }
      }
    }

    w.sort((a, b) => a.distance.compareTo(b.distance));
    return w;
  }

  List<_DistItem> _selectNeighbors(
      List<_DistItem> candidates, int maxNeighbors) {
    if (candidates.length <= maxNeighbors) {
      return candidates;
    }
    candidates.sort((a, b) => a.distance.compareTo(b.distance));
    return candidates.sublist(0, maxNeighbors);
  }

  void _shrinkNeighbors(_HnswNode node, int level, int maxM) {
    final neighborIds = node.neighborsAt(level);
    if (neighborIds.length <= maxM) {
      return;
    }

    final list = <_DistItem>[];
    for (final nId in neighborIds) {
      final nNode = _nodes[nId];
      if (nNode != null) {
        final d = VectorMath.distance(
          node.record.vector,
          nNode.record.vector,
          _metric,
        );
        list.add(_DistItem(nId, d));
      }
    }

    list.sort((a, b) => a.distance.compareTo(b.distance));
    final kept = list.take(maxM).map((e) => e.id).toSet();
    node.setNeighborsAt(level, kept);
  }

  @override
  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'type': 'hnsw',
      'metric': _metric.name,
      'm': m,
      'efConstruction': efConstruction,
      'efSearch': efSearch,
      'dimension': _dimension,
      'entryPointId': _entryPointId,
      'maxLayer': _maxLayer,
      'records': _nodes.values.map((n) => n.record.toJson()).toList(
            growable: false,
          ),
      'graph': _nodes.values
          .map((n) => {
                'id': n.id,
                'level': n.level,
                'neighbors': n.serializeNeighbors(),
              })
          .toList(growable: false),
    };
  }

  /// Deserializes an [HnswIndex] from a JSON-compatible map.
  factory HnswIndex.fromJson(Map<String, dynamic> json) {
    final metricName = json['metric'] as String? ?? 'cosine';
    final metric = DistanceMetric.values.firstWhere(
      (m) => m.name == metricName,
      orElse: () => DistanceMetric.cosine,
    );

    final index = HnswIndex(
      metric: metric,
      m: json['m'] as int? ?? 16,
      efConstruction: json['efConstruction'] as int? ?? 64,
      efSearch: json['efSearch'] as int? ?? 32,
    );

    final recordsList = json['records'] as List<dynamic>?;
    if (recordsList != null) {
      for (final item in recordsList) {
        final record = VectorRecord.fromJson(item as Map<String, dynamic>);
        index._nodes[record.id] = _HnswNode(
          id: record.id,
          record: record,
          level: 0,
        );
        if (index._dimension == -1) {
          index._dimension = record.dimension;
        }
      }
    }

    final graphList = json['graph'] as List<dynamic>?;
    if (graphList != null) {
      for (final item in graphList) {
        final nodeMap = item as Map<String, dynamic>;
        final id = nodeMap['id'] as String;
        final level = nodeMap['level'] as int;
        final rawNeighbors = nodeMap['neighbors'] as List<dynamic>;

        final node = index._nodes[id];
        if (node != null) {
          final reconstructed = _HnswNode(
            id: id,
            record: node.record,
            level: level,
          );
          for (var l = 0; l < rawNeighbors.length; l++) {
            final nList = (rawNeighbors[l] as List<dynamic>).cast<String>();
            reconstructed.setNeighborsAt(l, nList.toSet());
          }
          index._nodes[id] = reconstructed;
        }
      }
    }

    index._entryPointId = json['entryPointId'] as String?;
    index._maxLayer = json['maxLayer'] as int? ?? -1;

    return index;
  }
}

class _HnswNode {
  final String id;
  final VectorRecord record;
  final int level;
  final List<Set<String>> _neighbors;

  _HnswNode({
    required this.id,
    required this.record,
    required this.level,
  }) : _neighbors = List.generate(level + 1, (_) => <String>{});

  Set<String> neighborsAt(int l) {
    if (l < 0 || l >= _neighbors.length) {
      return const <String>{};
    }
    return _neighbors[l];
  }

  int neighborCountAt(int l) {
    if (l < 0 || l >= _neighbors.length) {
      return 0;
    }
    return _neighbors[l].length;
  }

  void addNeighbor(int l, String neighborId) {
    while (_neighbors.length <= l) {
      _neighbors.add(<String>{});
    }
    _neighbors[l].add(neighborId);
  }

  void removeNeighbor(int l, String neighborId) {
    if (l >= 0 && l < _neighbors.length) {
      _neighbors[l].remove(neighborId);
    }
  }

  void setNeighborsAt(int l, Set<String> neighborIds) {
    while (_neighbors.length <= l) {
      _neighbors.add(<String>{});
    }
    _neighbors[l] = neighborIds;
  }

  List<List<String>> serializeNeighbors() {
    return _neighbors.map((s) => s.toList(growable: false)).toList();
  }
}

class _DistItem {
  final String id;
  final double distance;

  const _DistItem(this.id, this.distance);
}
