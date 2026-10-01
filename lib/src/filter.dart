/// Predicate filter for evaluating metadata attached to vector records.
abstract class MetadataFilter {
  /// Default constructor for [MetadataFilter].
  const MetadataFilter();

  /// Evaluates whether the given [metadata] and optional [tag] satisfies this filter.
  bool evaluate(Map<String, dynamic> metadata, {String? tag});

  /// Serializes this filter into a JSON map.
  Map<String, dynamic> toJson();

  /// Deserializes a [MetadataFilter] from a JSON map.
  static MetadataFilter fromJson(Map<String, dynamic> json) {
    final op = json['op'] as String?;
    switch (op) {
      case 'eq':
        return Filter.eq(json['key'] as String, json['val']);
      case 'neq':
        return Filter.neq(json['key'] as String, json['val']);
      case 'gt':
        return Filter.gt(json['key'] as String, json['val'] as num);
      case 'gte':
        return Filter.gte(json['key'] as String, json['val'] as num);
      case 'lt':
        return Filter.lt(json['key'] as String, json['val'] as num);
      case 'lte':
        return Filter.lte(json['key'] as String, json['val'] as num);
      case 'in':
        return Filter.inList(
          json['key'] as String,
          json['val'] as List<dynamic>,
        );
      case 'contains':
        return Filter.contains(json['key'] as String, json['val']);
      case 'tag':
        return Filter.tag(json['val'] as String);
      case 'and':
        final filtersJson = json['filters'] as List<dynamic>;
        return Filter.and(
          filtersJson
              .map((e) => MetadataFilter.fromJson(e as Map<String, dynamic>))
              .toList(),
        );
      case 'or':
        final filtersJson = json['filters'] as List<dynamic>;
        return Filter.or(
          filtersJson
              .map((e) => MetadataFilter.fromJson(e as Map<String, dynamic>))
              .toList(),
        );
      case 'not':
        return Filter.not(
          MetadataFilter.fromJson(json['filter'] as Map<String, dynamic>),
        );
      default:
        throw FormatException('Unknown filter op: $op');
    }
  }
}

/// Factory class for building declarative metadata filters.
class Filter {
  Filter._();

  /// Matches records where `metadata[key] == value`.
  static MetadataFilter eq(String key, dynamic value) =>
      _FieldComparisonFilter(key, _CompOp.eq, value);

  /// Matches records where `metadata[key] != value`.
  static MetadataFilter neq(String key, dynamic value) =>
      _FieldComparisonFilter(key, _CompOp.neq, value);

  /// Matches numeric records where `metadata[key] > value`.
  static MetadataFilter gt(String key, num value) =>
      _FieldComparisonFilter(key, _CompOp.gt, value);

  /// Matches numeric records where `metadata[key] >= value`.
  static MetadataFilter gte(String key, num value) =>
      _FieldComparisonFilter(key, _CompOp.gte, value);

  /// Matches numeric records where `metadata[key] < value`.
  static MetadataFilter lt(String key, num value) =>
      _FieldComparisonFilter(key, _CompOp.lt, value);

  /// Matches numeric records where `metadata[key] <= value`.
  static MetadataFilter lte(String key, num value) =>
      _FieldComparisonFilter(key, _CompOp.lte, value);

  /// Matches records where `metadata[key]` is present in [values].
  static MetadataFilter inList(String key, List<dynamic> values) =>
      _InListFilter(key, values);

  /// Matches records where `metadata[key]` contains [item] (for strings or lists).
  static MetadataFilter contains(String key, dynamic item) =>
      _ContainsFilter(key, item);

  /// Matches records whose [VectorRecord.tag] equals [tag].
  static MetadataFilter tag(String tag) => _TagFilter(tag);

  /// Matches records where ALL [filters] evaluate to true.
  static MetadataFilter and(List<MetadataFilter> filters) =>
      _AndFilter(filters);

  /// Matches records where ANY [filters] evaluate to true.
  static MetadataFilter or(List<MetadataFilter> filters) => _OrFilter(filters);

  /// Inverts the result of [filter].
  static MetadataFilter not(MetadataFilter filter) => _NotFilter(filter);

  /// Custom arbitrary predicate function (not serializable to JSON).
  static MetadataFilter custom(
    bool Function(Map<String, dynamic> metadata, String? tag) predicate,
  ) =>
      _CustomFilter(predicate);
}

enum _CompOp { eq, neq, gt, gte, lt, lte }

class _FieldComparisonFilter implements MetadataFilter {
  final String key;
  final _CompOp op;
  final dynamic val;

  const _FieldComparisonFilter(this.key, this.op, this.val);

  @override
  bool evaluate(Map<String, dynamic> metadata, {String? tag}) {
    final actual = metadata[key];
    final v = val;
    if (actual == null) {
      return op == _CompOp.neq;
    }

    switch (op) {
      case _CompOp.eq:
        return actual == v;
      case _CompOp.neq:
        return actual != v;
      case _CompOp.gt:
        if (actual is num && v is num) {
          return actual > v;
        }
        return false;
      case _CompOp.gte:
        if (actual is num && v is num) {
          return actual >= v;
        }
        return false;
      case _CompOp.lt:
        if (actual is num && v is num) {
          return actual < v;
        }
        return false;
      case _CompOp.lte:
        if (actual is num && v is num) {
          return actual <= v;
        }
        return false;
    }
  }

  @override
  Map<String, dynamic> toJson() => {
        'op': op.name,
        'key': key,
        'val': val,
      };
}

class _InListFilter implements MetadataFilter {
  final String key;
  final List<dynamic> values;

  const _InListFilter(this.key, this.values);

  @override
  bool evaluate(Map<String, dynamic> metadata, {String? tag}) {
    final actual = metadata[key];
    return values.contains(actual);
  }

  @override
  Map<String, dynamic> toJson() => {
        'op': 'in',
        'key': key,
        'val': values,
      };
}

class _ContainsFilter implements MetadataFilter {
  final String key;
  final dynamic item;

  const _ContainsFilter(this.key, this.item);

  @override
  bool evaluate(Map<String, dynamic> metadata, {String? tag}) {
    final actual = metadata[key];
    final it = item;
    if (actual is String && it is String) {
      return actual.contains(it);
    }
    if (actual is Iterable) {
      return actual.contains(it);
    }
    return false;
  }

  @override
  Map<String, dynamic> toJson() => {
        'op': 'contains',
        'key': key,
        'val': item,
      };
}

class _TagFilter implements MetadataFilter {
  final String expectedTag;

  const _TagFilter(this.expectedTag);

  @override
  bool evaluate(Map<String, dynamic> metadata, {String? tag}) {
    return tag == expectedTag;
  }

  @override
  Map<String, dynamic> toJson() => {
        'op': 'tag',
        'val': expectedTag,
      };
}

class _AndFilter implements MetadataFilter {
  final List<MetadataFilter> filters;

  const _AndFilter(this.filters);

  @override
  bool evaluate(Map<String, dynamic> metadata, {String? tag}) {
    for (final f in filters) {
      if (!f.evaluate(metadata, tag: tag)) {
        return false;
      }
    }
    return true;
  }

  @override
  Map<String, dynamic> toJson() => {
        'op': 'and',
        'filters': filters.map((f) => f.toJson()).toList(),
      };
}

class _OrFilter implements MetadataFilter {
  final List<MetadataFilter> filters;

  const _OrFilter(this.filters);

  @override
  bool evaluate(Map<String, dynamic> metadata, {String? tag}) {
    if (filters.isEmpty) {
      return true;
    }
    for (final f in filters) {
      if (f.evaluate(metadata, tag: tag)) {
        return true;
      }
    }
    return false;
  }

  @override
  Map<String, dynamic> toJson() => {
        'op': 'or',
        'filters': filters.map((f) => f.toJson()).toList(),
      };
}

class _NotFilter implements MetadataFilter {
  final MetadataFilter filter;

  const _NotFilter(this.filter);

  @override
  bool evaluate(Map<String, dynamic> metadata, {String? tag}) {
    return !filter.evaluate(metadata, tag: tag);
  }

  @override
  Map<String, dynamic> toJson() => {
        'op': 'not',
        'filter': filter.toJson(),
      };
}

class _CustomFilter implements MetadataFilter {
  final bool Function(Map<String, dynamic> metadata, String? tag) predicate;

  const _CustomFilter(this.predicate);

  @override
  bool evaluate(Map<String, dynamic> metadata, {String? tag}) =>
      predicate(metadata, tag);

  @override
  Map<String, dynamic> toJson() => {
        'op': 'custom',
        'description': 'runtime_predicate',
      };
}
