/// Web stub implementation of file storage.
Future<void> saveStringToFile(dynamic file, String content) async {
  throw UnsupportedError(
      'File persistence is not supported on the Web platform. Use saveToJsonString() instead.');
}

/// Web stub implementation of file reading.
Future<String> readStringFromFile(dynamic file) async {
  throw UnsupportedError(
      'File persistence is not supported on the Web platform. Use loadFromJsonString() instead.');
}
