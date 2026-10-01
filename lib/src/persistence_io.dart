import 'dart:io';

/// Native IO implementation of atomic file persistence.
Future<void> saveStringToFile(dynamic file, String content) async {
  final File f = file is File ? file : File(file.toString());
  final tempFile = File('${f.path}.tmp');
  await tempFile.writeAsString(content, flush: true);
  if (await f.exists()) {
    await f.delete();
  }
  await tempFile.rename(f.path);
}

/// Native IO implementation of file reading.
Future<String> readStringFromFile(dynamic file) async {
  final File f = file is File ? file : File(file.toString());
  return f.readAsString();
}
