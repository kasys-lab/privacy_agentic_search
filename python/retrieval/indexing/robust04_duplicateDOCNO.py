import os
import re
from collections import defaultdict
import pyterrier as pt

DISK45_PATH = "/home/franz/.ir_datasets/disks45/corpus"

print(f"Scanning files from: {DISK45_PATH}")

# Find all files recursively
files = pt.io.find_files(DISK45_PATH)

# Remove obvious junk/non-document files
bad_patterns = [
    '/CR/',
    '/AUX/',
    'READCHG',
    'READFRCG',
    '/DTDS/',
    'MD5SUM'
]

for b in bad_patterns:
    files = [f for f in files if b not in f]

print(f"After filtering: {len(files)} files")

# Regex for TREC DOCNO tags
docno_re = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>", re.IGNORECASE)

# Track where docnos appear
docno_to_files = defaultdict(list)

# Track malformed files
files_with_no_docno = []

for path in files:

    try:
        with open(path, "r", errors="ignore") as f:
            text = f.read()

        matches = docno_re.findall(text)

        # No DOCNO found at all
        if not matches:
            files_with_no_docno.append(path)
            continue

        for m in matches:
            docno = m.strip()
            docno_to_files[docno].append(path)

    except Exception as e:
        print(f"ERROR reading {path}: {e}")

print("\n==============================")
print("FILES WITH NO DOCNO TAGS")
print("==============================")

for p in files_with_no_docno:
    print(p)

print("\n==============================")
print("DUPLICATE DOCNOS")
print("==============================")

duplicate_count = 0

for docno, paths in sorted(docno_to_files.items()):

    if len(paths) > 1:

        duplicate_count += 1

        print("\n--------------------------------")
        print(f"DOCNO: {repr(docno)}")
        print(f"Occurrences: {len(paths)}")

        for p in paths:
            print("  ", p)

print("\n==============================")
print("SUMMARY")
print("==============================")
print(f"Duplicate DOCNOs: {duplicate_count}")
print(f"Files with no DOCNO: {len(files_with_no_docno)}")