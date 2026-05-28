import os
import re
from collections import defaultdict
import pyterrier as pt

DISK45_PATH = "/home/franz/.ir_datasets/disks45/corpus"

print(f"Scanning files from: {DISK45_PATH}")

# Find all files
files = pt.io.find_files(DISK45_PATH)

# Remove non-document/junk files
bad_patterns = [
    '/CR/',
    '/AUX/',
    '/DTDS/',
    'MD5SUM',
    'READCHG',
    'READFRCG',

    # README files containing example docs
    'READMEFB',
    'READMEFR',
    'READMEFT',
    'READMELA'
]

for b in bad_patterns:
    files = [f for f in files if b not in f]

print(f"After filtering: {len(files)} files")

# Regexes
doc_re = re.compile(r"<DOC>", re.IGNORECASE)
docno_re = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>", re.IGNORECASE)

# Statistics
total_docs = 0
empty_docnos = 0

docno_to_files = defaultdict(list)

for path in files:

    try:
        with open(path, "r", errors="ignore") as f:
            text = f.read()

        # Count DOC blocks
        docs_in_file = len(doc_re.findall(text))
        total_docs += docs_in_file

        # Extract DOCNOs
        matches = docno_re.findall(text)

        for m in matches:

            docno = m.strip()

            if docno == "":
                empty_docnos += 1

            docno_to_files[docno].append(path)

    except Exception as e:
        print(f"ERROR reading {path}: {e}")

# Find duplicates
duplicates = {
    k: v for k, v in docno_to_files.items()
    if len(v) > 1
}

print("\n==============================")
print("COLLECTION STATISTICS")
print("==============================")

print(f"Files scanned:        {len(files)}")
print(f"Total DOC blocks:     {total_docs}")
print(f"Total DOCNOs:         {len(docno_to_files)}")
print(f"Duplicate DOCNOs:     {len(duplicates)}")
print(f"Empty DOCNOs:         {empty_docnos}")

if duplicates:

    print("\n==============================")
    print("DUPLICATE DOCNOS")
    print("==============================")

    for docno, paths in sorted(duplicates.items()):

        print("\n--------------------------------")
        print(f"DOCNO: {repr(docno)}")
        print(f"Occurrences: {len(paths)}")

        for p in paths:
            print("  ", p)