import os
import sys
import pyterrier as pt

# Maintain single-core server etiquette
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["JAVA_OPTS"] = "-XX:ParallelGCThreads=1 -XX:CICompilerCount=2 -XX:ActiveProcessorCount=1"

if not pt.java.started():
    pt.java.init()

DISK45_PATH = "/home/franz/.ir_datasets/disks45/corpus"
INDEX_DIR = "../../../data/indexes/robust04/pyterrier"

# Ensure target clean workspace directories exist
os.makedirs(INDEX_DIR, exist_ok=True)

print(f"Scanning files from source raw data path: {DISK45_PATH}")
files = pt.io.find_files(DISK45_PATH)

# --- EXPANDED FIX: Drop structural DTD folders and root text logs ---
bad_patterns = [
    '/CR/', '/AUX/', 'READCHG', 'READFRCG',
    '/DTDS/', 'MD5SUM',  # Clears out the source logs causing the key crash
    'READMEFB', 'READMEFR', 'READMEFT', 'READMELA' # Exclude README files with sample docs
]

for b in bad_patterns:
    files = list(filter(lambda f: b not in f, files))
    
print(f"Filters applied. Indexing {len(files)} target document chunks...")

pt.terrier.J.ApplicationSetup.setProperty(
    "metaindex.compressed.reverse.allow.duplicates",
    "true"
)

# Initialize indexer configuration properties directly
indexer = pt.TRECCollectionIndexer(INDEX_DIR, verbose=True)
indexref = indexer.index(files)

#print("Inverted index created successfully without constraints!")