import os
import argparse
import string
from pathlib import Path
from tqdm import tqdm
import ir_datasets

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

dataPath = os.path.join(PROJECT_ROOT, "data")
vocabSavePath = os.path.join(dataPath, "vocabularies")

#os.environ['JAVA_HOME'] = '/ssd/data/faggioli/SOFTWARE/jdk-11.0.11'

#os.environ['IR_DATASETS_HOME'] = "/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/CORPORA/IR_DATASETS_CORPORA/"
os.environ['IR_DATASETS_HOME'] = os.path.expanduser("~")

collections_mapping = {
    'robust04': 'disks45/nocr/trec-robust-2004',
    'trec-covid': 'beir/trec-covid',
    'msmarco-passage': 'msmarco-passage'
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection", default="robust04", choices=['robust04', 'trec-covid', 'msmarco-passage'])
    args = parser.parse_args()

    print(f"Loading ir_dataset for: {args.collection}")
    dataset = ir_datasets.load(collections_mapping[args.collection])

    vocab = set()
    docs = []
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    def process_field(txt): 
        return str(txt).lower().replace("\n", " ").translate(translator)

    print(f"Step 1: Processing documents to extract unique words...")

    for doc in tqdm(dataset.docs_iter()):
        fields = [f for f in doc._fields if f not in ["doc_id", "doc_no"]]
        content = " ".join(list(map(process_field, [getattr(doc, f) for f in fields])))
        docs.append(content.split())

    print("Step 2: Flattening tokens into a unique vocabulary set...")
    vocab = set().union(*docs)
    
    # Clean up empty strings if any slipped past the translation layout
    vocab.discard("")

    Path(vocabSavePath).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(vocabSavePath, f"{args.collection}-vocab.txt")

    print(f"Step 3: Saving {len(vocab)} unique terms to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for word in sorted(vocab):
            f.write(f"{word}\n")
            
    print("Vocabulary successfully created!")
