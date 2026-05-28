import os
import pyterrier as pt
from pathlib import Path
import ir_datasets
from ir_datasets.util.download import DownloadConfig
import string
from tqdm import tqdm
from collections import Counter, namedtuple
import json
import argparse

if not pt.started():
    pt.init()

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collection", default="robust04")  # "convdr"
parser.add_argument("-k", "--topk", default=-1, type=int)  # "convdr"
args = parser.parse_args()

def msmarco_generate():
    Doc = namedtuple('Doc', 'doc_id text')
    msmarcofile = os.path.expanduser("~/work/24-ECIR-FF/data/MSMARCO/collection/collection.tsv")
    if not os.path.exists(msmarcofile):
        print(f"Error: MSMARCO file not found at {msmarcofile}")
        return
    with pt.io.autoopen(msmarcofile, 'rt') as corpusfile:
        for l in corpusfile:
            try:
                docno, passage = l.split("\t")
            except Exception as e:
                print(l)
                pass
            yield Doc(docno, passage)


#os.environ['JAVA_HOME'] = '/ssd/data/faggioli/SOFTWARE/jdk-11.0.11'
# Use your Conda-installed Java
conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    os.environ['JAVA_HOME'] = conda_prefix

#os.environ['IR_DATASETS_HOME'] = "/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/CORPORA/IR_DATASETS_CORPORA/"
# Set local directory for downloads
os.environ['IR_DATASETS_HOME'] = os.path.expanduser("~")

if args.collection == "robust04":
    dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
    iterator = dataset.docs_iter()
elif args.collection=="msmarco-passage":
    iterator = msmarco_generate()

# Set output path (make sure the directory exists)
base_vocab_path = os.path.expanduser("~/work/24-ECIR-FF/data/vocabularies")
os.makedirs(base_vocab_path, exist_ok=True)

vocpath = f"{base_vocab_path}/{args.collection}-vocab-frequencies.txt"


def process_document(doc):
    fields = [f for f in doc._fields if f != "doc_id"]
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    def process_field(txt): return str(txt).lower().replace("\n", " ").translate(translator)
    out = " ".join(list(map(process_field, [getattr(doc, f) for f in fields])))
    return out


if not os.path.exists(vocpath):
    print(f"Generating frequency dictionary for {args.collection}...")
    vocab = {}

    for doc in tqdm(iterator):
        content = process_document(doc)
        frequency = Counter(content.split())

        for w, f in frequency.items():
            if w in vocab:
                vocab[w] += f
            else:
                vocab[w] = f

    with open(vocpath, "w") as f:
        json.dump(vocab, f)
    print(f"SUCCESS: Frequency dictionary created at: {vocpath}")

else:
    print(f"LOADING: Existing frequency dictionary found at: {vocpath}")
    with open(vocpath, "r") as f:
        vocab = json.load(f)

# Sort and select Top-K
print(f"Sorting vocabulary to select top-{args.topk} words...")
topkwords = [w for w, _ in sorted(list(vocab.items()), key=lambda x: -x[1])][:args.topk]

final_output_path = f"{base_vocab_path}/{args.collection}-vocab-{args.topk}.txt"

with open(final_output_path, "w") as f:
    for l in topkwords:
        f.write(f"{l}\n")

print(f"SUCCESS: Final vocabulary file created at: {final_output_path}")
