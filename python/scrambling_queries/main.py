import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, f"{PROJECT_ROOT}/python")
#sys.path.append(".")

import pyterrier as pt
from embeddings import glove
import numpy as np
import argparse
import importlib
from pathlib import Path
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"

dataPath = os.path.join(PROJECT_ROOT, "data")
savePath = os.path.join(dataPath, "queries")

collections = {'robust04': 'irds:disks45/nocr/trec-robust-2004',
               'trec-covid': 'irds:beir/trec-covid',
               'msmarco-passage': 'irds:msmarco-passage/trec-dl-2019/judged'}


def dp_mechanism(queries, args):
    print("    + importing representation")
    representation = glove(f"{dataPath}/vectors/{args.vectors}.txt",
                           vocab=f"{dataPath}/vocabularies/{args.collection}-vocab.txt",
                           workers=1)

    print("    + representing queries")
    queries['embeddings'] = queries['query'].apply(representation.encode_sentence)

    print("    + initializing mechanism")
    class_ = getattr(importlib.import_module("scrambling_queries.mechanisms"), args.mechanism)
    mechanism = class_(representation.get_size(), epsilon=args.epsilon, embeddings=representation.get_embeddings_matrix())

    print("\nChecking for Empty Embeddings before Scrambling Loop...")
    for idx, row in queries.iterrows():
        emb = np.asarray(row['embeddings'])
        if emb.size == 0:
            print(f"⚠️  WARNING: Query ID '{row['qid']}' resulted in an EMPTY embedding array!")
            print(f"   Original clean text: '{row['query']}'")
            print(f"   Check if these words exist in your GloVe file.\n")

    print("    + computing protected vectors and representations")
    for k in tqdm(range(args.num_perturbations)):
        queries['noisy_embeddings'] = queries['embeddings'].apply(mechanism.get_protected_vectors)

        queries['noisy_query'] = queries['noisy_embeddings'].apply(lambda x: " ".join(representation.get_k_closest_terms(x, 1)))

        # check if the path exists or create it
        Path(f"{savePath}/{args.collection}/{args.mechanism}").mkdir(parents=True, exist_ok=True)

        # save the query
        queries[["qid", "noisy_query"]].to_csv(f"{savePath}/{args.collection}/{args.mechanism}/{args.epsilon}_{args.vectors}_{k}.csv",
                                               index=False, header=False, sep=";")


def sota_approaches(queries, args, dataset):

    class_ = getattr(importlib.import_module("scrambling_queries.sota"), args.mechanism)
    scrambler = class_(dataset=dataset, collection=args.collection)

    scrambled_queries = scrambler.scramble(queries)

    # check if the path exists or create it
    Path(f"{savePath}/{args.collection}/{args.mechanism}").mkdir(parents=True, exist_ok=True)

    # save the query
    scrambled_queries.to_csv(f"{savePath}/{args.collection}/{args.mechanism}/{args.mechanism}.csv",
                                           index=False, header=False, sep=";")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection", default="robust04")
    parser.add_argument("-e", "--epsilon", default=5, type=float)  # "convdr"
    parser.add_argument("-k", "--num_perturbations", default=20, type=int)  # "convdr", was 100 before
    parser.add_argument('-m', '--mechanism', default="MahalanobisMechanism")
    parser.add_argument('-v', '--vectors', default="glove.6B.300d")
    args = parser.parse_args()

    epsilon_targets = [1.0, 5.0, 10.0, 12.5, 15.0, 17.5, 20.0, 50.0]

    # Commented out: installed in Conda environemnt
    #os.environ['JAVA_HOME'] = '/ssd/data/faggioli/SOFTWARE/jdk-11.0.11'
    if not pt.started():
        pt.init()

    print("    + importing queries")
    qfield = {"robust04": "title", "msmarco-passage": "text", "trec-covid": "query"}
    dataset = pt.get_dataset(collections[args.collection])
    queries = dataset.get_topics(qfield[args.collection])
    #print(queries)

    # remove punctuation
    punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    translator = str.maketrans(punctuation, ' ' * len(punctuation))
    queries['query'] = queries['query'].apply(lambda x: x.lower().replace("-", "").translate(translator)) # changed to really apply the punctuation removal

    for eps in epsilon_targets:
        args.epsilon = eps
        print(f"\n========================================================")
        print(f"computing perturbed queries for {args.collection} using {args.mechanism} with epsilon {args.epsilon}")
        print(f"\n========================================================")

        print("    + scrambling")
        if hasattr(importlib.import_module("scrambling_queries.mechanisms"), args.mechanism):
            dp_mechanism(queries, args)
        else:
            sota_approaches(queries, args, dataset)


