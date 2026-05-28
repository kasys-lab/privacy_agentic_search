import os
import argparse
import importlib
import random
import numpy as np
import pandas as pd
from embeddings import glove
from utils.Timer import Timer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # This ensures that even if you use random logic elsewhere, it's consistent
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    set_seed(42)  # Set a fixed seed for reproducibility
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data")
    SAVE_PATH = os.path.join(DATA_PATH, "protected-queries")

    os.makedirs(SAVE_PATH, exist_ok=True)

    #basePath = "/ssd/data/faggioli/24-ECIR-FF"
    #dataPath = f"{basePath}/data"
    #savePath = f"{dataPath}/protected-queries"

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epsilon", default=5, type=int)  # "convdr"
    parser.add_argument('-m', '--mechanism', default="MahalanobisMechanism")
    parser.add_argument("-n", "--ntests", default=10, type=int)
    parser.add_argument('-r', '--repetitions', default=20, type=int) # it was 100 before
    parser.add_argument('-v', '--vectors', default="glove.6B.300d")
    parser.add_argument('-vo', '--vocab', default="robust04-vocab-30000.txt")
    parser.add_argument('-w', '--write', action='store_true', default=False, help="Whether to write the results to a file.")

    args = parser.parse_args()

    vocab_path = os.path.join(DATA_PATH, "vocabularies", args.vocab)
    vector_path = os.path.join(DATA_PATH, "vectors", f"{args.vectors}.txt")

    print(f"Loading embeddings from: {vector_path}")
    print(f"Using vocabulary from: {vocab_path}")

    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Vector file not found. Check your path: {vector_path}")

    representation = glove(vector_path, vocab=vocab_path)

    class_ = getattr(importlib.import_module("scrambling_queries.mechanisms"), args.mechanism)
    mechanism = class_(representation.get_size(), epsilon=args.epsilon, embeddings=representation.get_embeddings_matrix())

    words = list(representation._word2int.keys())

    f_out = None
    if args.write:
        results_filename = f"mapping_e{args.epsilon}_{args.mechanism}_{args.ntests}ntests_{args.repetitions}reps.txt"
        results_file_path = os.path.join(SAVE_PATH, results_filename)
        print(f"Writing results to: {results_file_path}")
        f_out = open(results_file_path, "w")

    freqs = []
    try:
        for i in range(args.ntests):
            w = random.choice(words)
            reps = " ".join([w] * args.repetitions)
            perturbed = mechanism.get_protected_vectors(representation.encode_sentence(reps))
            encodings = representation.get_k_closest_terms(perturbed, 1)
            #print(encodings)

            if f_out:
                    for entry in encodings:
                        f_out.write(f"{w}\t{entry}\n")
                    f_out.write("\n--------------------\n\n")

            freq = np.sum([1 if t == w else 0 for t in encodings]) / args.repetitions
            freqs.append(freq)

            print(f"word: {w}: freq not changed: {freq:.3f} -- curr avg: {np.mean(freqs):.3f}")
    finally:
        if f_out:
            print("-" * 50)
            print(f"FINAL AVERAGE SUCCESS RATE: {np.mean(freqs):.3f}")
            
            f_out.close()
            print(f"Results written to: {results_file_path}")


if __name__ == '__main__':
    main()