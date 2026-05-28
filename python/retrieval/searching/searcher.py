import pandas as pd
import argparse
import os
#from search_faiss import search_faiss
from search_pyterrier import search_pyterrier
import string

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collection", default="robust04")
parser.add_argument("-q", "--queries", default=None)
parser.add_argument("--mechanism", default=None)
parser.add_argument("-m", "--model", default="tasb")
args = parser.parse_args()

mechanism_eps_values = [
    1.0,
    5.0,
    10.0,
    12.5,
    15.0,
    17.5,
    20.0,
    50.0
]

saveFolder = f"../../../data/runs"
os.makedirs(saveFolder, exist_ok=True)

#remove punctuation from queries
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))

def run_retrieval(query_file_name):
    # read queries
    query_path = f"../../../data/queries/{args.collection}/valid/{query_file_name}.csv"
    print(f"\nLoading queries from: {query_path}")
    queries = pd.read_csv(query_path, header=None, names=["qid", "query"], sep=";", dtype={"query":str})

    queries['query'] = queries['query'].apply(lambda x: str(x).translate(translator))

    if args.model in ["tasb", "contriever", "glove"]:
        out = search_faiss(queries, args.collection, args.model)
    else:
        out = search_pyterrier(queries, args.collection, args.model)

    output_path = f"{saveFolder}/{args.collection}_{args.model.replace('_', '-')}_{query_file_name}.csv"
    out.to_csv(output_path, header=None, index=None, sep="\t")
    
    print(f"Saved run to: {output_path}")


if args.queries is not None:
    run_retrieval(args.queries)

elif args.mechanism is not None:
    for eps in mechanism_eps_values:
        query_file_name = f"{args.mechanism}_{eps}"

        print("\n===================================")
        print(f"Running mechanism sweep for eps={eps}")
        print("===================================")

        run_retrieval(query_file_name)

else:
    raise ValueError(
        "Provide either --queries/-q or --mechanism"
    )