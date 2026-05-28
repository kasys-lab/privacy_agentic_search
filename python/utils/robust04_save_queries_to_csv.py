import pyterrier as pt
import pandas as pd
import os

if not pt.java.started():
    pt.java.init()

dataset = pt.get_dataset(
    "irds:disks45/nocr/trec-robust-2004"
)

topics = dataset.get_topics()

print(topics.head())
print(topics.columns)

# Robust04 uses the "title" field as the query
topics = topics[["qid", "title"]]

topics = topics.rename(
    columns={"title": "query"}
)

topics["qid"] = topics["qid"].astype(str)

save_path = "../../data/queries/robust04/valid/topics.csv"

os.makedirs(
    os.path.dirname(save_path),
    exist_ok=True
)

topics.to_csv(
    save_path,
    sep=";",
    header=False,
    index=False
)

print(f"Saved {len(topics)} queries to:")
print(save_path)