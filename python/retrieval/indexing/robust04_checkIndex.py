import pyterrier as pt

INDEX_DIR = "../../../data/indexes/robust04/pyterrier"

# Initialize PyTerrier
if not pt.java.started():
    pt.java.init()

# Load index
index = pt.IndexFactory.of(INDEX_DIR)

print("\n==============================")
print("INDEX STATISTICS")
print("==============================")

stats = index.getCollectionStatistics()

print(f"Documents:      {stats.getNumberOfDocuments():,}")
print(f"Tokens:         {stats.getNumberOfTokens():,}")
print(f"Unique terms:   {stats.getNumberOfUniqueTerms():,}")
print(f"Postings:       {stats.getNumberOfPointers():,}")

print("\n==============================")
print("TEST RETRIEVAL - Query: 'tokyo trip'")
print("==============================")

# Create BM25 retriever
bm25 = pt.terrier.Retriever(index, wmodel="BM25")

# Run query
results = bm25.search("tokyo trip")

#print(f"Retrieved {len(results)} results\n")

# Show top 3 with title + snippet
for rank, (_, row) in enumerate(results.head(3).iterrows(), start=1):

    title = row.get("title", "")
    body = row.get("body", "")

    # Clean whitespace
    title = " ".join(str(title).split())
    body = " ".join(str(body).split())

    # Short preview
    snippet = body[:500]

    print("================================")
    print(f"Rank:   {rank}")
    print(f"DOCNO:  {row['docno']}")
    print(f"Score:  {row['score']:.4f}")
    print(f"Title:  {title}")
    print("\nSnippet:")
    print(snippet)
    print()