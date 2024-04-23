from src.text_clustering import ClusterClassifier
from datasets import load_dataset

SAMPLE = 100_000

texts = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train").select(range(SAMPLE))["text"]

cc = ClusterClassifier(embed_device="cuda")

# run the pipeline:
embs, labels, summaries = cc.fit(texts)

# show the results
cc.show()

# save 
cc.save("./cc_100k")