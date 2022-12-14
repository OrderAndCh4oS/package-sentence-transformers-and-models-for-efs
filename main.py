import sys

sentence_transformers_path = '/Users/seancooper/code/sentence_transformers/sentence_transformers_lib'
sys.path.append(sentence_transformers_path)

from sentence_transformers import SentenceTransformer, util

query = "A significant proportion of alcoholics manage to live with the disease daily"
docs = [
    "Some 85% of alcoholics don't mind drinking and live quite happily day to day",
    "Alcoholism is a devastating affliction, that often causes harm to the drinkers loved ones",
    "I like ponies and riding bareback especially",
    "I hear venice is nice this time of year",
    "one in ten people drink more than they should, one to two litres is enough for most people"
]


model = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')
modelPath = "./msmarco-distilbert-cos-v5-model"
model.save(modelPath)
model = SentenceTransformer(modelPath)


def main():
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    print(doc_emb.shape)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = list(zip(docs, scores))

    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    for doc, score in doc_score_pairs:
        print(score, doc)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
