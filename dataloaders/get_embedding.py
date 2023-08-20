import gensim.downloader as api
import numpy as np
model = api.load('glove-wiki-gigaword-300')  # download trained model
categories = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']

w2vectors = []
for word in categories:
    w2vectors.append(model[word])
w2v = [np.expand_dims(x, axis=0) for x in w2vectors]
embeddings = np.concatenate(w2v, axis=0)
np.save('dataloaders/S3DIS_glove.npy', embeddings)
# 'word2vec-ruscorpora-300',glove-wiki-gigaword-300, 'word2vec-google-news-300','conceptnet-numberbatch-17-06-300','fasttext-wiki-news-subwords-300'


categories = ['clutter', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf',
                   'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door',
                   'window', 'curtain', 'refrigerator', 'picture', 'cabinet', 'furniture']
w2vectors = []

for word in categories:
    w2vectors.append(model[word])

w2v = [np.expand_dims(x, axis=0) for x in w2vectors]
embeddings = np.concatenate(w2v, axis=0)
np.save('dataloaders/ScanNet_glove.npy', embeddings)