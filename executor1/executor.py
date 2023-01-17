import numpy as np
import torch
import sys, os
import pandas as pd
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from jina import Executor, requests, Document, DocumentArray
from utils_docs import get_docs_embedding, get_similarity, load_model



class MyExecutor(Executor):
    """MyExecutor This is the executor class that runs all the task and logic to be applied on DocumentArray.

    Args:
        Executor (Executor): Inherits from class Executor of Jina.
    """
    def __init__(self, docs_path: str = 'documents/summations.csv', **kwargs):
        """__init__ The constructor of Executor to load the necessary models, documents, and get the embedding.

        Args:
            docs_path (str, optional): _description_. Defaults to 'documents/summations.csv'.
        """
        print("Executer starting!")
        super().__init__(**kwargs)
        # with open(docs_path, 'r') as f:
        #     self.documents_raw = f.readlines().decode(errors='ignore')
        docs_path = os.path.join(dir_path, docs_path)
        self.documents_raw = pd.read_csv(docs_path,  encoding='utf-8')
        self.documents_raw = list(self.documents_raw.text.values)
        self.docs = DocumentArray.empty(len(self.documents_raw))
        self.tokenizer, self.model = load_model()
        self.docs.embeddings = get_docs_embedding(self.documents_raw, self.tokenizer, self.model)
        
    @requests(on='/doc-similarity')
    def bar(self, docs: DocumentArray, **kwargs):
        """bar This functions calculate the similarity of the input documents with target documents.

        Args:
            docs (DocumentArray): List of documents in this case text.
        """
        # Loop through the doc in input document array, get the embedding, calculate it similarity with target documents.
        for doc in docs:
            # print(doc.content)
            doc.embedding = get_docs_embedding([doc.text], self.tokenizer, self.model)
            similarity_scores, similarity_idx = get_similarity(doc, self.docs, doc.adjacency)
            doc.embedding = None
            scores = [Document(text = text.strip(), weight = similarity) for text, similarity in (zip(np.array(self.documents_raw)[similarity_idx], similarity_scores))]
            doc.matches = DocumentArray(scores)

# It's here for test purpose it should be delete. 
if __name__ == "__main__":
    test_obj = MyExecutor(docs_path='documents\summations.csv' )
    da = test_obj.bar(DocumentArray(Document(text="Preventing amyloid plaque formation can halt Alzherimer's progression", adjacency=5)))
    print("Hello!")