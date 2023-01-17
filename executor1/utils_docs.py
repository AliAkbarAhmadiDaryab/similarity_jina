import torch 
import numpy as np
from transformers import AutoTokenizer, AutoModel
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

def load_model():
    """ This function is used to load the model and return the tokenizer and transformer model.
    """
    model_name = 'dmis-lab/biobert-v1.1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model
    
def get_docs_embedding(sentences, tokenizer, model):
    """get_docs_embedding: This function is used to get the embeddings for a given list of sentences.
    Args:
        sentences (list): The list of sentences to be embedded. 
        tokenizer (bert tokenizer): This is the sentence tokenizer from the same model.
        model (model bert): This is the loaded BERT model.

    Returns:
        Pytorch tensor: For every sentence it returns the contextual representation of the sentence of CLS token.
    """
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    token_embeddings = model_output.last_hidden_state[:, 0, :]
    print(f"Token embeddings shape: {token_embeddings.size()}")
    return token_embeddings     

def get_similarity(request_doc, documents, total_similar_docs):
    """get_similarity It returns the similarity scores and their indexes in the documents.

    Args:
        request_doc (Document): The request sentence for which we look for the most similar documents.
        documents (DocumentArray): The rest of documents
        total_similar_docs (Integer): The total top ranked documents.

    Returns:
        numpy array, numpy array : The scores and score index.
    """
    documents_embeddings_np = documents.embeddings.detach().numpy()
    request_embedding_np = request_doc.embedding.detach().numpy()
    scores = cosine_similarity(request_embedding_np, documents_embeddings_np)[0]
    return scores[scores.argsort()[-total_similar_docs:][::-1]], scores.argsort()[-total_similar_docs:][::-1]
    
    