from jina import Client, DocumentArray, Document

if __name__ == '__main__':
    c = Client(host='http://0.0.0.0:54322')
    da = c.post('/doc-similarity', DocumentArray(Document(text="Preventing amyloid plaque formation can halt Alzherimer's progression", adjacency=5)))
    matches = da[0].matches
    
    for match in matches:
        print(f"text: {match.text}, similarity score: {match.weight}")
