# Weave Similarity

## Description
This is the test project to get the rank the similar documents to input document/request.
## Version
Test
## Visuals
<!-- ![Endpoint structure!](project_src/visual.png) -->

## Run the endpoint locally.
To run the endpoint locally make sure you install the jina. Following link helps you to install it.
https://docs.jina.ai/get-started/install/
For example to install in local env.
```
pip install -U jina
```
Activate the local environemt, navigate to project directory and run the following command.
```
jina flow --uses flow.yml
```
As a result of the above command you should see the following. 
──── 🎉 Flow is ready to serve! ────
╭────────────── 🔗 Endpoint ───────────────╮
│  ⛓     Protocol                    GRPC  │
│  🏠       Local           0.0.0.0:54321  │
│  🔒     Private    192.168.200.56:54321  │
│  🌍      Public    81.223.121.124:54321  │
╰──────────────────────────────────────────╯
To reference see here: https://docs.jina.ai/get-started/create-app/

## Make request to local endpoint.
Run the file client.py file. To run the file make sure you have install the jina and your server is running.
You only need to change the text, and the adjacency value.
It is possible to have more a request with multiple text. Look here: https://docarray.jina.ai/fundamentals/documentarray/
The adjacency defines the total number of top most similar documents. For example here it returns the 5 most similar documents.
```
from jina import Client, DocumentArray, Document

if __name__ == '__main__':
    c = Client(host='http://127.0.0.1:54322')
    da = c.post('/doc-similarity', DocumentArray(Document(text="Preventing amyloid plaque formation can halt Alzherimer's progression", adjacency=5)))
    matches = da[0].matches
    
    for match in matches:
        print(f"text: {match.text}, similarity score: {match.weight}")
```


## Support
Ali Akbar Ahmadi, Data Scientist, sau.ahmadi@gmail.com.

## Roadmap


## Contributing


## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License


## Project status
Development.
