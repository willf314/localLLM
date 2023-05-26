# initiliase Qdrant vector DB

import json
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# constants
COLLECTION_NAME = "collection1"
EMBED_VECTOR_SIZE = 5120

# load environment variable
QDRANT_URL = os.environ.get("QDRANT_URL")

if __name__ == '__main__':

    # connect to Qdrant 
    qdrant_client = QdrantClient(QDRANT_URL)
    print("connected to Qdrant db [" + QDRANT_URL+"]")

    # recreate collection. This will delete any existing collection with the same name
    qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBED_VECTOR_SIZE, distance=Distance.COSINE)
    )  

    # inform success
    print("created collection [" + COLLECTION_NAME + "] with Vector size:" + str(EMBED_VECTOR_SIZE))
    print("initialisation complete")
    
    
