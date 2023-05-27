# Vector database wrapper used by AI Control Service

import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, CollectionInfo, ScoredPoint

# constants
COLLECTION_NAME = "collection1"
MAX_CHUNKS_TO_RETURN = 5


class VectorDB:

    # constructor for VectorDB wrapper class. Connects to the specified DB service endpoint
    def __init__(self, db_url,logger):
        self.url = db_url
        self.logger = logger

        # connect to Qdrant vector database
        self.logger.info("connecting to qdrant db at [" + self.url + "] ...")
        self.vectorDB = QdrantClient(self.url)
        self.logger.info("succesfully connected to vector DB")

        # show collection status info. this will throw an error if collection has not been created with initVectorDB 
        self.logger.info("connecting to collection [" + COLLECTION_NAME + "]...")
        collectionInfo = self.vectorDB.get_collection(collection_name=COLLECTION_NAME) 
        self.logger.info("collection status:")
        self.logger.info("   status:%s", str(collectionInfo.status))
        self.logger.info("   points_count:%s", str(collectionInfo.points_count))
        self.logger.info("   vectors_count:%s", str(collectionInfo.vectors_count))
        self.logger.info("   segments_count:%s", str(collectionInfo.segments_count))    
        self.logger.info("   payload_schema:%s", str(collectionInfo.payload_schema))
    
    # persist chunk to vector db - Qdrant
    def persist_chunk(self, idx, source_filename, text_chunk, embedding):  
        self.vectorDB.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=idx, 
                vector=embedding,
                payload={"text": text_chunk, "source_filename" : source_filename}
            )        
        ]
        )
        return()

    # perform vector DB similarity search
    def retrieve_matches(self, embedding):
        # perform search, return up to 5 results
        search_result = self.vectorDB.search(
                collection_name=COLLECTION_NAME,
                query_vector=embedding, 
                limit=MAX_CHUNKS_TO_RETURN,
                with_payload=True,
                with_vectors=False
                )    

        # log search results
        self.logger.info("retrieved " + str(search_result.count) + " results")
        for result in search_result:
            self.logger.info("id:" + str(result.id))
            self.logger.info("score:" + str(result.score))
            self.logger.info("payload:" + str(result.payload))
            self.logger.info("\n")
                 
        return search_result

    # join search result payload together to form a context string
    # this is in the db wrapper class because different Vector DB's return results in different formats    
    def convert_to_string(self, search_results):
        context=""
        for result in search_results:
            context += json.dumps(result.payload) + " "   
        return context


    

    






