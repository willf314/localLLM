# Vector database wrapper used by AI Control Service

import json
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct, CollectionInfo, ScoredPoint, Record

# constants
COLLECTION_NAME = "collection1"
MAX_CHUNKS_TO_RETURN = 5
TRIM_CHUNK_LEN = 80                

# helper function to trim the length of a chunk, and remove any newline characters for logging purposes
def trimChunk(chunk, max_length):
    chunk = chunk.replace('\n', ' ').replace('\r', ' ')
    if len(chunk) <= max_length:
        return chunk
    else:
        return chunk[:max_length] + "..."

# Vector database wrapper class called by the AI Control service. Modify this class if changing vector databases
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
    def persist_chunk(self, source_filename, text_chunk, embedding):  
        
        # create a uuid based on hostname and time
        idx = str(uuid.uuid1())
        
        # insert chunk into vector db
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

        # log summary info
        self.logger.info("persisted chunk to db:")
        self.logger.info("  id:[" + idx + "]")
        self.logger.info("  chunk size:" + str(len(text_chunk)))
        self.logger.info("  source_filename:[" + source_filename + "]")
        
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
        i=0
        for result in search_result:
            i += 1
            self.logger.info("retrieved chunk:")
            self.logger.info("  id:" + str(result.id))
            self.logger.info("  score:" + str(result.score))
            self.logger.info("  payload:" + str(result.payload))
            self.logger.info("\n")
                 
        self.logger.info("retrieved " + str(i) + " results")
        
        return search_result

    # returns true if the file already exists in the vectorDB
    def file_exists(self, source_filename):
        # look for any existing chunks with this filename. Note we only need 1 result
        search_result = self.vectorDB.scroll(
            collection_name=COLLECTION_NAME,
            limit=1,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_filename",
                        match=models.MatchValue(value=source_filename),
                    ),
                ]
            ),
        )

        # unpack search_result. Could find nothing in API doco on purpose of the Union object 
        records, union_obj = search_result
        
        # count results - count property didn't seem to work?
        
        i=0
        for record in records:
            i += 1
            self.logger.error("oops - found chunk with matching source_filename:")
            self.logger.error("  id:[" + str(record.id) + "]")
            self.logger.error("  payload:[" + trimChunk(str(record.payload), TRIM_CHUNK_LEN) + "]")
                                                    
        # return true if we found at least 1 matching chunk         
        return ( i > 0 )

    # join search result payload together to form a context string    
    def convert_to_string(self, search_results):
        # for now we take the full Qdrant payload JSON including field names and values and concatenate it into a string
        context=""
        for result in search_results:
            context += json.dumps(result.payload) + " "   
        return context
    
    
    

    






