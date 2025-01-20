from fastapi import FastAPI
from pydantic import BaseModel
import openai
import pinecone

# Initialize OpenAI API and Pinecone
openai.api_key = "your_openai_api_key"
pinecone.init(api_key="your_pinecone_api_key", environment="us-west1-gcp")

app = FastAPI()

# Example index setup (replace with actual index name)
index = pinecone.Index("your_index_name")

class Query(BaseModel):
    query: str

@app.post("/search/")
async def search(query: Query):
    # Perform vector search with Pinecone
    query_vector = openai.Embedding.create(input=query.query, model="text-embedding-ada-002")["data"][0]["embedding"]
    results = index.query([query_vector], top_k=5, include_metadata=True)
    
    return {"results": results["matches"]}
