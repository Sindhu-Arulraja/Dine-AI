from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import chromadb
from chromadb.utils import embedding_functions

app = FastAPI()

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

chroma_client = None
collection = None

def get_chroma_collection():
    global chroma_client, collection
    if collection is not None:
        return collection
    if os.path.exists("./chroma_db"):
        try:
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            collection = chroma_client.get_collection(name="restaurants", embedding_function=sentence_transformer_ef)
            print("Connected to ChromaDB successfully.")
            return collection
        except Exception as e:
            print(f"ChromaDB connect error: {e}")
            return None
    return None

@app.get("/")
async def serve_frontend():
    return FileResponse('static/index.html')

@app.get("/results.html")
async def serve_results():
    return FileResponse('static/results.html')

@app.get("/api/catalog-search")
async def get_catalog(
    cuisine: str = Query(""),
    location: str = Query(""),
    event: str = Query(""),
    budget: str = Query(""),
    people: str = Query(""),
    time: str = Query("")
):
    col = get_chroma_collection()
    if not col:
         return JSONResponse(status_code=500, content={"status": "error", "message": "Vector Database not initialized."})

    search_terms = []
    if cuisine: search_terms.append(f"{cuisine} food")
    if location: search_terms.append(f"in {location}")
    if event: search_terms.append(f"vibe: {event}")
    if budget: search_terms.append(f"{budget} budget")

    query_text = " ".join(search_terms) if search_terms else "Popular restaurant in Bangalore"

    try:
        # Phase 5: Fetching pure 8 results for the UI grid, no LLM
        results = col.query(query_texts=[query_text], n_results=8)
        
        top_restaurants = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                
                # We pull all possible UI data out of Chroma for the frontend popup modal
                top_restaurants.append({
                    "name": meta.get('name', 'Unknown Restaurant'),
                    "cuisines": meta.get('cuisines', 'General'),
                    "cost": meta.get('cost', 'N/A'),
                    "rating": meta.get('rating', 'New'),
                    "address": meta.get('address', 'Location not provided'),
                    "phone": meta.get('phone', 'No contact number'),
                    "votes": meta.get('votes', '0')
                })

        # If zero matches found explicitly, fallback search to ensure grid isn't empty
        if not top_restaurants:
             fallback_res = col.query(query_texts=["best restaurants in bangalore"], n_results=8)
             if fallback_res['documents'] and len(fallback_res['documents'][0]) > 0:
                 for i, doc in enumerate(fallback_res['documents'][0]):
                     meta = fallback_res['metadatas'][0][i]
                     top_restaurants.append({
                         "name": meta.get('name', 'Unknown Restaurant'),
                         "cuisines": meta.get('cuisines', 'General'),
                         "cost": meta.get('cost', 'N/A'),
                         "rating": meta.get('rating', 'New'),
                         "address": meta.get('address', 'Location not provided'),
                         "phone": meta.get('phone', 'No contact number'),
                         "votes": meta.get('votes', '0')
                     })

        return {"status": "success", "cards": top_restaurants}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Search Error: {str(e)}"})
