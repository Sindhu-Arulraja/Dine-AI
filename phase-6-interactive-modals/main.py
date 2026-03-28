from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

# Load ENV variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

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

# Phase 6: Localized AI Ask Assistant
@app.get("/api/ask-restaurant")
async def ask_restaurant(name: str = Query(""), question: str = Query("")):
    if not GROQ_API_KEY:
        return JSONResponse(status_code=500, content={"error": "GROQ_API_KEY missing from .env"})

    col = get_chroma_collection()
    if not col:
         return JSONResponse(status_code=500, content={"error": "Vector Database missing."})

    try:
        # Perform specific vector search on the exact restaurant name to pull its specific reviews/metadata
        # We append name+question to hit the most relevant context vectors for this place
        search_query = f"{name} {question}"
        results = col.query(query_texts=[search_query], n_results=1)
        
        context_str = ""
        if results['documents'] and len(results['documents'][0]) > 0:
            context_str = f"Restaurant Name: {name}\nDetails: {results['documents'][0][0]}"

        client = Groq(api_key=GROQ_API_KEY)
        
        system_prompt = f"You are the official digital concierge precisely for the restaurant '{name}'. Your goal is to enthusiastically answer the user's question about the restaurant using ONLY the provided metadata context. If you don't know the answer from the context provided, apologize elegantly. Be conversational, do not sound like a robot."
        
        user_prompt = f"""
RESTAURANT CONTEXT:
{context_str if context_str else "No specific context available."}

USER QUESTION:
{question}

Answer concisely (1 short paragraph) and beautifully. Do not use generic intro statements.
"""

        async def stream_generator():
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.5,
                    max_tokens=300,
                    stream=True
                )
                
                for chunk in chat_completion:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content

            except Exception as e:
                yield f"\n\n**[ERROR]** Encountered AI streaming error: {str(e)}"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Search Error: {str(e)}"})
