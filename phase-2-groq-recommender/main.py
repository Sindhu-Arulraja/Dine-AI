from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import json
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq

# Load ENV variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

app = FastAPI()

# Make static directory exist
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Connect to ChromaDB lazily so the server can start even if DB is building
chroma_client = None
collection = None

def get_chroma_collection():
    global chroma_client, collection
    if collection is not None:
        return collection
    if os.path.exists("./chroma_db"):
        try:
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            # We don't strictly need to pass the sentence transformer back in for queries, because chromadb remembers it implicitly if installed, but doing so prevents errors
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

@app.get("/api/ai-recommendation")
async def get_ai_recommendation(
    cuisine: str = Query(""),
    location: str = Query(""),
    event: str = Query(""),
    budget: str = Query(""),
    people: str = Query(""),
    time: str = Query("")
):
    if not GROQ_API_KEY:
        return JSONResponse(status_code=500, content={"status": "error", "message": "GROQ_API_KEY not found in .env file. Please add it to test the LLM."})
        
    col = get_chroma_collection()
    if not col:
         return JSONResponse(status_code=500, content={"status": "error", "message": "Vector Database not initialized. Please run embed_data.py first."})

    # Formulate vector search query semantic string
    search_terms = []
    if cuisine: search_terms.append(f"{cuisine} food")
    if location: search_terms.append(f"in {location}")
    if event: search_terms.append(f"vibe: {event}")
    if budget: search_terms.append(f"{budget} budget")

    query_text = " ".join(search_terms) if search_terms else "Great restaurant in Bangalore"

    try:
        # Semantic Search in ChromaDB
        results = col.query(
            query_texts=[query_text],
            n_results=3
        )
        
        context_str = ""
        top_restaurants = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                context_str += f"Restaurant {i+1}: {doc}\n"
                top_restaurants.append(meta['name'])
        else:
             return {"status": "success", "recommendation": "I'm sorry, no database matches were found."}

        # Setup Groq RAG call
        client = Groq(api_key=GROQ_API_KEY)
        system_prompt = "You are 'DineAI', a premium, highly knowledgeable culinary concierge for Bangalore."
        
        user_prompt = f"""
The user wants a restaurant recommendation with these details:
Cuisine: {cuisine or 'Any'}
Location: {location or 'Any'}
Event: {event or 'Not specified'}
Budget: {budget or 'Any'}
Party Size: {people or 'Not specified'}
Time: {time or 'Not specified'}

Based on a semantic vector search of our dataset, here are the top 3 contextual matches:
{context_str}

Please respond directly to the user in a sophisticated, friendly tone. Recommend ONE or TWO of these exact restaurants. Explain *why* they fit the requested mood/vibe based on their inputs. Do NOT invent outside restaurants. Keep it concise (1-2 paragraphs). Return output in clean markdown.
"""
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=600
        )
        
        llm_response = chat_completion.choices[0].message.content
        
        return {
            "status": "success", 
            "recommendation": llm_response,
            "top_matches_used": top_restaurants
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"AI Engine Error: {str(e)}"})
