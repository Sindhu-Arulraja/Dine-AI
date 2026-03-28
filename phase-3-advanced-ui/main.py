from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq

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
        return JSONResponse(status_code=500, content={"status": "error", "message": "GROQ_API_KEY not found in .env file."})
        
    col = get_chroma_collection()
    if not col:
         return JSONResponse(status_code=500, content={"status": "error", "message": "Vector Database not initialized."})

    # Semantic string query
    search_terms = []
    if cuisine: search_terms.append(f"{cuisine} food")
    if location: search_terms.append(f"in {location}")
    if event: search_terms.append(f"vibe: {event}")
    if budget: search_terms.append(f"{budget} budget")

    query_text = " ".join(search_terms) if search_terms else "Great restaurant in Bangalore"

    try:
        # Phase 3: Hallucination-proof data retrieval
        results = col.query(query_texts=[query_text], n_results=3)
        
        context_str = ""
        top_restaurants_full = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                context_str += f"Restaurant {i+1}: {doc}\n"
                
                # We extract full metadata to send to the UI for glass cards
                top_restaurants_full.append({
                    "name": meta.get('name', 'Unknown'),
                    "cuisines": meta.get('cuisines', ''),
                    "cost": meta.get('cost', ''),
                    "rating": meta.get('rating', '')
                })

        client = Groq(api_key=GROQ_API_KEY)
        
        # Phase 3 strict tone and hallucination calibration
        system_prompt = "You are 'DineAI', an elite culinary concierge for Bangalore. You MUST strictly adhere to the restaurant context provided. If no restaurants are provided or they clearly do not meet the user's explicit filter requirements, apologize elegantly and do NOT hallucinate outside restaurants."
        
        user_prompt = f"""
USER CONTEXT:
Cuisine: {cuisine or 'Any'}
Location: {location or 'Any'}
Event: {event or 'Not specified'}
Budget: {budget or 'Any'}
Party Size: {people or 'Not specified'}
Time: {time or 'Not specified'}

VERIFIED VECTOR MATCHES:
{context_str if context_str else "THERE ARE NO RESTAURANTS MATCHING THIS REQUEST. APOLOGIZE TO THE USER."}

Respond directly directly to the user in a sophisticated, friendly tone. Recommend ONE or TWO of the verified restaurants. Explain exactly why they fit the vibe based on the user's inputs. Use markdown formatting. Highlight the restaurant name in **bold**. (Keep it concise: 1-2 paragraphs max).
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
        
        return {
            "status": "success", 
            "recommendation": chat_completion.choices[0].message.content,
            "top_matches_used": top_restaurants_full
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"AI Engine Error: {str(e)}"})
