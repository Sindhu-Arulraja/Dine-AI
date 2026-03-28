from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq
import json

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
        # Phase 4 extended: Fetching 6 results instead of 3 to provide alternate places
        results = col.query(query_texts=[query_text], n_results=6)
        
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
        
        # Phase 4 prompt logic updated to handle primary and alternate places
        system_prompt = "You are 'DineAI', an elite culinary concierge for Bangalore. You MUST strictly adhere to the restaurant context provided. If no restaurants are provided or they clearly do not meet the user's explicit filter requirements, apologize elegantly and do NOT hallucinate outside restaurants. Format beautifully in markdown."
        
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

As an elite concierge, please review the verified vector matches above.
Even though there are up to 6 matches, our UI ALREADY displays all of them graphically to the user.
Therefore, your ONLY job is to select the absolute best 1 or 2 restaurants from the list, and write a highly enthusiastic, conversational recommendation explaining why they perfectly match the user's vibe and location. 
DO NOT list all the restaurants. DO NOT create an "Alternate Places" section. Keep it to 1-2 concise paragraphs. Use markdown formatting and highlight restaurant names in **bold**.
"""

        # We construct an Async Generator for SSE Streaming (Phase 4)
        async def stream_generator():
            try:
                # 1. Immediately yield the JSON metadata for the UI (Cards + Images)
                metadata_packet = json.dumps({"cards": top_restaurants_full})
                yield f"[METADATA] {metadata_packet}\n\n[DELIMITER]\n"
                
                # 2. Start token streaming from Groq
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.7,
                    max_tokens=600,
                    stream=True
                )
                
                for chunk in chat_completion:
                    if chunk.choices[0].delta.content is not None:
                        # stream the literal character chunks
                        yield chunk.choices[0].delta.content

            except Exception as e:
                yield f"\n\n**[ERROR]** Encountered AI Error: {str(e)}"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"AI Engine Error: {str(e)}"})
