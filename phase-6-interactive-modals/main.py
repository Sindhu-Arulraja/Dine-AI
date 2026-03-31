from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import os
import json
from groq import Groq
from dotenv import load_dotenv

# Setup Absolute Base Dir for Serverless
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load ENV variables
load_dotenv(os.path.join(BASE_DIR, ".env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

app = FastAPI()

os.makedirs("static", exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

ZOMATO_PATH = os.path.join(BASE_DIR, "..", "phase-2-groq-recommender", "zomato_subset.json")
_RESTAURANT_CACHE = None

def get_restaurant_data():
    global _RESTAURANT_CACHE
    if _RESTAURANT_CACHE is None:
        try:
            with open(ZOMATO_PATH, "r", encoding="utf-8") as f:
                _RESTAURANT_CACHE = json.load(f)
        except Exception as e:
            print("Failed to load JSON:", e)
            _RESTAURANT_CACHE = []
    return _RESTAURANT_CACHE

def robust_search(query_text, n_results=8):
    data = get_restaurant_data()
    if not data:
         return {'documents': [[]], 'metadatas': [[]]}
         
    q_terms = [t.lower() for t in query_text.split() if len(t) > 2]
    
    scored = []
    for i, r in enumerate(data):
        score = 0
        name = str(r.get("name", "")).lower()
        cuisine = str(r.get("cuisines", "")).lower()
        loc = str(r.get("location", "")).lower()
        
        doc_str = f"{name} {cuisine} {loc}"
        
        for term in q_terms:
            if term in doc_str:
                score += 1
        scored.append((score, i))
        
    scored.sort(reverse=True, key=lambda x: x[0])
    top_indices = [idx for sc, idx in scored[:n_results]]
    
    docs = []
    metas = []
    for idx in top_indices:
        r = data[idx]
        name = str(r.get("name", "Unknown Restaurant"))
        cuis = str(r.get("cuisines", "General"))
        loc = str(r.get("location", ""))
        cost = str(r.get("approx_cost(for two people)", "N/A"))
        dl = str(r.get("dish_liked", ""))
        rate = str(r.get("rate", "New"))
        address = str(r.get("address", "Location not provided"))
        phone = str(r.get("phone", "No contact number"))
        votes = str(r.get("votes", "0"))
        
        docs.append(f"{name} is located in {loc}. They serve {cuis}. Dishes: {dl}")
        metas.append({
            "name": name,
            "cuisines": cuis,
            "cost": cost,
            "rating": rate,
            "address": address,
            "phone": phone,
            "votes": votes
        })
    return {'documents': [docs], 'metadatas': [metas]}

# Serve Frontend explicitly
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

@app.get("/results.html")
async def serve_results():
    return FileResponse(os.path.join(BASE_DIR, "static", "results.html"))

@app.get("/api/debug")
async def debug_endpoint():
    import sys
    data = get_restaurant_data()
    if not data:
        return {"status": "fatal", "python_version": sys.version, "message": "Failed to load JSON completely. Check logs."}
    return {"status": "ok", "python_version": sys.version, "document_count": len(data), "path": ZOMATO_PATH}

@app.get("/api/catalog-search")
async def get_catalog(
    cuisine: str = Query(""),
    location: str = Query(""),
    event: str = Query(""),
    budget: str = Query(""),
    people: str = Query(""),
    time: str = Query("")
):
    data = get_restaurant_data()
    if not data:
         return JSONResponse(status_code=500, content={"status": "error", "message": "Dataset not initialized."})

    search_terms = []
    if cuisine: search_terms.append(f"{cuisine} food")
    if location: search_terms.append(f"in {location}")
    if event: search_terms.append(f"vibe: {event}")
    if budget: search_terms.append(f"{budget} budget")

    query_text = " ".join(search_terms) if search_terms else "Popular restaurant in Bangalore"

    try:
        # Phase 5: Fetching pure 8 results for the UI grid, no LLM
        results = robust_search(query_text, 8)
        
        top_restaurants = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                
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
             fallback_res = robust_search("best restaurants in bangalore", 8)
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

    data = get_restaurant_data()
    if not data:
         return JSONResponse(status_code=500, content={"error": "Dataset missing."})

    try:
        search_query = f"{name} {question}"
        results = robust_search(search_query, 1)
        
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
