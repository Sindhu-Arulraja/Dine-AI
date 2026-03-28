from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import json

app = FastAPI()

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

# Mount the static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse('static/index.html')

@app.get("/api/ping")
async def ping():
    return {"status": "ok", "message": "Python API is live."}

@app.get("/api/search")
async def search_restaurants(location: str = Query(""), cuisine: str = Query("")):
    if os.path.exists("zomato_subset.json"):
        with open("zomato_subset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            results = []
            for r in data:
                loc_match = location.lower() in str(r.get("location", "")).lower() if location else True
                cuis_match = cuisine.lower() in str(r.get("cuisines", "")).lower() if cuisine else True
                if loc_match and cuis_match:
                    results.append(r)
            return {"status": "success", "count": len(results), "results": results}
    return {"status": "error", "message": "Dataset not found. Please run data_pipeline.py"}

@app.get("/api/data-status")
async def data_status():
    if os.path.exists("zomato_subset.json"):
        with open("zomato_subset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return {"status": "data_ready", "count": len(data), "sample": data[0] if data else None}
    return {"status": "data_missing", "message": "Dataset not found. Please run data_pipeline.py"}
