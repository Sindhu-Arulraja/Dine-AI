import json
import chromadb
from chromadb.utils import embedding_functions

def embed_restaurants():
    print("Loading restaurant data from zomato_subset.json...")
    with open('zomato_subset.json', 'r', encoding='utf-8') as f:
        restaurants = json.load(f)
        
    print(f"Initializing ChromaDB with {len(restaurants)} restaurants...")
    
    # Create persistent Chroma client
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Use default sentence transformers embedding
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="restaurants",
        embedding_function=sentence_transformer_ef
    )
    
    docs = []
    metadatas = []
    ids = []
    
    print("Formatting objects for embedding string...")
    for idx, r in enumerate(restaurants):
        name = str(r.get("name", "Unknown Restaurant"))
        cuisine = str(r.get("cuisines", ""))
        location = str(r.get("location", ""))
        cost = str(r.get("approx_cost(for two people)", ""))
        dish_liked = str(r.get("dish_liked", ""))
        rate = str(r.get("rate", ""))
        
        # Create a rich semantic profile
        doc = f"{name} is a restaurant located in {location}. They serve {cuisine}. Popular dishes include {dish_liked}. Rating: {rate}. Cost for two: {cost}."
        docs.append(doc)
        
        metadatas.append({
            "name": name,
            "location": location,
            "cuisines": cuisine,
            "cost": cost,
            "rating": rate
        })
        ids.append(str(idx))
    
    print("Embedding and adding to ChromaDB... (This will download the model the first time)")
    # We will process in batches
    batch_size = 500
    for i in range(0, len(docs), batch_size):
        collection.add(
            documents=docs[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        print(f"Added batch {int(i/batch_size) + 1}")
        
    print("Successfully built the Vector Database!")

if __name__ == "__main__":
    embed_restaurants()
