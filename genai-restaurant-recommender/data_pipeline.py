import pandas as pd
from datasets import load_dataset
import json
import os

def fetch_and_clean_data():
    print("Loading Zomato dataset from Hugging Face...")
    try:
        # Load the dataset
        dataset = load_dataset("ManikaSaini/zomato-restaurant-recommendation")
        
        # Convert the first split (usually 'train') to a pandas dataframe
        df = dataset['train'].to_pandas()
        print(f"Loaded {len(df)} rows.")
        
        # For Phase 1 testing, we'll extract a random sample of 2000 restaurants 
        # so we get a good mix of locations (Koramangala, Indiranagar, etc) and cuisines.
        df_subset = df.sample(2000, random_state=42)
        
        # Select important columns if they exist (will gracefully handle missing ones)
        # Assuming typical Zomato columns
        # Fill NA values to avoid JSON serialization errors
        df_subset.fillna("", inplace=True)
        
        data_list = df_subset.to_dict(orient='records')
        
        # Save to JSON
        output_file = 'zomato_subset.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4)
            
        print(f"Phase 1: Successfully saved 50 restaurants to {output_file}")
        return True
    except Exception as e:
        print(f"Error fetching data: {e}")
        return False

if __name__ == "__main__":
    fetch_and_clean_data()
