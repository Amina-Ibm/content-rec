import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from appwrite.client import Client
from appwrite.services.storage import Storage

# Appwrite configuration
PROJECT_ID = "67f4f5e00020850f31e6"
API_KEY = "standard_06b52af3aaa585eb01cda0b5378eadf69d59cfacd0222a5e3d18a988fb605d59f881df7e4c14f4efffd42414bc01ad6e8b966cdf9d954c2ac3b5704d9eeaf6cee34b2bbf39eca83b2a6df4ce5d0d68de19998f42c0175e3d951fe66c72dd6141731f6f2885062258499f3e7b94ad4bd01f0272824b3d9b810907fd25daac179c"
BUCKET_ID = "6807c93d00133c40f257"
CHUNK_FILES = ["chunk_1.csv", "chunk_2.csv", "chunk_3.csv", "chunk_4.csv", "chunk_5.csv", "chunk_6.csv"]

# Initialize Appwrite client
client = Client()
client.set_endpoint("https://cloud.appwrite.io/v1")
client.set_project(PROJECT_ID)
client.set_key(API_KEY)

storage = Storage(client)

# Download and merge CSV chunks
def download_and_merge_chunks(chunk_files):
    dataframes = []
    for chunk in chunk_files:
        try:
            file = storage.get_file_download(BUCKET_ID, chunk)
            with open(chunk, "wb") as f:
                f.write(file)
            df = pd.read_csv(chunk)
            dataframes.append(df)
            os.remove(chunk)  # Clean up after merging
        except Exception as e:
            print(f"Error downloading {chunk}: {e}")
    return pd.concat(dataframes, ignore_index=True)

def recommend_books(book_name, book_names, book_cosine_sim, top_n=5):
    if book_name.lower() not in book_names.str.lower().values:
        return {"error": "Book not found in dataset!"}
    
    input_idx = book_names.str.lower()[book_names.str.lower() == book_name.lower()].index[0]
    sim_scores = book_cosine_sim[input_idx].toarray().flatten()
    top_indices = np.argsort(sim_scores)[::-1][1:top_n+1]
    
    return book_names.iloc[top_indices].tolist()

# Main cloud function entry point
def main(req, res):
    try:
        data = json.loads(req.payload)
        book_name = data.get('book_title', '')

        if not book_name:
            return res.json({"success": False, "message": "Book title is required"}, 400)

        # Merge all book list chunks
        book_list_df = download_and_merge_chunks(CHUNK_FILES)
        book_names = pd.Series(book_list_df["Name"])

        # Load precomputed cosine similarity matrix
        book_cosine_sim = load_npz("feature_matrix.npz")

        recommendations = recommend_books(book_name, book_names, book_cosine_sim)

        return res.json({
            "success": True,
            "book": book_name,
            "recommendations": recommendations
        })

    except Exception as e:
        return res.json({"success": False, "message": str(e)}, 500)
