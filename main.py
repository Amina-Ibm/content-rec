import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from appwrite.client import Client
from appwrite.services.storage import Storage
import traceback

# Appwrite configuration
PROJECT_ID = "67f4f5e00020850f31e6"
API_KEY = "standard_06b52af3aaa585eb01cda0b5378eadf69d59cfacd0222a5e3d18a988fb605d59f881df7e4c14f4efffd42414bc01ad6e8b966cdf9d954c2ac3b5704d9eeaf6cee34b2bbf39eca83b2a6df4ce5d0d68de19998f42c0175e3d951fe66c72dd6141731f6f2885062258499f3e7b94ad4bd01f0272824b3d9b810907fd25daac179c"
BUCKET_ID = "6807c93d00133c40f257"
CHUNK_FILES = ["6807e02f000be5a64b1e", "6807dd7000090ac0ed94", "6807ddba00012e228853", "6807e04f000eb187d7d7", "6807e0750014f993886d", "6807de66003b1af04ae6"]

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
    
    if not dataframes:
        raise Exception("Failed to load any data chunks")
        
    return pd.concat(dataframes, ignore_index=True)

def recommend_books(book_name, book_df, book_cosine_sim, top_n=5):
    # Make sure we're working with dataframes and series properly
    if not isinstance(book_df, pd.DataFrame):
        raise TypeError("book_df must be a pandas DataFrame")
    
    if "Name" not in book_df.columns:
        raise ValueError("book_df must contain a 'Name' column")
    
    # Get book names as a Series
    book_names = book_df["Name"]
    
    # Convert input book name to lowercase
    book_name_lower = book_name.lower()
    
    # Create mask for matching books (case-insensitive)
    matching_mask = book_names.str.lower() == book_name_lower
    
    # Check if book exists in dataset
    if not matching_mask.any():
        return {"error": "Book not found in dataset!"}
    
    # Find the index of the book
    input_idx = matching_mask.idxmax()  # Get first matching index
    
    if input_idx is None or np.isnan(input_idx):
        return {"error": "Book index could not be determined"}
    
    # Convert to integer to ensure proper indexing
    input_idx = int(input_idx)
    
    # Get similarity scores
    try:
        # Handle both sparse and dense matrices
        if hasattr(book_cosine_sim, 'toarray'):
            row = book_cosine_sim[input_idx]
            if hasattr(row, 'toarray'):
                sim_scores = row.toarray().flatten()
            else:
                sim_scores = np.array(row).flatten()
        else:
            sim_scores = np.array(book_cosine_sim[input_idx]).flatten()
        
        # Get top similar books excluding the input book
        top_indices = np.argsort(sim_scores)[::-1][1:top_n+1]
        top_indices = [int(i) for i in top_indices]  # Ensure integer indices
        
        return book_names.iloc[top_indices].tolist()
    except Exception as e:
        stack_trace = traceback.format_exc()
        error_msg = f"Error calculating recommendations: {str(e)}\n{stack_trace}"
        print(error_msg)
        return {"error": error_msg}

# Main cloud function entry point
def main(context):
    try:
        context.log("Parsing payload...")
        data = json.loads(context.req.body_raw)

        book_name = data.get('book_title', '')
        context.log(f"Book title received: {book_name}")

        if not book_name:
            return {
                "statusCode": 400,
                "body": json.dumps({"success": False, "message": "Book title is required"})
            }

        context.log("Downloading and merging chunks...")
        book_list_df = download_and_merge_chunks(CHUNK_FILES)
        context.log("Chunk files loaded.")

        context.log("Current working directory: " + os.getcwd())
        context.log("Files in directory: " + str(os.listdir()))


        context.log("Loading similarity matrix...")
        book_cosine_sim = load_npz("function/feature_matrix.npz")
        context.log("Cosine similarity matrix loaded.")

        context.log("Generating recommendations...")
        recommendations = recommend_books(book_name, book_list_df, book_cosine_sim)
        context.log(f"Recommendations: {recommendations}")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "success": True,
                "book": book_name,
                "recommendations": recommendations
            })
        }

    except Exception as e:
        context.error(f"Internal error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"success": False, "message": str(e)})
        }

