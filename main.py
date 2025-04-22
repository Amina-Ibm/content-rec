import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import json

def main(req, res):
    # Get book title from the request
    try:
        data = json.loads(req.payload)
        book_name = data.get('book_title', '')
        
        if not book_name:
            return res.json({
                "success": False,
                "message": "Book title is required"
            }, 400)
        
        # Load the similarity matrix and book list
        book_cosine_sim = load_npz("feature_matrix.npz")
        book_names = pd.Series(pd.read_csv("book_list.csv")["Name"])  # Make sure column name matches
        
        # Get recommendations
        recommendations = recommend_books(book_name, book_names, book_cosine_sim)
        
        return res.json({
            "success": True,
            "book": book_name,
            "recommendations": recommendations
        })
    
    except Exception as e:
        return res.json({
            "success": False,
            "message": str(e)
        }, 500)

def recommend_books(book_name, book_names, book_cosine_sim, top_n=5):
    """Recommend books similar to the given book using precomputed similarity scores."""
    
    # Get the index of the input book
    if book_name.lower() not in book_names.str.lower().values:
        return {"error": "Book not found in dataset!"}
    
    input_idx = book_names.str.lower()[book_names.str.lower() == book_name.lower()].index[0]
    
    # Get similarity scores for this book
    sim_scores = book_cosine_sim[input_idx].toarray().flatten()
    
    # Get top-n similar books (excluding itself)
    top_indices = np.argsort(sim_scores)[::-1][1:top_n+1]
    
    return book_names.iloc[top_indices].tolist()