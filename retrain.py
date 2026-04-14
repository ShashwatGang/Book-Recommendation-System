import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sqlite3

def run_pipeline():
    print("🚀 Starting Model Retraining Pipeline...")

    # 1. Load Historical CSV Data
    # Update these paths if your CSV files are in a different folder
    print("Loading historical CSV data...")
    books = pd.read_csv('data/Books.csv', low_memory=False)
    ratings = pd.read_csv('data/Ratings.csv')

    # 2. Load Fresh Data from SQLite Database
    print("Fetching fresh data from database...")
    conn = sqlite3.connect('library.db')
    
    # Read the new tables into Pandas DataFrames
    new_books = pd.read_sql_query("SELECT * FROM book", conn)
    new_ratings = pd.read_sql_query("SELECT * FROM rating", conn)
    conn.close()

    # Rename columns to match the original CSV structure
    new_books.rename(columns={'isbn': 'ISBN', 'title': 'Book-Title', 'author': 'Book-Author', 'image_url': 'Image-URL-M'}, inplace=True)
    new_ratings.rename(columns={'user_identifier': 'User-ID', 'isbn': 'ISBN', 'rating_score': 'Book-Rating'}, inplace=True)

    # 3. Merge Historical and Fresh Data
    print("Merging datasets...")
    # Combine the books and ratings. drop_duplicates ensures we don't have overlapping entries.
    all_books = pd.concat([books, new_books]).drop_duplicates(subset=['ISBN'])
    all_ratings = pd.concat([ratings, new_ratings])

    # Merge ratings with book titles
    ratings_with_name = all_ratings.merge(all_books, on='ISBN')

    # 4. Recreate Popularity Based System
    print("Calculating new Popular Books...")
    ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')
    
    num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
    num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
    
    avg_rating_df = ratings_with_name.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
    avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)
    
    popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
    popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)
    popular_df = popular_df.merge(all_books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]

    # 5. Recreate Collaborative Filtering System
    print("Filtering users and books for Collaborative Matrix...")
    # Users with > 200 ratings
    x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
    experienced_users = x[x].index
    filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(experienced_users)]

    # Books with > 50 ratings
    y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
    famous_books = y[y].index
    final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

    # Create the Pivot Table
    print("Generating Pivot Table and Cosine Similarity Matrix...")
    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pt.fillna(0, inplace=True)

    # Compute Math
    similarity_scores = cosine_similarity(pt)

    # 6. Export the New Models
    print("Saving updated models to disk...")
    pickle.dump(popular_df, open('popular.pkl', 'wb'))
    pickle.dump(pt, open('pt.pkl', 'wb'))
    pickle.dump(all_books, open('books.pkl', 'wb'))
    pickle.dump(similarity_scores, open('similarity_scores.pkl', 'wb'))

    print("✅ Retraining Complete! The web app will use the new data upon restart.")

if __name__ == '__main__':
    run_pipeline()