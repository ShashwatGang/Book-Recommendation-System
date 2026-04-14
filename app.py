from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import requests
import pickle
import numpy as np

# --- 1. LOAD YOUR MACHINE LEARNING DATA ---
# This must happen before your routes try to use these variables!
popular_df = pickle.load(open('popular.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))

# --- 2. INITIALIZE FLASK ---
app = Flask(__name__)

# --- 3. DATABASE CONFIGURATION ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///library.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- DATABASE MODELS ---
class Book(db.Model):
    isbn = db.Column(db.String(20), primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    author = db.Column(db.String(255))
    image_url = db.Column(db.String(500))

class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_identifier = db.Column(db.String(100), nullable=False) 
    isbn = db.Column(db.String(20), db.ForeignKey('book.isbn'), nullable=False)
    rating_score = db.Column(db.Integer, nullable=False)

# Create the database tables if they don't exist yet
with app.app_context():
    db.create_all()

# ... Your @app.route functions go below here ...

@app.route('/')
def index():
    return render_template('index.html',
                           book_name = list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input').strip() # .strip() removes accidental spaces

    # --- THE SAFETY CHECK ---
    # Check if the book actually exists in our pre-calculated pivot table
    if user_input not in pt.index:
        error_msg = f"Sorry, '{user_input}' is not in our recommendation matrix yet. Please check your spelling or try another book!"
        return render_template('recommend.html', error_message=error_msg)

    # If it does exist, proceed with the math!
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)

    return render_template('recommend.html', data=data)

@app.route('/add_book', methods=['POST'])
def add_book():
    isbn = request.form.get('isbn').strip()

    # 1. Check if we already have this book in the database
    existing_book = Book.query.get(isbn)
    if existing_book:
        return f"Book '{existing_book.title}' is already in our database!"

    # 2. Ping the Google Books API
    api_url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
    response = requests.get(api_url)
    
    if response.status_code != 200:
        return "Error contacting the Book API."
        
    data = response.json()

    # 3. Parse the JSON response
    if 'items' not in data:
        return "Invalid ISBN or book not found on Google Books."

    book_info = data['items'][0]['volumeInfo']
    
    title = book_info.get('title', 'Unknown Title')
    # Google returns authors as a list, so we join them into a single string
    authors = ", ".join(book_info.get('authors', ['Unknown Author'])) 
    image_url = book_info.get('imageLinks', {}).get('thumbnail', '')

    # 4. Save to Database
    new_book = Book(isbn=isbn, title=title, author=authors, image_url=image_url)
    db.session.add(new_book)
    db.session.commit()

    return redirect(url_for('index')) # Or redirect to a success page

@app.route('/rate_book', methods=['POST'])
def rate_book():
    user_name = request.form.get('user_name').strip()
    isbn = request.form.get('isbn').strip()
    rating_score = request.form.get('rating_score')

    # Basic validation
    if not user_name or not isbn or not rating_score:
        return "Please fill out all fields."

    try:
        rating_score = int(rating_score)
        if rating_score < 1 or rating_score > 10:
            return "Rating must be between 1 and 10."
    except ValueError:
        return "Invalid rating format."

    # Ensure the book exists before rating it
    book = Book.query.get(isbn)
    if not book:
        return "You must add this book via ISBN before rating it!"

    # Save the rating
    new_rating = Rating(user_identifier=user_name, isbn=isbn, rating_score=rating_score)
    db.session.add(new_rating)
    db.session.commit()

    return f"Thanks for rating {book.title}, {user_name}!"


if __name__ == '__main__':
    app.run(debug=True)