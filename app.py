import os
from flask import Flask, render_template, request, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load product data
csv_file_path = 'amazon.csv'
data = pd.read_csv(csv_file_path)

# Data Cleaning
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')  # Convert to numeric, replace invalid values with NaN
data['rating_count'] = data['rating_count'].str.replace(',', '').astype(str)
data['rating_count'] = pd.to_numeric(data['rating_count'], errors='coerce')  # Convert to numeric, replace invalid values with NaN

# Remove rows with NaN values in important columns
data = data.dropna(subset=['rating', 'rating_count'])

# Precompute the TF-IDF matrix
data['combined_features'] = data[['Name', 'category', 'about_product']].fillna('').agg(' '.join, axis=1)
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['combined_features'])

# Helper function to get recommendations
def get_recommendations(query, num_recommendations=5):
    matching_products = data[data['Name'].str.lower().str.contains(query.lower(), na=False)]
    if matching_products.empty:
        return pd.DataFrame()  # Return empty DataFrame if no match
    
    product_index = matching_products.index[0]
    similarity_scores = cosine_similarity(tfidf_matrix[product_index], tfidf_matrix).flatten()
    similar_indices = similarity_scores.argsort()[-(num_recommendations + 1):-1][::-1]
    
    return data.iloc[similar_indices]

@app.route('/')
def home():
    # Trending products (e.g., high ratings and reviews)
    trending_products = data[(data['rating'] > 4.0) & (data['rating_count'] > 500)]
    trending_products = trending_products.sample(8) if len(trending_products) >= 8 else data.sample(8)
    random_product_image_urls = trending_products['img_link'].tolist()

    # Predefined video URL from static/videos
    video_url = url_for('static', filename='videos/v.mp4')  # Assuming the video is saved as 'v.mp4'

    return render_template(
        'index.html',
        video_url=video_url,
        trending_products=trending_products.iterrows(),
        random_product_image_urls=random_product_image_urls
    )

@app.route('/main', methods=['GET', 'POST'])
def main():
    recommendations = pd.DataFrame()
    message = None
    if request.method == 'POST':
        query = request.form.get('prod', '')
        num_recommendations = int(request.form.get('nbr', 5))  # Default to 5 recommendations
        recommendations = get_recommendations(query, num_recommendations)
        message = "Here are your recommendations" if len(recommendations) > 0 else "No recommendations found."
    
    return render_template('main.html', content_based_rec=recommendations.iterrows(), message=message)

if __name__ == '__main__':
    app.run(debug=True)
