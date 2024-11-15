
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import json
nltk.download('stopwords')
nltk.download('wordnet')
# Step 1: Text Cleaning and Tokenization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def predict_text(input_text):
    max_words = 200000
    max_len = 500

    # Load the saved model
    model = load_model('./utils/latest_models/third_approach/sub_category_text_classification_model.keras')

    # Load the tokenizer
    with open('./utils/latest_models/third_approach/sub_category_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load the label encoder
    with open('./utils/latest_models/third_approach/label_encoder_sub_category.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)

        # Clean and preprocess
    new_data_cleaned = [clean_text(text) for text in input_text]

    # Convert text to sequences
    new_sequences = tokenizer.texts_to_sequences(new_data_cleaned)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=max_len)  
    # Make predictions
    predictions = model.predict(new_padded_sequences)

    # Convert predictions to label indices
    predicted_indices = np.argmax(predictions, axis=1)

    # Decode indices to original category labels
    predicted_labels = label_encoder.inverse_transform(predicted_indices)

  
    print(f"Predicted Sub Category: {predicted_labels[0]}")
    subcategory_label=predicted_labels[0]

    # Load the JSON file
    with open("./utils/latest_models/third_approach/category_to_subcategory.json", "r") as f:
        category_to_subcategory = json.load(f)

    # Create a reverse mapping: sub_category -> category
    subcategory_to_category = {}
    for category, subcategories in category_to_subcategory.items():
        for subcategory in subcategories:
            subcategory_to_category[subcategory] = category

    category_label=subcategory_to_category.get(subcategory_label, "Category not found")

   

    return category_label, subcategory_label


import re
def clean_text(text):
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    # Lemmatization
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return text
