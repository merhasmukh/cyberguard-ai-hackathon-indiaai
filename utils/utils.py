
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
loaded_model = load_model('./utils/multi_output_model.h5')
loaded_model.compile(
    loss={
        'category_output': 'sparse_categorical_crossentropy', 
        'subcategory_output': 'sparse_categorical_crossentropy'
    },
    optimizer='adam',
    metrics={
        'category_output': 'accuracy', 
        'subcategory_output': 'accuracy'
    })
# Load the tokenizer (you need to save and load the tokenizer as well)
# For this example, let's assume you have saved your tokenizer
import pickle

# Load the tokenizer
with open('./utils/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# Load the label encoders from files
with open('./utils/label_encoder_category.pickle', 'rb') as handle:
    label_encoder_category = pickle.load(handle)

with open('./utils/label_encoder_subcategory.pickle', 'rb') as handle:
    label_encoder_subcategory = pickle.load(handle)

def predict_text(input_text):
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequence, maxlen=200)  # Use the same maxlen used during training

    # Make predictions
    category_prediction, subcategory_prediction = loaded_model.predict(padded_sequence)

    # Decode the predicted classes
    category = np.argmax(category_prediction, axis=1)[0]  # Get the predicted category index
    subcategory = np.argmax(subcategory_prediction, axis=1)[0]  # Get the predicted subcategory index
    category_label = label_encoder_category.inverse_transform([category])[0]
    subcategory_label = label_encoder_subcategory.inverse_transform([subcategory])[0]
    print(category_label,subcategory_label)
    return category_label, subcategory_label
