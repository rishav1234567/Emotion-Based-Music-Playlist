import numpy as np # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore

# Import our modules
from . import data_preprocessing, model, playlist_generator


def load_data():
    features, labels = data_preprocessing.process_directory('data/train')
    # Reshape features for CNN: (samples, features, 1)
    features = np.expand_dims(features, axis=-1)
    # Encode labels as integers then convert to categorical
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_cat = to_categorical(labels_encoded)
    return features, labels_cat, le

def train_model():
    X, y, le = load_data()
    input_shape = (X.shape[1], 1)
    num_classes = y.shape[1]
    cnn_model = model.create_model(input_shape, num_classes)
    # Train the model
    cnn_model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)
    # Save the model for future use
    cnn_model.save('models/emotion_cnn.h5')
    return cnn_model, le

def predict_emotion(file_path, cnn_model, le):
    feature = data_preprocessing.extract_features(file_path)
    # Reshape to match input shape of the model: (1, features, 1)
    feature = np.expand_dims(feature, axis=0)
    feature = np.expand_dims(feature, axis=-1)
    pred = cnn_model.predict(feature)
    emotion = le.inverse_transform([np.argmax(pred)])
    return emotion[0]

def main():
    # Try loading a previously trained model; otherwise, train a new one.
    try:
        from tensorflow.keras.models import load_model # type: ignore
        cnn_model = load_model('models/emotion_cnn.h5')
        # To get label encoder, we load data (in a production system you would save this too)
        _, _, le = load_data()
        print("Model loaded successfully.")
    except Exception as e:
        print("Training new model...")
        cnn_model, le = train_model()
    
    # Predict the emotion of a sample audio file (ensure file exists in data/test)
    sample_file = 'data/test/sample.wav'
    predicted_emotion = predict_emotion(sample_file, cnn_model, le)
    print("Predicted emotion:", predicted_emotion)
    
    # Generate and print the playlist based on predicted emotion
    playlist = playlist_generator.generate_playlist(predicted_emotion)
    print(f"Playlist for emotion '{predicted_emotion}':")
    for song in playlist:
        print(song)

if __name__ == '__main__':
    main()
