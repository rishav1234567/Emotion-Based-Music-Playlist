# 🎭 Emotion Based Music Playlist Generator

An AI-powered web application that detects human emotions from a webcam feed and automatically generates a music playlist using Spotify API.

---

## 🚀 Live Demo pending 



---

## 📌 Features

- 🎥 Real-time face detection using webcam
- 🧠 Deep Learning (CNN) based emotion recognition
- 🎵 Spotify API integration for music recommendation
- 🌐 Flask-based web application
- 😊 Emotion-based playlist generation (Happy, Sad, Angry, etc.)
- 📊 CSV-based song dataset mapping emotions to songs

---

## 🧠 Emotion Classes

- Angry 😡  
- Disgust 🤢  
- Fear 😨  
- Happy 😄  
- Sad 😢  
- Surprise 😲  
- Neutral 😐  

---

## 🏗️ Tech Stack

- Python 🐍
- Flask 🌐
- OpenCV 🎥
- TensorFlow / Keras 🧠
- Spotipy (Spotify API) 🎵
- HTML, CSS, JavaScript 🎨
- Pandas & NumPy 📊

---

## 📁 Project Structure

EmotionBasedMusicPlaylist/
│
├── app.py
├── camera.py
├── train.py
├── test_webcam.py
├── utils.py
├── model.h5
├── haarcascade_frontalface_default.xml
│
├── songs/
│ ├── angry.csv
│ ├── happy.csv
│ ├── sad.csv
│ ├── neutral.csv
│ ├── surprised.csv
│ ├── fearful.csv
│ └── disgusted.csv
│
├── templates/
│ └── index.html
│
├── static/
│
├── requirements.txt
├── README.md
└── LICENSE


---

## ⚙️ Installation & Setup

### 1. Clone repository
```bash
git clone https://github.com/your-username/EmotionBasedMusicPlaylist.git
cd EmotionBasedMusicPlaylist
2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
3. Install dependencies
pip install -r requirements.txt
4. Setup environment variables

Create a .env file in root directory:

SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:5000/callback
5. Run the application
python app.py
🔁 How It Works
User opens web app
Webcam captures face in real-time
CNN model predicts emotion
Emotion is mapped to song dataset
Spotify API fetches recommended songs
Playlist is displayed to user
🎯 API Used
Spotify Web API (via Spotipy)
📦 Requirements

Install all dependencies:

flask
opencv-python
numpy
tensorflow
keras
spotipy
python-dotenv
pandas
scikit-learn
matplotlib
pillow
⚠️ Important Notes
Do NOT upload .env file to GitHub
Do NOT upload venv/ folder
Ensure Spotify Redirect URI matches exactly in dashboard and code
Model file model.h5 must exist before running prediction
👨‍💻 Author

Rishav Shakya
Python Developer | AI/ML Enthusiast

📜 License

This project is licensed under the MIT License.


---

## 🚀 Done

---

If you want next upgrade, I can help you:
- 🔥 :contentReference[oaicite:0]{index=0}
- 🚀 :contentReference[oaicite:1]{index=1}
- 🎯 :contentReference[oaicite:2]{index=2}
- 🎥 :contentReference[oaicite:3]{index=3}


