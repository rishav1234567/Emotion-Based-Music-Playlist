

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from camera import gen_frames, music_rec
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
print("Looking for .env file at:", env_path)
if not os.path.exists(env_path):
    raise FileNotFoundError(f".env file not found at {env_path}")
load_dotenv(dotenv_path=env_path)

app = Flask(__name__)

# Spotify OAuth setup
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
SPOTIFY_REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:5000/callback')
scope = "user-read-playback-state,user-modify-playback-state"

print("SPOTIFY_CLIENT_ID:", SPOTIFY_CLIENT_ID)
print("SPOTIFY_CLIENT_SECRET:", SPOTIFY_CLIENT_SECRET)
print("SPOTIFY_REDIRECT_URI:", SPOTIFY_REDIRECT_URI)

if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    raise ValueError("SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET is not set. Check .env file.")

sp_oauth = SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope=scope
)

def play_track(track_uri, access_token):
    """Play a track using the provided track URI and access token."""
    sp_play = spotipy.Spotify(auth=access_token)
    try:
        sp_play.start_playback(uris=[track_uri])
        return {"status": "success", "message": f"Playing track: {track_uri}"}
    except spotipy.SpotifyException as e:
        return {"status": "error", "message": str(e)}

headings = ("Name", "Album", "Artist", "Play")
try:
    df1 = music_rec()
    if 'URI' not in df1.columns:
        print("Warning: Initial music_rec() missing URI column. Adding empty URI column.")
        df1['URI'] = ''
    df1 = df1.head(15)
except Exception as e:
    print(f"Error in music_rec(): {str(e)}")
    df1 = pd.DataFrame(columns=['Name', 'Album', 'Artist', 'URI'])

@app.route('/')
def index():
    token_info = sp_oauth.get_cached_token()
    access_token = token_info['access_token'] if token_info else None
    return render_template('index.html', headings=headings, data=df1, access_token=access_token)

@app.route('/login')
def login():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    try:
        code = request.args.get('code')
        token_info = sp_oauth.get_access_token(code)
        return redirect(url_for('index'))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/t')
def gen_table():
    return df1.to_json(orient='records')

@app.route('/play', methods=['POST'])
def play():
    try:
        data = request.json
        track_uri = data.get('track_uri')
        access_token = data.get('access_token')
        if not track_uri or not access_token:
            return jsonify({"status": "error", "message": "Missing track URI or access token"}), 400
        result = play_track(track_uri, access_token)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)