import pandas as pd # type: ignore

def generate_playlist(emotion, songs_db='data/songs.csv'):
    df = pd.read_csv(songs_db)
    # Filter songs matching the predicted emotion
    playlist = df[df['emotion'] == emotion]
    return playlist[['song', 'artist']].to_dict('records')

if __name__ == '__main__':
    emotion = 'happy'
    playlist = generate_playlist(emotion)
    for song in playlist:
        print(song)
