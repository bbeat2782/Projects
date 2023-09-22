import os
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import pickle
import util


scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # Insecure transport for local testing

# Information needed for establishing a connection
api_service_name = "youtube"
api_version = "v3"
client_secrets_file = "client/client_secret.json"
token_pickle_file = "token/token.pickle"

# Create credentials and YouTube API client using OAuth flow
flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
    client_secrets_file, scopes
)
credentials = None

# Check if token data exists in the pickle file
if os.path.exists(token_pickle_file):
    with open(token_pickle_file, 'rb') as token_file:
        credentials = pickle.load(token_file)

# If no valid credentials are available, run the OAuth flow
if not credentials or not credentials.valid:
    credentials = flow.run_local_server()

    # Store the credentials in the pickle file for reuse
    with open(token_pickle_file, 'wb') as token_file:
        pickle.dump(credentials, token_file)

youtube = googleapiclient.discovery.build(api_service_name, api_version, credentials=credentials)


# Delete songs from the workout playlist
util.delete_songs_in_playlist(youtube)
util.update_skipped()

# update selected
util.select_songs()

# Select songs randomly
workout_playlist = util.fetch_from_sql("SELECT artist, title FROM song_metainfo WHERE selected=1 ORDER BY RANDOM()")
artists = []
titles = []
for artist, title in workout_playlist:
    artists.append(artist)
    titles.append(title)

video_ids = []
for artist, title in zip(artists, titles):
    # get stored youtube video_id
    video_id = util.fetch_from_sql(
        f"SELECT video_id FROM song_metainfo WHERE artist='{artist}' AND title='{title}'"
    )[0][0]

    # if not stored, search video_id using youtube api
    if not video_id:
        video_id = util.search_video_id(title, artist, youtube)
        # caching video_id to use quota efficiently
        util.insert_into_sql(
            f"UPDATE song_metainfo SET video_id='{video_id}' WHERE artist='{artist}' AND title='{title}'"
        )
    video_ids.append(video_id)


# Add songs to the workout playlist
util.adding_song_to_playlist(video_ids, youtube)

print("Playlist updated")
