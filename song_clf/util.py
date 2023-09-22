import sqlite3
import cred
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from googleapiclient.errors import HttpError

# for adding new songs
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import time
import scipy
from scipy.signal import argrelextrema


SQLDBPATH = 'data/songs.db'


def insert_into_sql(statement, statement2=None, db_path=SQLDBPATH):
    """
    Insert or update sql database based on the statement parameter

    Parameters
    ----------
    statement : str
        The first statement to execute
    statement2 : str
        The second statement to execute
    db_path : str
        Path to sql database that you want to execute the statements

    Returns
    -------
    None
    """
    
    try:
        sqliteConnection = sqlite3.connect(db_path)
        cursor = sqliteConnection.cursor()
        cursor.execute(statement)
        if statement2 != None:
            cursor.execute(statement2)
        sqliteConnection.commit()
        cursor.close()
    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)


def fetch_from_sql(statement, db_path=SQLDBPATH):
    """
    Fetch from sql database based on the statement parameter

    Parameters
    ----------
    statement : str
        The statement to execute
    db_path : str
        Path to sql database that you want to execute the statements

    Returns
    -------
    fetch_items : list
        A list of elements that match the statement from sql database
    """

    try:
        sqliteConnection = sqlite3.connect(db_path)
        cursor = sqliteConnection.cursor()
        cursor.execute(statement)
        fetch_items = cursor.fetchall()
        cursor.close()

        return fetch_items
    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)


def delete_songs_in_playlist(youtube, playlist_id=cred.youtube_workout_PLAYLISTID):
    """
    Deletes all songs in a playlist and returns a list of video_ids I listened to
    """

    # Get all items in the playlist
    video_ids, playlist_item_ids = [], []
    next_page_token = None
    while True:
        request = youtube.playlistItems().list(
            part='snippet',
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()
        video_ids_sub, playlist_item_ids_sub = map(
            list, zip(*[(r['snippet']['resourceId']['videoId'], r['id']) for r in response.get('items')])
        )
        video_ids.extend(video_ids_sub)
        playlist_item_ids.extend(playlist_item_ids_sub)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    # Delete each item from the playlist
    for video_id, item_id in zip(video_ids, playlist_item_ids):
        request = youtube.playlistItems().delete(
            id=item_id
        )
        try:
            response = request.execute()
            deselect_selected_songs(video_id)
        except HttpError as e:
            if e.resp.status == 409 and 'SERVICE_UNAVAILABLE' in str(e):
                time.sleep(3)
                response = request.execute()
                deselect_selected_songs(video_id)
            else:
                raise


def deselect_selected_songs(video_id):
    insert_into_sql(
        "UPDATE song_metainfo SET selected=0 "
        f"WHERE video_id = '{video_id}'"
    )
    print("Playlist item deleted:", video_id)


def search_video_id(song_title, song_artist, youtube):
    "Returns video id"
    search_query = f'{song_title} {song_artist} lyrics'
    max_results = 1

    request = youtube.search().list(
        part="id",
        q=search_query,
        type="video",
        maxResults=max_results
    )
    response = request.execute()

    # Extract video IDs from the search results
    video_id = [item["id"]["videoId"] for item in response.get("items")][0]

    return video_id


def adding_song_to_playlist(video_ids, youtube, playlist_id=cred.youtube_workout_PLAYLISTID):
    "With video_ids, add songs to the workout playlist"
    for video_id in video_ids:
        request = youtube.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": video_id
                    }
                }
            }
        )
        try:
            response = request.execute()
            print("Video added to playlist:", response['snippet']['title'])
        except HttpError:
            print(f"{video_id} didn't work")


def update_skipped():
    skipped_songs = fetch_from_sql('SELECT video_id FROM song_metainfo WHERE selected=1')
    for row in skipped_songs:
        insert_into_sql(
            "UPDATE song_metainfo SET skip=1, selected=0 "
            f"WHERE video_id = '{row[0]}'"
        )


def cosine_similarity_features(df, timbre, vectors):
    # contains which class is the most similar
    cosine_similarity_result = []
    # contains raw similarity value for all classes
    cosine_similarity_raw_result = []

    # Calculate how much each song mean timbre values are similar to each class
    for _, row in df[timbre].iterrows():
        result = []
        for vector in vectors:
            result.append(cosine_similarity([row.values], [vector])[0][0])
        cosine_similarity_raw_result.append(result)
        cosine_similarity_result.append(np.argmax(result))

    # Combine results into a dataframe
    cosine_result = pd.concat(
        [pd.DataFrame(cosine_similarity_raw_result), pd.DataFrame(cosine_similarity_result)],
        axis=1
    )
    cosine_result.columns = ['vec1', 'vec2', 'vec3', 'vec_argmax']
    cosine_result['vec_argmax'] = cosine_result['vec_argmax'].astype('category')
    cosine_result.index = df.index

    return pd.concat([df, cosine_result], axis=1)


def process(df, timbre, vectors, categorical_cols):
    "Add cosine similarity features and apply one hot encoding"
    df = cosine_similarity_features(df, timbre, vectors)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df = df.set_index('id')
    return df


def split_to_train_test(df, unlabeled_df=None):
    # Split df into train/test
    X = df.drop(columns=['result'])
    y = df['result']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        stratify=y,
                                                        shuffle=True,
                                                        test_size=0.2,
                                                        random_state=42)

    # Calculating mean timbre vector for each class in the training set
    vectors, timbre = [], [f't{i}' for i in range(1, 13)]
    for i in range(3):
        vectors.append(X_train.loc[y_train[y_train == i].index][timbre].mean().values)
    X_train = process(
        X_train, timbre, vectors, ['key', 'mode', 'vec_argmax', 'time_signature']
    )
    y_train.index = X_train.index

    if unlabeled_df is None:
        X_test = process(
            X_test, timbre, vectors, ['key', 'mode', 'vec_argmax', 'time_signature']
        )
        X = process(X, timbre, vectors, ['key', 'mode', 'vec_argmax', 'time_signature'])

        y_test.index = X_test.index
        y.index = X.index

        return X_train, y_train, X_test, y_test, X, y
    else:
        unlabeled_y = unlabeled_df['result']
        unlabeled_X = unlabeled_df.drop(columns=['result'])
        unlabeled_X = process(
            unlabeled_X, timbre, vectors, ['key', 'mode', 'vec_argmax', 'time_signature']
        )
        col_diff = set(X_train.columns).difference(set(unlabeled_X.columns))
        for col in col_diff:
            unlabeled_X[col] = False
        unlabeled_X = unlabeled_X.reindex(columns=X_train.columns)

        unlabeled_y.index = unlabeled_X.index

        return X_train, y_train, unlabeled_X, unlabeled_y


def reset_selected():
    insert_into_sql(
        "UPDATE song_metainfo SET selected=0"
    )


def update_selected(ids):
    reset_selected()
    all_ids = "','".join(ids)
    insert_into_sql(
        "UPDATE song_metainfo SET selected=1 "
        f"WHERE id in ('{all_ids}')"
    )


def select_songs(num_songs=50):
    conn = sqlite3.connect('data/songs.db')
    train = pd.read_sql('SELECT * FROM train', conn)
    X_train, y_train, _, _, X, _ = split_to_train_test(train)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X)

    # Include components up to 95% variability
    pca = PCA(n_components=0.95)
    pca.fit(X_train_scaled)

    X_train_pca = pca.transform(X_train_scaled)
    X_pca = pca.transform(X_scaled)

    lr = LogisticRegression(solver='newton-cg', class_weight='balanced', max_iter=1000)
    lr.fit(X_train_pca, y_train)

    pred = lr.predict(X_pca)
    output = pd.DataFrame(index=X.index, data=pred).rename(columns={0: 'result'})
    workout_output = output[output['result'] == 2]
    skip_songs = [s[0] for s in fetch_from_sql("SELECT id FROM song_metainfo WHERE skip=1")]
    workout_output = workout_output.drop(
        set(skip_songs).intersection(workout_output.index)
    )

    update_selected(workout_output.sample(num_songs).index)


# functions for creating a dataset

def get_descriptive_statistics(data):
    "Returns mean, std, median, skewness, kurtosis, and iqr"

    mean, std = np.mean(data), np.std(data)
    median = np.median(data)
    skewness = scipy.stats.skew(data)
    kurtosis = scipy.stats.kurtosis(data, fisher=True, bias=True)
    iqr = np.subtract(*np.percentile(data, [75, 25]))

    return [mean, std, median, skewness, kurtosis, iqr]


def count_local_max(arr):
    "Returns number of local max in arr"
    return len(argrelextrema(np.array(arr), np.greater)[0])


def pitches_analysis(audio_info):
    "Returns engineered features from pitches"

    pitches = [info['pitches'] for info in audio_info]
    pitches_df = pd.DataFrame(pitches)
    mean = pitches_df.mean().values

    max_pitch = pitches_df.apply(lambda row: row.idxmax(), axis=1)
    pitch_abs_diff = max_pitch.diff().iloc[1:].abs()
    pitch_abs_diff_mean = pitch_abs_diff.mean()
    pitch_abs_diff_std = pitch_abs_diff.std()

    pitches_sum = pitches_df.sum(axis=1)
    pitches_sum_mean = pitches_sum.mean()
    pitches_sum_std = pitches_sum.std()

    result = [
        list(mean),
        [pitch_abs_diff_mean, pitch_abs_diff_std, pitches_sum_mean, pitches_sum_std]
    ]
    return [element for sublist in result if isinstance(sublist, list) for element in sublist]


def timbre_analysis(audio_info):
    "Returns timbre engineered features"

    result_df = pd.DataFrame()

    def process_timbre(category, df, result_df):
        stats = df.apply(get_descriptive_statistics).transpose()
        stats = stats.rename(columns={0: f'{category}_mean', 1: f'{category}_std',
                                      2: f'{category}_median', 3: f'{category}_skewness',
                                      4: f'{category}_kurtosis', 5: f'{category}_iqr'})
        result_df = pd.concat([result_df, stats], axis=1)
        return result_df

    durations = [d['duration'] for d in audio_info]
    timbre = pd.DataFrame([d['timbre'] for d in audio_info])
    timbre_diff = timbre.diff()[1:]
    timbre_slope = timbre_diff.apply(lambda col: col / durations[:-1])
    result_df = process_timbre("default", timbre, result_df)
    result_df = process_timbre("diff", timbre_diff, result_df)
    result_df = process_timbre("slope", timbre_slope, result_df)

    return [element for sublist in result_df.to_numpy() for element in sublist]
    

def loudness_analysis(audio_info, results):
    "Returns loudness engineered features"

    variables = [
        'duration', 'loudness_max', 'loudness_start', 'loudness_max_time', 'loudness_end'
    ]
    for variable in variables:
        data = [info[variable] for info in audio_info]
        results[variable] = get_descriptive_statistics(data)

    loudness_start = [s['loudness_start'] for s in audio_info]
    loudness_max = [s['loudness_max'] for s in audio_info]
    loudness_gap = np.subtract(loudness_max, loudness_start)

    results['loudness_gap'] = [loudness_gap.mean(),
                               loudness_gap.sum(),
                               loudness_gap.std(),
                               len(loudness_gap),
                               count_local_max(loudness_gap)]
    return results


def min_max_timbre_analysis(audio_info):
    "Count the number of times each timbre is max/min in each segment"

    scaler = MinMaxScaler()
    # The first value represents the average loudness.
    # Thus, I didn't include it when evaluating relative order
    scaled_timbre = scaler.fit_transform(pd.DataFrame([s['timbre'][1:] for s in audio_info]))
    timbre_min = pd.Series([np.argmin(s) for s in scaled_timbre]).value_counts().sort_index()
    timbre_min = timbre_min.reindex(range(11), fill_value=0).values
    timbre_max = pd.Series([np.argmax(s) for s in scaled_timbre]).value_counts().sort_index()
    timbre_max = timbre_max.reindex(range(11), fill_value=0).values

    return np.ravel([timbre_min, timbre_max])


def segments_feature_engineering(audio_info):
    "Applying feature engineering to segments in audio analysis"

    results = {}
    results = loudness_analysis(audio_info, results)
    results['pitches'] = pitches_analysis(audio_info)
    results['timbre'] = timbre_analysis(audio_info)
    results['timbre_min_max'] = min_max_timbre_analysis(audio_info)

    return results


def remove_quotations(word):
    word = word.replace("'", '').replace('"', '')
    return word


def create_audio_features_and_metainfo_dataset(sp, public_playlists):
    all_songs = []
    for _, playlist_id in public_playlists.items():
        results = sp.playlist_items(playlist_id)
        songs = results['items']
        all_songs = all_songs + songs

    dfs, tracks, track_ids, track_artist_names, track_titles = [], [], [], [], []
    for item in all_songs:
        track_id = item['track']['id']
        artists = item['track']['album']['artists']
        artists_name = remove_quotations([artist['name'] for artist in artists][0])
        track_title = remove_quotations(item['track']['name'])

        # append only new track
        if not fetch_from_sql(f"SELECT * FROM song_metainfo WHERE id = '{track_id}'"):
            track_ids.append(track_id)
            track_artist_names.append(artists_name)
            track_titles.append(track_title)

    # Getting track audio features
    chunk_size = 100  # specifying chunk_size because the api accepts maximum 100 tracks
    track_id_sub = [track_ids[i:i+chunk_size] for i in range(0, len(track_ids), chunk_size)]
    track_id_audio_features = []
    for sublist in track_id_sub:
        track_id_audio_features = track_id_audio_features + sp.audio_features(sublist)
    track_id_audio_features = [features for features in track_id_audio_features if features is not None]
    dfs.append(pd.DataFrame(track_id_audio_features))
    tracks.append(pd.DataFrame({'id': track_ids,
                                'artist': track_artist_names,
                                'title': track_titles}))

    song_df = pd.concat(dfs).drop_duplicates()
    song_metainfo_df = pd.concat(tracks).drop_duplicates()
    song_metainfo_df = song_metainfo_df[song_metainfo_df['id'].isin(song_df['id'])]

    return song_df, song_metainfo_df


def retrieve_audio_analysis(sp, song_df):
    # Check for new songs that have been added to the playlists
    with open("data/all_audio_analysis.json", 'r') as json_file:
        all_audio_analysis = json.load(json_file)

    collected_tracks = all_audio_analysis.keys()
    new_songs = song_df[~song_df['id'].apply(lambda id: id in collected_tracks)]
    print(f'Number of new songs: {len(new_songs)}')

    # Retrieve audio_analysis data from Spotify for the new songs and updates the local file
    new_audio_analysis = {}
    for song_id in new_songs['id']:
        try:
            audio_analysis = sp.audio_analysis(song_id)
            new_audio_analysis[song_id] = audio_analysis
            # To avoid exceeding rate limit
            time.sleep(5)
        except Exception as e:
            print(f"{song_id} didn't work")
            print(e)

    all_audio_analysis.update(new_audio_analysis)
    with open("data/all_audio_analysis.json", "w") as outfile:
        json.dump(all_audio_analysis, outfile)

    return all_audio_analysis


def feature_engineering_dataset(all_audio_analysis):
    # Create a dataframe of engineered features
    audio_analysis_dict, timbre_mean = {}, []
    for audio_id, audio_analysis in all_audio_analysis.items():
        # segments feature engineering
        segments = audio_analysis['segments']
        desc_stat = segments_feature_engineering(segments)
        features = [element for sublist in desc_stat.values() for element in sublist]
        audio_analysis_dict[audio_id] = features

        # timbre mean for cosine similarity feature later
        timbre_mean.append(np.mean([s['timbre'] for s in segments], axis=0))

    # Combining the two
    fe_df = pd.DataFrame(audio_analysis_dict).transpose()
    timbre_vector = pd.DataFrame(timbre_mean)
    timbre_vector.columns = [
        't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12'
    ]
    timbre_vector.index = fe_df.index
    fe_df = fe_df.merge(timbre_vector, left_index=True, right_index=True)

    return fe_df


def create_train_dataset(song_df, fe_df):
    # combined track audio features with engineered features from track audio analysis
    train = song_df.merge(fe_df, how='inner', left_on='id', right_index=True)
    train = train.drop(columns=['uri', 'track_href', 'analysis_url', 'type'])
    train = train.set_index('id')
    train = train.reset_index()
    train['result'] = -1

    return train


def append_data_to_db(conn, train, song_metainfo_df):
    train.to_sql('train', conn, if_exists='append', index=False)
    song_metainfo_df.to_sql('song_metainfo', conn, if_exists='append', index=False)


def fetch_labeled_and_new_dataset(conn):
    new_songs = pd.DataFrame(fetch_from_sql('SELECT * FROM train WHERE result = -1'))
    labeled_songs = pd.DataFrame(fetch_from_sql('SELECT * FROM train WHERE result != -1'))
    table_info = conn.execute("PRAGMA table_info(train)").fetchall()
    cols = [info[1] for info in table_info]
    new_songs.columns = cols
    labeled_songs.columns = cols
    X_train, y_train, X_new, y_new = split_to_train_test(labeled_songs, unlabeled_df=new_songs)

    return X_train, y_train, X_new, y_new


def classify_new_songs(X_train, y_train, X_new, y_new):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_new_scaled = scaler.fit_transform(X_new)

    # Include components up to 95% variability
    pca = PCA(n_components=0.95)
    pca.fit(X_train_scaled)

    X_train_pca = pca.transform(X_train_scaled)
    X_new_pca = pca.transform(X_new_scaled)

    lr = LogisticRegression(solver='newton-cg', class_weight='balanced', max_iter=1000)
    lr.fit(X_train_pca, y_train)

    pred_new = lr.predict(X_new_pca)

    for spotify_id, pred_label in zip(y_new.index, pred_new):
        insert_into_sql(
            f"UPDATE train SET result='{pred_label}' WHERE id='{spotify_id}'"
        )


def adding_new_songs(public_playlists):
    "Add new songs from the public_playlists"

    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=cred.client_ID,
            client_secret=cred.client_SECRET,
            redirect_uri=cred.redirect_URI
        )
    )

    song_df, song_metainfo_df = create_audio_features_and_metainfo_dataset(
        sp, public_playlists
    )
    all_audio_analysis = retrieve_audio_analysis(sp, song_df)
    fe_df = feature_engineering_dataset(all_audio_analysis)
    train = create_train_dataset(song_df, fe_df)

    conn = sqlite3.connect('data/songs.db')
    append_data_to_db(conn, train, song_metainfo_df)
    X_train, y_train, X_new, y_new = fetch_labeled_and_new_dataset(conn)

    if X_train.shape[1] != X_new.shape[1]:
        raise Exception("Dataset shapes do not match")

    classify_new_songs(X_train, y_train, X_new, y_new)
