import sqlite3
import plotly.graph_objects as go
import numpy as np
import pandas as pd

sqldb_path = "data/rating_database.db"


def insert_into_sql(statement, db_path=sqldb_path):
    """
    Insert or update sql database based on the statement parameter

    Parameters
    ----------
    statement : str
        The statement to execute
    db_path : str
        Path to sqlite database that you want to execute the statements

    Returns
    -------
    None
    """

    try:
        sqliteConnection = sqlite3.connect(db_path)
        cursor = sqliteConnection.cursor()
        cursor.execute(statement)
        sqliteConnection.commit()
        cursor.close()
    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)


def fetch_from_sql(statement, db_path=sqldb_path):
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


def get_num_review(ids, critic=True):
    if critic:
        get_movie_with_id = (
            f'SELECT critic_rating FROM Critics WHERE movie_id in ({",".join(ids)})'
        )
    else:
        get_movie_with_id = (
            f'SELECT user_rating FROM Users WHERE movie_id in ({",".join(ids)})'
        )
    return len(fetch_from_sql(get_movie_with_id))


def multi_plot(df, genres_vc):
    def add_data(fig, col, genre):
        fig.add_trace(
            go.Histogram(
                x=df[df.apply(lambda r: genre in r.values, axis=1)][col],
                name=col,
                histnorm="probability density",
                hovertemplate="Rating: %{x}, Proportion: %{y}",
            )
        )

    def create_layout_button(column):
        return dict(
            label=column,
            method="update",
            args=[
                {
                    "visible": np.ravel(
                        [[v, v] for v in genres_vc.index.isin([column])]
                    ),
                    "title": column,
                    "showlegend": True,
                }
            ],
        )

    fig = go.Figure()
    for genre in genres_vc.index:
        add_data(fig, "critic_rating", genre)
        add_data(fig, "user_rating", genre)

    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                active=0,
                showactive=True,
                buttons=list(
                    genres_vc.index.map(lambda column: create_layout_button(column))
                ),
            )
        ],
        barmode="overlay",
        title="Critics and users rating distribution by genre",
        xaxis_title="Rating",
        yaxis_title="Proportion",
    )
    fig.update_traces(opacity=0.75)

    for k in range(2, 40):
        fig.update_traces(visible=False, selector=k)

    fig.show()


def create_critic_stat_with_userscore():
    combined_stat = {}
    for rating in np.arange(0.0, 10.1, 0.1):
        rating = np.round(rating, 1)
        try:
            ids = [
                str(row[0])
                for row in fetch_from_sql(
                    f"SELECT movie_id FROM Movie WHERE movie_user_rating == {rating}"
                )
            ]
            stat = pd.DataFrame(
                fetch_from_sql(
                    "SELECT neg, neu, pos, compound FROM Critics "
                    f'WHERE movie_id in ({",".join(ids)})'
                )
            ).apply(["mean", "std"])
            stat = stat.transpose().to_numpy().flatten()
            combined_stat[rating] = stat
        except:
            continue

    stat_df = pd.DataFrame(index=combined_stat.keys(), data=combined_stat.values())
    stat_df.rename(
        columns={
            0: "neg_mean",
            1: "neg_std",
            2: "neu_mean",
            3: "neu_std",
            4: "pos_mean",
            5: "pos_std",
            6: "compound_mean",
            7: "compound_std",
        },
        inplace=True,
    )

    return stat_df


def create_dropdown_lineplot(df):
    sentiment_types = ["compound", "neg", "neu", "pos"]

    def add_data(fig, col):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                name=col,
                mode="lines",
                hovertemplate="Rating: %{x}, Mean score: %{y}",
            )
        )

    def create_layout_button(column):
        return dict(
            label=column,
            method="update",
            args=[
                {
                    "visible": np.ravel(pd.Series(sentiment_types).isin([column])),
                    "title": column,
                    "showlegend": False,
                }
            ],
        )

    fig = go.Figure()
    for sentiment_type in sentiment_types:
        add_data(fig, sentiment_type + "_mean")

    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                active=0,
                showactive=True,
                buttons=list(
                    pd.Series(sentiment_types).map(
                        lambda column: create_layout_button(column)
                    )
                ),
            )
        ],
        title="Mean sentiment scores for critic reviews",
        xaxis_title="Rating",
        yaxis_title="Mean score",
    )

    for k in range(1, 4):
        fig.update_traces(visible=False, selector=k)

    fig.show()


def critic_sentiment_with_userscore():
    df = create_critic_stat_with_userscore()
    create_dropdown_lineplot(df)


def multi_lineplot(rating_type, tname):
    features = ["movie_id", "neg", "neu", "pos", "compound", rating_type]
    df = pd.DataFrame(
        fetch_from_sql(f'SELECT {", ".join(features)} FROM {tname}')
    ).set_axis(features, axis=1)
    df[features[-1]] = df[features[-1]].astype(int)
    df = df.drop("movie_id", axis=1).groupby(rating_type).agg(["mean", "std"])
    df.columns = ["_".join(col) for col in df.columns]

    create_dropdown_lineplot(df)


def combine_critic_sentiment_with_userscore():
    combined = pd.DataFrame()
    for rating in np.arange(0.0, 10.1, 0.1):
        rating = np.round(rating, 1)
        try:
            ids = [
                str(row[0])
                for row in fetch_from_sql(
                    f"SELECT movie_id FROM Movie WHERE movie_user_rating == {rating}"
                )
            ]
            rows = pd.DataFrame(
                fetch_from_sql(
                    f'SELECT neg, neu, pos, compound FROM Critics WHERE movie_id in ({",".join(ids)})'
                )
            )
            rows["user_rating"] = rating
            combined = pd.concat([combined, rows])
        except:
            continue

    return combined.rename(columns={0: "neg", 1: "neu", 2: "pos", 3: "compound"})


def create_dropdown_hist(df, rating_type):
    def adding_data(fig, df, r, t, c):
        fig.add_trace(
            go.Histogram(
                visible=False,
                x=df[t],
                name=f"{r}_{t}",
                marker_color=c,
                hovertemplate="Sentiment score: %{x}, Count: %{y}",
            )
        )

    fig = go.Figure()
    for r, df in df.groupby(rating_type):
        adding_data(fig, df, r, "neg", "#EF553B")
        adding_data(fig, df, r, "neu", "#00CC96")
        adding_data(fig, df, r, "pos", "#636EFA")

    for i in range(3):
        fig.data[i].visible = True

    steps = []

    for i in range(len(fig.data) // 3):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * 3 * len(fig.data)},
                {
                    "title": f"Sentiment distribution of rating: {fig.data[i*3]['name'].split('_')[0]}"
                },
            ],
        )
        tmp_i = 3 * i
        for r in range(3):
            step["args"][0]["visible"][tmp_i + r] = True
        steps.append(step)

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Rating: ", "visible": False},
            pad={"t": 50},
            steps=steps,
        )
    ]

    fig.update_layout(
        sliders=sliders,
        barmode="overlay",
        showlegend=False,
        title="Sentiment distribution of rating: 0",
        xaxis_title="Sentiment score",
        yaxis_title="Count",
    )
    fig.update_traces(opacity=0.75)

    fig.show()


def sentiment_multi_hist(rating_type, tname):
    features = ["movie_id", "neg", "neu", "pos", "compound", rating_type]
    df = pd.DataFrame(
        fetch_from_sql(f'SELECT {", ".join(features)} FROM {tname}')
    ).set_axis(features, axis=1)
    df[rating_type] = df[rating_type].astype(int)

    create_dropdown_hist(df, rating_type)


def plot_genre_by_yr(df, genres_vc):
    def add_data(fig, col, genre):
        fig.add_trace(
            go.Histogram(
                x=df[df.apply(lambda r: genre in r.values, axis=1)][col],
                name=genre,
                # histnorm = 'probability density',
                hovertemplate="Year: %{x}, Count: %{y}",
            )
        )

    def add_total(fig):
        fig.add_trace(
            go.Histogram(
                x=df["year"],
                # histnorm = 'probability density',
                name="Total",
                hovertemplate="Year: %{x}, Count: %{y}",
            )
        )

    def create_layout_button(column):
        return dict(
            label=column,
            method="update",
            args=[
                {
                    "visible": np.ravel(
                        [[v, v] for v in genres_vc.index.isin([column])]
                    ),
                    "title": column,
                    "showlegend": True,
                }
            ],
        )

    fig = go.Figure()

    for genre in genres_vc.index:
        add_total(fig)
        add_data(fig, "year", genre)

    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                active=0,
                showactive=True,
                buttons=list(
                    genres_vc.index.map(lambda column: create_layout_button(column))
                ),
            ),
        ],
        barmode="overlay",
        title="Critics and users rating distribution by genre",
        xaxis_title="Year",
        yaxis_title="Count",
    )
    fig.update_traces(opacity=0.75)

    for k in range(2, 40):
        fig.update_traces(visible=False, selector=k)

    fig.show()
