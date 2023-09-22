# Projects
Collection of data science projects

## 1. [Workout Song Recommendation](https://github.com/bbeat2782/Projects/tree/main/song_clf)
Recommending songs using Logistic Regression and Spotify & Youtube APIs

### Used skills
- Labeled 1300+ songs into 3 categories and collected audio feature data using Spotify API in python
- Applied feature engineering and `Principal Component Analysis` to create a dataset of 114 features
- Achieved f1 weighted score of 0.68 using a `logistic regression` model
- Created a workflow to add new songs in a `SQLite` database and update YouTube playlist automatically using Youtube API and SQLite database


## 2. [Metacritic User Rating Prediction](https://github.com/bbeat2782/Projects/tree/main/rating_pred)
Predicting Metacritic user ratings with data available before movies are released such as critic reviews, critic ratings, and genre.

### Used skills
- Websraping movie info, critic reviews, and user reviews from [Metacritic](https://www.metacritic.com/movie) using Python `BeautifulSoup` and storing in `SQLite` database.
- Extracting text sentiment features using `nltk.sentiment.SentimentIntensityAnalyzer` and keywords using `KeyBERT`.
- Analyzing scraped data using `plotly`.
- Predicting user rating with `lightgbm.LGBMRegressor` and hyperparameter tuning with `Optuna`.


## 3. [San Diego Collision EDA](https://github.com/bbeat2782/Projects/tree/main/san_diego_collision_eda)
Exploratory data analysis project using Tableau for analyzing San Diego County Collision since 2016

### Used skills
- Retrieving geo coordinates using Google Geocoding API
- Storing data to `SQLite` database
- Creating interactive visual with Tableau
