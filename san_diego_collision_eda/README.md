# San Diego Collision EDA
Exploratory data analysis project using Tableau for analyzing San Diego County Collision since 2016

## Overview
- Grabbed original dataset from [City of San Diego Open Data Portal](https://data.sandiego.gov/datasets/police-collisions/)
- Applied geocoding to unique addresses using Python and Google Geocoding API and
- Stored latitude, longitude, zipcode in SQLite database
- Engineered features from the text of each unique charge description to group them into larger categories
- Created interactive visualization with Tableau so that one can look at specific location in San Diego County