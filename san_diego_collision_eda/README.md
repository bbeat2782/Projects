# San Diego Collision EDA
Exploratory data analysis project using Tableau for analyzing San Diego County Collision since 2016

## Overview
- Grabbed original dataset from City of San Diego Open Data Portal, [Traffic Collisions - basic reports](https://data.sandiego.gov/datasets/police-collisions/) and [Police Beats](https://data.sandiego.gov/datasets/police-beats/) and grid map of San Diego from [SanGis](https://www.sangis.org/)
- Applied geocoding to unique addresses using Python and Google Geocoding API
- Stored latitude, longitude, zipcode in SQLite database
- Engineered features from the text of each unique charge description to group them into larger categories
- Created interactive visualization with Tableau so that viewers can look at specific location in San Diego County

### Data Collection
**Traffic Collisions - basic reports** dataset contains the following columns and rows.

|    |   report_id | date_time           |   police_beat |   address_no_primary | address_pd_primary   | address_road_primary   | address_sfx_primary   | address_pd_intersecting   | address_name_intersecting   | address_sfx_intersecting   | violation_section   | violation_type   | charge_desc                                            |   injured |   killed | hit_run_lvl   |
|---:|------------:|:--------------------|--------------:|---------------------:|:---------------------|:-----------------------|:----------------------|:--------------------------|:----------------------------|:---------------------------|:--------------------|:-----------------|:-------------------------------------------------------|----------:|---------:|:--------------|
|  0 |      171111 | 2015-01-14 20:00:00 |           835 |                 4200 |                      | JUNIPER                | STREET                |                           |                             |                            | MISC-HAZ            | VC               | MISCELLANEOUS HAZARDOUS VIOLATIONS OF THE VEHICLE CODE |         0 |        0 | MISDEMEANOR   |
|  1 |      192016 | 2015-03-19 12:00:00 |           622 |                 5200 |                      | LINDA VISTA            | ROAD                  |                           |                             |                            | MISC-HAZ            | VC               | MISCELLANEOUS HAZARDOUS VIOLATIONS OF THE VEHICLE CODE |         0 |        0 | MISDEMEANOR   |
|  2 |      190012 | 2015-03-24 03:05:00 |           626 |                 1000 | W                    | WASHINGTON             | STREET                |                           |                             |                            | 22107               | VC               | TURNING MOVEMENTS AND REQUIRED SIGNALS                 |         2 |        0 | nan           |
|  3 |      191866 | 2015-03-27 23:56:00 |           613 |                 2800 |                      | WORDEN                 | STREET                |                           |                             |                            | 22107               | VC               | TURNING MOVEMENTS AND REQUIRED SIGNALS                 |         1 |        0 | nan           |
|  4 |      185207 | 2015-07-06 11:45:00 |           813 |                 2800 |                      | EL CAJON               | BOULEVARD             |                           |                             |                            | 20002(A)            | VC               | HIT AND RUN                                            |         0 |        0 | MISDEMEANOR   |

**Police Beats** dataset contains the mapping of police_beat values to name of neighbors in San Diego.

The grid map of San Diego downloaded from SanGis

<img
  src="/imh/sd_grid.png",
  alt="Grid map of San Diego",
  style="display: inline-block; margin: 0 auto; max-width: 300px">
  