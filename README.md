# Gridlock-2026

Flipkart Gridlock Hackathon 2.0 — Traffic Demand Prediction Solution
================================================================================

INTRODUCTION
-------------
The objective of this challenge was to accurately predict traffic demand using historical traffic information along with weather, road, temporal, and geographical features. 

Traffic demand is influenced by multiple factors such as time of day, road infrastructure, weather conditions, and location. Therefore, instead of relying on raw features directly, the solution focuses on extracting meaningful patterns through feature engineering and spatial analysis before applying machine learning. The overall strategy was to build a robust pipeline capable of capturing both temporal traffic behavior and location-based traffic characteristics while maintaining strong generalization performance on unseen data.

The task was to predict traffic demand — a normalized value between 0 and 1 — at specific road locations and 15-minute time slots across Singapore's road network. The evaluation metric is:
    score = max(0, 100 * R2_score(actual, predicted))


SOLUTION STRATEGY
------------------
The solution was developed in the following stages:
  1. Exploratory Data Analysis (EDA)
  2. Data Cleaning and Preprocessing
  3. Target Variable Transformation
  4. Temporal Feature Engineering
  5. Geographical Feature Engineering
  6. Spatial Clustering
  7. Model Training using CatBoost
  8. Multi-Seed Ensembling
  9. Final Prediction Generation

The primary focus was to convert the available raw information into features that better represent real-world traffic behavior.


DATA UNDERSTANDING 
-------------------
The dataset covers 1,249 unique geohash locations across two days (Day 48 and Day 49), with 96 time slots per day (one per 15 minutes). The training set has 77,299 rows; the test set has 41,778 rows (Day 49 only). Before building the model, extensive exploratory analysis was performed to understand the dataset, including inspection of dimensions, identification of feature types, missing value analysis, and statistical summaries of numerical variables.

Key findings from EDA:
- Target Variable Analysis: Demand distribution was highly right-skewed. Most locations sit near zero (median ~0.048), while a few high-traffic corridors push the mean up to ~0.094. A small number of observations contained extremely high demand values. The presence of outliers indicated the strict need for target transformation.
- Temporal Analysis: Traffic demand varied significantly throughout the day. Peak traffic periods showed higher average demand, and time-related information appeared to be highly informative.
- Road and Weather Analysis: Traffic behavior differed considerably depending on road characteristics. RoadType is the single strongest predictor; Highway locations average 0.61 demand — nearly 10x higher than Residential roads at 0.057. Weather conditions also showed a measurable influence on demand.
- Spatial Analysis: Location-wise traffic demand was analyzed using geohash values. Certain geographical regions consistently exhibited higher traffic demand, proving that location information was expected to contribute significantly to prediction performance. 

The findings from this EDA directly influenced the feature engineering strategy.


DATA CLEANING AND PREPROCESSING
--------------------------------
Target Transform:
The raw demand distribution is highly skewed. To help the model learn more evenly across the full range of values, we applied two transformations:
  1. Winsorize at the 99.5th percentile — this caps extreme spikes that would otherwise dominate the RMSE loss and cause the model to overfit to rare events.
  2. log1p transform — compresses the resulting distribution into a roughly symmetric shape that gradient boosting handles much better.
At inference time, we reverse with expm1 and clip predictions to [0, 1].

Missing Value Handling:
- Categorical columns (RoadType, Weather, LargeVehicles, Landmarks): Filled with the string 'Unknown'. CatBoost treats this as a distinct, meaningful category — sensor failures and missing road metadata are themselves informative signals, not noise.
- Temperature: Forward-filled and back-filled within each geohash location (sensors nearby tend to share conditions), with a global median fallback for any remaining gaps.


FEATURE ENGINEERING
--------------------
We engineered 18 features in total — 7 categorical and 11 numeric.

Time Features:
  - Hour, Minute: extracted directly from the timestamp string
  - Quarter_Hour: 15-minute bucket (0, 1, 2, 3) as a categorical — this captures intra-hour demand patterns that hourly aggregation misses
  - Hour_sin, Hour_cos: cyclical encoding using total minutes out of 1440, ensuring that 11:45 PM and 12:00 AM are treated as temporally close
  - DayOfWeek: day modulo 7
  - IsWeekend: binary indicator for weekend vs. weekday behavior

Spatial Features:
  - Latitude, Longitude: decoded from the 6-character geohash strings using a custom base32 decoder (no external geohash library required)
  - Dist_to_Center: Euclidean distance from the grid's geographic center (1.29°N, 103.85°E), capturing proximity to the central business district
  - Spatial_Zone: K-Means clustering (k=8) applied to latitude and longitude, grouping locations into organic geographic neighborhoods. CatBoost uses this as a categorical feature to learn zone-specific demand patterns

Original Columns Retained:
  Geohash, RoadType, NumberofLanes, LargeVehicles, Landmarks, Temperature, Weather — all passed directly to CatBoost.


MODEL ARCHITECTURE
-------------------
We used CatBoost as our sole model after confirming it outperformed more complex multi-model ensembles. The key insight was how CatBoost handles categorical features. Instead of manually encoding geohash, RoadType, Weather, and other categoricals into numbers, we passed them as raw strings and let CatBoost compute its own Ordered Target Statistics internally. This approach:
  - Avoids target leakage (categories are encoded on shuffled subsets)
  - Produces richer representations than label or frequency encoding
  - Removes the need for cross-validation to safely compute target encodings

Final hyperparameters:
  iterations:   2000
  learning_rate:0.06
  depth:         7
  eval_metric: RMSE

We trained on 100% of the training data — no holdout — because the test set covers a separate day (Day 49) and time-based validation consistently underestimated true leaderboard performance anyway.


MULTI-SEED GEOMETRIC MEAN ENSEMBLE
------------------------------------
A single model can sometimes be sensitive to random initialization. To improve robustness, multiple CatBoost models were trained using different random seeds: [42, 0, 7, 13, 99].

Each seed changes CatBoost's internal behavior in subtle ways:
  - Row shuffling order for ordered target encoding
  - Which features each tree evaluates
  - Split selection within decision tree construction

The predictions from all models were combined using geometric mean ensembling:
    final = (p1 * p2 * p3 * p4 * p5) ^ (1/5)

Why Ensemble? Benefits include:
  - Reduced variance
  - Improved stability
  - Better generalization
  - Lower sensitivity to random fluctuations

This approach consistently produced more reliable predictions than a single model. We chose the geometric mean over the arithmetic mean because it is more conservative with high predictions — naturally dampening overconfident spikes in the right tail — and suits the skewed nature of traffic demand. Critically, this requires no meta-model, eliminating any risk of overfitting at the ensemble stage.


TOOLS AND LIBRARIES USED
-------------------------
Programming Language: 
- Python

Core Libraries:
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- CatBoost


WHAT DID NOT WORK
------------------
- LightGBM + XGBoost + CatBoost -> Ridge stacking (90.71): The Ridge meta-model overfit to cross-validation patterns that did not transfer to the actual test day.
- Manual target encoding: CatBoost's native handling consistently outperformed any manual encoding scheme we tried.
- Arithmetic mean ensemble: The geometric mean performed better because it is more conservative for large predictions, which matter most in a right-skewed target setting.


SCORE PROGRESSION
------------------
Approach                                                    Score
------------------------------------------------------------
LGB + XGB + CatBoost -> Ridge stacking                      90.71
Geometric mean of base models                               90.79
Weighted blend of geometric means                           90.94
CatBoost native categoricals (single seed)                  91.62
CatBoost native categoricals + 5-seed geomean               91.689


CONCLUSION
-----------
The final solution combines exploratory data analysis, advanced feature engineering, spatial intelligence, clustering, and ensemble learning to predict traffic demand effectively. 

Special emphasis was placed on extracting meaningful temporal and geographical patterns from the data rather than relying solely on raw features. The combination of geohash decoding, spatial clustering, target transformation, and CatBoost ensembling resulted in a robust and scalable traffic demand prediction pipeline.

NOTE:The solution file with source code is provided as a Jupyter notebook under the name solution_best2_with_md.ipynb for the best scored model.
