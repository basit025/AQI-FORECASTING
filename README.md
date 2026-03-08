# 🌍 Indian Cities AQI Forecasting (ML & Deep Learning)

## 📌 Project Overview
**🚀 [Live Dashboard Demo](https://aqi-forecasting.streamlit.app/)** 

This project contains my end-to-end pipeline for predicting the Air Quality Index (AQI) of multiple major Indian cities using comprehensive 5-year chronological historical data. Air pollution is a major crisis, and having predictive foresight into whether tomorrow's air will be "Moderate" or completely "Hazardous" is critical for public health.

In this repository, I document exactly how I cleaned the messy real-world data, engineered predictive rolling features, built standard Machine Learning tree models alongside Deep Learning recurrent networks, and finally, visualized the results in a completely interactive **Streamlit Dashboard**.

---

## 🛠️ Data Preprocessing & Cleaning
Real-world IoT pollution data is incredibly noisy and filled with missing dates and corrupted sensor readings. Here is how I handled it:
1. **Target Filtering:** The primary goal is predicting `AQI`. I immediately dropped any rows where the target `AQI` was `NaN` because a model cannot learn without the ground-truth answers.
2. **Column Pruning:** Certain trace pollutants like `Xylene` and `Toluene` had massive data blackouts (over 50-60% missing). I completely dropped these columns because mathematically interpolating 3 years of missing data creates dangerous statistical hallucinations.
3. **Time-Series Integrity:** This is the most critical step. I explicitly forced the Pandas dataframe to resample every single city on a Daily (`D`) frequency.
    - If a sensor went down for 1 or 2 days, I used `.interpolate(method='time', limit=3)` to smoothly bridge the gap.
    - If a sensor went down for a whole month, I explicitly prevented interpolation. Why? Because if I interpolated a 30-day gap, my model would learn a perfectly straight, artificial line, ruining the realistic volatility of the dataset. Instead, large gaps remained `NaN`, which safely severed the mathematical shift connections.

---

## 🔬 Feature Engineering
A model is only as smart as the data it is fed. Instead of just showing the models "today's" pollution, I engineered a memory system for them.
1. **Lags & Rolling Averages:** I utilized `.shift()` and `.rolling()` to create new features for not just `AQI`, but also for the critical driving pollutants: `PM2.5`, `NO2`, and `CO`. 
    - The models now receive exactly what the AQI was 1 day ago, 7 days ago, and 30 days ago. 
    - They also receive the 3-day and 14-day rolling averages of PM2.5 to understand if physical pollution is "building up" in the atmosphere over the week.
2. **Temporal Features:** Because AQI is highly seasonal (e.g., Winter smog drops, Monsoon clearings), I extracted the `Month`, `Quarter`, and created an `Is_weekend` binary flag to help the model learn human traffic patterns.
3. **Preventing Data Leakage:** **This was paramount.** I specifically applied my 99th-percentile Outlier Clipping and my Yeo-Johnson Skewness transformations strictly *inside* the Cross-Validation loops. If I had applied these globally to the whole spreadsheet beforehand, the statistical averages of the "Future" test set would have leaked perfectly into the "Past" training set, artificially inflating the R² score.

---

## 🤖 Modeling Architecture
Because AQI is heavily influenced by strict chronological time, standard randomized train/test splits destroy the logical causality. Therefore, I built a brutal **Walk-Forward Time-Series Cross-Validation** loop.

### Why Multiple Models?
I explicitly chose to build and test *four* different model architectures across two entirely different machine learning paradigms (Tree-based ML vs. Recurrent Deep Learning).
My primary hypothesis was to determine whether traditional tabular Machine Learning (which requires intense manual lag feature engineering) could be outperformed by advanced Deep Learning (which natively understands sequential time without manual feature engineering). Forecasting pollution is notoriously volatile, so relying on a single algorithm without a comparative baseline is unscientific.

### 1. The Machine Learning Models
I trained two powerful Tree-based algorithms:
- **XGBoost Regressor:** Utilizes gradient boosting to sequentially correct its own errors.
- **Random Forest:** Builds hundreds of independent decision trees to battle variance and overfitting.
These models handled the tabular lag features exceptionally well and trained incredibly fast.

### 2. The Deep Learning Sequence Models
**Why Deep Learning?** While ML models look at a single row of data (today) and try to guess tomorrow, Recurrent Neural Networks (RNNs) look at sequential blocks of time. I wanted to see if an algorithm designed to naturally "remember" the sequence and flow of the past 14 days of weather could organically discover complex, non-linear atmospheric patterns that my engineered tabular features might have missed.

Instead of feeding flat, 1D rows into an algorithm, I restructured the dataset into 3D chronological blocks `(Samples, 14-Days Lookback, Features)`.
- **LSTM (Long Short-Term Memory):** Specialized in remembering long-term dependencies in the weather data without suffering from the vanishing gradient problem.
- **GRU (Gated Recurrent Unit):** A faster variant of the LSTM that proved equally capable at parsing the short 14-day chronological windows.
*Note: Because City IDs are categorical, I utilized a multi-input Keras `Embedding Layer` to mathematically represent the 200+ cities as dense vectors alongside the continuous weather data.*

---

## 🏆 Final Model Evaluation
To evaluate these architectures, I discarded the absolute best-performing time-window and the absolute worst-performing time-window, leaving me with a "Trimmed Mean" that represents the true, undeniable consistency of the models.

| Model Architecture | Avg R² (All) | Trimmed Mean R² | Avg RMSE (AQI pts) | Avg MAE (AQI pts) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | 0.9169 | **0.9198** | **33.55** | **21.98** |
| **RandomForest** | 0.9091 | 0.9080 | 35.21 | 23.10 |
| **LSTM (Deep Learning)** | 0.7151 | 0.6998 | 46.69 | 34.25 |
| **GRU (Deep Learning)** | 0.7102 | 0.7179 | 49.95 | 36.50 |

**Conclusion:** The **XGBoost Architecture** absolutely dominated the Forecasting task. XGBoost's ability to seamlessly ingest the tabular lag features (`PM2.5_lag1`, `AQI_roll7`) allowed it to consistently predict tomorrow's AQI within ~22 points of reality. The Deep Learning models struggled significantly more with the raw volatility.

---

## 📈 Interactive Streamlit Dashboard (`app.py`)
To make these powerful mathematical models accessible to a regular user, I developed a premium **Streamlit Interface**.

**🔗 [View the Live Web Dashboard Here](https://aqi-forecasting.streamlit.app/)** 

The dashboard automatically loads the `AQI_dataset.csv` and allows the user to:
1. Select any of the 26 cities from a dropdown to instantly filter the underlying analytics.
2. View Dynamic KPIs (Current AQI, Tomorrow's Predicted AQI, and a plain-English Risk Status like "Hazardous" or "Moderate").


### How to Run Locally:
```bash
# 1. Ensure dependencies are installed
pip install -r requirements.txt

# 2. Launch the Streamlit server
streamlit run app.py
```


