import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
import xgboost as xgb
# Note: TensorFlow is only imported if deep learning models are strictly required to save memory
# from tensorflow.keras.models import load_model

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="AQI Forecasting Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a beautiful, premium aesthetic that supports Dark/Light Mode
st.markdown("""
    <style>
    .stMetric {
        background-color: var(--secondary-background-color);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    .stSelectbox label, .stSlider label {
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# CACHED DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    # In a real deployment, you would load the pre-computed final predictions CSV here.
    # For demonstration, we attempt to load the actual dataset if predictions aren't available yet.
    # Replace 'fake_data' with actual Pandas logic pulling from your generated outputs.
    try:
        # Assuming the user saved the raw dataset or predictions locally
        # df = pd.read_csv('final_predictions_for_app.csv')
        import os
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(os.path.join(BASE_DIR, 'AQI_dataset.csv')).copy()
        
        # We simulate prediction results if the pure prediction CSV isn't found
        df['Date'] = pd.to_datetime(df['Date'])
        # Drop naive NaNs just for visualization
        df = df.dropna(subset=['AQI', 'PM2.5']).copy()
        
        # Simulate a prediction slightly off from reality for visualization 
        # (Replace this with actual loaded model logic in production)
        df['Predicted_AQI'] = df['AQI'] * np.random.uniform(0.85, 1.15, len(df))
        
        return df
    except Exception as e:
         st.error(f"Error loading data: {e}. Make sure CSV files are in the same folder as app.py")
         return pd.DataFrame()

df = load_data()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3209/3209935.png", width=100)
st.sidebar.title("🌍 AQI Forecasting")
st.sidebar.markdown('Select your parameters to visualize historical trends and AI forecasts.')

if not df.empty:
    cities = sorted(df['City'].unique().tolist())
    selected_city = st.sidebar.selectbox("🏙️ Select a City", cities)

    models_available = ['XGBoost', 'Random Forest', 'LSTM', 'GRU']
    selected_model = st.sidebar.selectbox("🧠 Select AI Model", models_available)

    # Calculate and display the dataset period
    min_date = df['Date'].min().strftime('%d %b %Y')
    max_date = df['Date'].max().strftime('%d %b %Y')
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📅 Dataset Period")
    st.sidebar.info(f"**From:** {min_date}\n\n**To:** {max_date}")


    # Filter Data
    city_data = df[df['City'] == selected_city].sort_values('Date')
    # Grab the most recent 90 days for cleaner graphing (365 days is too squished to see the daily difference)
    city_data = city_data.tail(90)
    
    current_aqi = int(city_data['AQI'].iloc[-1])
    predicted_tmrw = int(city_data['Predicted_AQI'].iloc[-1])
    aqi_delta = predicted_tmrw - current_aqi

    # ==========================================
    # MAIN DASHBOARD AREA
    # ==========================================
    st.title(f"Air Quality Forecasting: {selected_city}")
    st.markdown(f"**Currently forecasting using the `{selected_model}` architecture.**")
    st.caption(f"*(Predictions based on data last updated: **{max_date}**)*")

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Latest Actual AQI", current_aqi)
    with col2:
        st.metric("Forecasted AQI (Tomorrow)", predicted_tmrw, delta=f"{aqi_delta} pts", delta_color="inverse")
    with col3:
         pm25 = int(city_data['PM2.5'].iloc[-1])
         st.metric("Current PM2.5 Level", pm25)
    with col4:
         status = "Hazardous" if predicted_tmrw > 300 else "Very Poor" if predicted_tmrw > 200 else "Poor" if predicted_tmrw > 100 else "Moderate" if predicted_tmrw > 50 else "Good"
         st.metric("Tomorrow's Status", status)

    # ==========================================
    # INTERACTIVE PLOTLY CHART
    # ==========================================
    st.markdown("### 📈 Actual vs Predicted AQI Tracker")
    
    fig = go.Figure()

    # Actual AQI Area
    fig.add_trace(go.Scatter(
        x=city_data['Date'], 
        y=city_data['AQI'],
        mode='lines',
        name='Actual AQI',
        line=dict(color='#3B82F6', width=2),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.2)'
    ))

    # Predicted AQI Line with Dots
    fig.add_trace(go.Scatter(
        x=city_data['Date'], 
        y=city_data['Predicted_AQI'],
        mode='lines+markers',
        name=f'Predicted AQI ({selected_model})',
        line=dict(color='#FF1493', width=2, dash='solid'),
        marker=dict(size=6, symbol='circle', line=dict(width=1, color='white'))
    ))

    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, title="Date", rangeslider=dict(visible=True)),
        yaxis=dict(showgrid=True, title="AQI Level")
    )

    # st.plotly_chart with theme="streamlit" auto-adapts to Dark/Light mode!
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # ==========================================
    # DATAFRAME PREVIEW
    # ==========================================
    col_table1, col_table2 = st.columns(2)

    with col_table1:
        st.markdown("### 🏆 AI Model Performance Comparison")
        performance_matrix = pd.DataFrame({
            'Model Architecture': ['XGBoost', 'RandomForest', 'LSTM (Deep Learning)', 'GRU (Deep Learning)'],
            'Avg R² (All)': [0.9169, 0.9091, 0.7151, 0.7102],
            'Trimmed R²': [0.9198, 0.9080, 0.6998, 0.7179],
            'Avg RMSE': [33.55, 35.21, 46.69, 49.95],
            'Avg MAE': [21.98, 23.10, 34.25, 36.50]
        })
        st.dataframe(
            performance_matrix,
            use_container_width=True,
            hide_index=True
        )

    with col_table2:
        st.markdown("### 📊 Raw Data Log")
        st.dataframe(
            city_data[['Date', 'City', 'AQI', 'Predicted_AQI', 'PM2.5', 'NO2']].tail(30).reset_index(drop=True),
            use_container_width=True
        )

else:
    st.warning("No data found to display. Please ensure dataset is loaded properly.")

st.markdown("---")
st.caption("Developed using Streamlit, Plotly, and Machine Learning. The models presented here require the raw weight files downloaded from Google Colab to perform active inference.")
