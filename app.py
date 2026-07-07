import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os
import plotly.express as px

# --- 1. DASHBOARD CONFIGURATION ---
st.set_page_config(page_title="India AI Education Auditor", layout="wide")
st.title("🇮🇳 Socio-Digital Decoupling: AI Education Auditor")
st.markdown("Explore how regional 5G/4G infrastructure impacts student success with Generative AI.")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Add 'data' to the path
    data_path = os.path.join(current_dir, 'data', 'Integrated_AI_Auditor_Data.csv')
    df = pd.read_csv(data_path)
    return df

df = load_data()

# --- 3. TRAIN ML MODEL (In Background) ---
@st.cache_resource
def train_model(df):
    features = ['Daily_Usage_Hours', 'Avg_Regional_Speed_Mbps', 'Subjective_Access_Score']
    ml_df = df.dropna(subset=features + ['Impact_on_Grades'])
    
    X = ml_df[features]
    y = ml_df['Impact_on_Grades']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# --- 4. SIDEBAR: THE POLICY SIMULATOR ---
st.sidebar.header("⚙️ Policy Simulator")
st.sidebar.write("Simulate infrastructure upgrades to see the predicted impact on student grades.")

sim_usage = st.sidebar.slider("Student Daily AI Usage (Hours)", 0.0, 10.0, 2.5)
sim_access = st.sidebar.selectbox("Subjective Device Access Quality (1=Poor, 2=Med, 3=High)", [1, 2, 3], index=1)

st.sidebar.markdown("### Upgrade Network Infrastructure")
# This is the main variable we want to test
sim_speed = st.sidebar.slider("Simulated Regional Avg Speed (Mbps)", 5.0, 200.0, 25.0)

# Prediction based on user sliders
input_data = pd.DataFrame({
    'Daily_Usage_Hours': [sim_usage],
    'Avg_Regional_Speed_Mbps': [sim_speed],
    'Subjective_Access_Score': [sim_access]
})
predicted_impact = model.predict(input_data)[0]

# --- 5. MAIN DASHBOARD UI ---
col1, col2 = st.columns(2)

with col1:
    st.success(f"### Predicted Academic Impact: {predicted_impact:.2f} points")
    st.write("*(Scale: Negative values indicate grade drop, positive indicates improvement)*")
    
    if predicted_impact > 0:
        st.write("✅ **Conclusion:** This infrastructure profile supports productive AI learning.")
    else:
        st.error("⚠️ **Conclusion:** Infrastructure bottleneck detected. AI tools may cause frustration rather than learning.")

with col2:
    # Show the current reality of the dataset
    st.markdown("### Current Reality: Top 5 Fastest Telecom Circles")
    top_circles = df.groupby('lsa')['Avg_Regional_Speed_Mbps'].mean().sort_values(ascending=False).head()
    st.dataframe(top_circles)

st.markdown("---")
st.markdown("### 📡 The Infrastructure Baseline (TRAI Data)")
# Reusing your boxplot logic from 2.py
fig4, ax4 = plt.subplots(figsize=(12, 6))
# Sort LSAs by average speed for a clean waterfall look
sorted_lsas = df.groupby('lsa')['Avg_Regional_Speed_Mbps'].median().sort_values(ascending=False).index

sns.boxplot(
    data=df, 
    x='lsa', 
    y='Avg_Regional_Speed_Mbps', 
    order=sorted_lsas,
    palette='magma',
    ax=ax4
)
ax4.set_title('Regional Speed Discrepancy (The State Report Card)', fontsize=14, fontweight='bold')
ax4.set_ylabel('Speed (Mbps)')
ax4.set_xlabel('')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig4)


# --- 6. TRUE GEOSPATIAL VISUALIZATION ---
st.markdown("---")
st.markdown("### 🗺️ The Digital Divide: National Infrastructure Map")
st.write("Hover over a Telecom Circle (LSA) to see its average 5G/4G speed and the resulting impact on student grades.")

# Step A: Define Approximate Coordinates for Telecom LSAs
# (Because LSAs are different from standard state borders, we use central coordinates)
lsa_coords = {
    'Delhi': {'lat': 28.7041, 'lon': 77.1025},
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
    'Maharashtra': {'lat': 19.7515, 'lon': 75.7139},
    'UP West': {'lat': 28.9845, 'lon': 77.7064}, # Meerut/Agra area
    'UP East': {'lat': 26.8467, 'lon': 80.9462}, # Lucknow/Varanasi area
    'Madhya Pradesh': {'lat': 22.9734, 'lon': 78.6569},
    'Punjab': {'lat': 31.1471, 'lon': 75.3412},
    'Tamil Nadu': {'lat': 11.1271, 'lon': 78.6569},
    'Rajasthan': {'lat': 27.0238, 'lon': 74.2179},
    'North East': {'lat': 25.5788, 'lon': 91.8933}, # Shillong/Guwahati center
    'Bihar': {'lat': 25.0961, 'lon': 85.3131},
    'Andhra Pradesh': {'lat': 15.9129, 'lon': 79.7400},
    'Orissa': {'lat': 20.9517, 'lon': 85.0985},
    'Himachal Pradesh': {'lat': 31.1048, 'lon': 77.1665},
    'Kerala': {'lat': 10.8505, 'lon': 76.2711},
    'Gujarat': {'lat': 22.2587, 'lon': 71.1924},
    'Assam': {'lat': 26.2006, 'lon': 92.9376},
    'Jammu & Kashmir': {'lat': 33.7782, 'lon': 76.5762},
    'Karnataka': {'lat': 15.3173, 'lon': 75.7139},
    'Haryana': {'lat': 29.0588, 'lon': 76.0856},
    'West Bengal': {'lat': 22.9868, 'lon': 87.8550}
}

# Step B: Map the coordinates to our dataframe
# We group by LSA first so we only plot one bubble per region
map_df = df.groupby('lsa').agg({
    'Avg_Regional_Speed_Mbps': 'mean',
    'Impact_on_Grades': 'mean',
    'Trust_in_AI_Tools': 'mean'
}).reset_index()

# Extract coordinates safely
map_df['lat'] = map_df['lsa'].map(lambda x: lsa_coords.get(x, {}).get('lat', 20.5937)) # Default to central India
map_df['lon'] = map_df['lsa'].map(lambda x: lsa_coords.get(x, {}).get('lon', 78.9629))

# Step C: Render the Interactive Plotly Map
fig_map = px.scatter_mapbox(
    map_df, 
    lat="lat", 
    lon="lon", 
    hover_name="lsa", 
    hover_data={
        "Avg_Regional_Speed_Mbps": ':.2f',
        "Impact_on_Grades": ':.2f',
        "Trust_in_AI_Tools": ':.2f',
        "lat": False, "lon": False
    },
    color="Avg_Regional_Speed_Mbps",
    size="Trust_in_AI_Tools",
    color_continuous_scale=px.colors.sequential.Plasma,
    size_max=30,
    zoom=3.5, 
    center={"lat": 22.0, "lon": 79.0},
    mapbox_style="carto-positron",
    title="Telecom Circles by Average Speed and AI Trust"
)

fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)

# --- 7. STUDENT DEMOGRAPHICS & TOOL PREFERENCES ---
st.markdown("---")
st.markdown("### 📊 Student AI Usage Demographics")
st.write("Understanding how students access Generative AI.")

col3, col4 = st.columns(2)

with col3:
    # Reusing your pivot table logic from ML.ipynb
    usage_agg = df.groupby(['Preferred_AI_Tool', 'Device_Used'])['Daily_Usage_Hours'].sum().reset_index()
    usage_pivot = usage_agg.pivot(index='Preferred_AI_Tool', columns='Device_Used', values='Daily_Usage_Hours').fillna(0)
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    usage_pivot.plot(kind='bar', stacked=True, colormap='Set2', ax=ax2)
    ax2.set_title('Total Daily AI Usage Hours by Device')
    ax2.set_ylabel('Hours')
    ax2.set_xlabel('AI Tool')
    plt.xticks(rotation=45)
    st.pyplot(fig2)

with col4:
    # A quick pie chart showing the Trust levels
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    df['Trust_in_AI_Tools'].value_counts().sort_index().plot(kind='pie', autopct='%1.1f%%', cmap='Wistia', ax=ax3)
    ax3.set_title('Overall Student Trust in AI (1 = Low, 5 = High)')
    ax3.set_ylabel('')
    st.pyplot(fig3)