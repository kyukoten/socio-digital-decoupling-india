import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# --- 1. LOAD THE INTEGRATED DATA ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Go up to main folder
data_path = os.path.join(project_root, 'data', 'Integrated_AI_Auditor_Data.csv')

df = pd.read_csv(data_path)

# 2. Set up the plotting style
sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(16, 6))

# Plot: Does better infrastructure lead to higher trust in AI?
sns.boxplot(
    data=df, 
    x='Trust_in_AI_Tools', 
    y='Avg_Regional_Speed_Mbps', 
    palette='viridis'
)

plt.title('The Digital Divide: Regional Internet Speed vs. Student Trust in AI', fontsize=18, fontweight='bold')
plt.xlabel('Trust in AI Tools (Scale: 1 to 5)', fontsize=14)
plt.ylabel('Regional Avg Speed (Mbps)', fontsize=14)
plt.tight_layout()
plt.show()

# Quick Correlation Check
corr = df[['Avg_Regional_Speed_Mbps', 'Trust_in_AI_Tools', 'Impact_on_Grades', 'Daily_Usage_Hours']].corr()
print("Correlation Matrix:")
print(corr.round(3))

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Select the features for clustering
cluster_features = ['Avg_Regional_Speed_Mbps', 'Daily_Usage_Hours', 'Trust_in_AI_Tools', 'Impact_on_Grades']
df_cluster = df.dropna(subset=cluster_features)
X = df_cluster[cluster_features]

# 2. Scale the data (Crucial for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply K-Means (We expect 3 distinct socio-digital groups)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cluster['Digital_Persona'] = kmeans.fit_predict(X_scaled)

# Map numeric clusters to human-readable names based on their centers
# (We will just name them Cluster 0, 1, 2 for now and visualize them)
cluster_mapping = {0: 'Zone A', 1: 'Zone B', 2: 'Zone C'}
df_cluster['Digital_Persona'] = df_cluster['Digital_Persona'].map(cluster_mapping)

# 4. Visualize the Clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_cluster, 
    x='Avg_Regional_Speed_Mbps', 
    y='Impact_on_Grades', 
    hue='Digital_Persona', 
    palette='Set1',
    alpha=0.7,
    s=100
)
plt.title('Student Clustering: Network Speed vs Academic Impact', fontsize=16, fontweight='bold')
plt.xlabel('Avg Regional Speed (Mbps)')
plt.ylabel('Impact on Grades')
plt.legend(title='Student Segment')
plt.tight_layout()
plt.show()

# Print the average stats for each cluster
print("Cluster Profiles (Average Values):")
print(df_cluster.groupby('Digital_Persona')[cluster_features].mean().round(2))

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Prepare the Data for Machine Learning
# We need to convert text columns (categorical) into numbers
ml_df = df.copy()

# Select features we want to test
features_to_use = [
    'Year_of_Study', 
    'Daily_Usage_Hours', 
    'Trust_in_AI_Tools', 
    'Awareness_Level', 
    'Avg_Regional_Speed_Mbps',
    'Subjective_Access_Score',
    'Infrastructure_Efficiency'
]

# Drop rows with missing values in these specific columns
ml_df = ml_df.dropna(subset=features_to_use + ['Impact_on_Grades'])

X = ml_df[features_to_use]
y = ml_df['Impact_on_Grades']

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_train, y_train)

# --- REPLACEMENT FOR PART 4 IN 5.py ---
import shap

print("\n🚀 Initializing SHAP Explainer (Proving Causal Inference)...")

# 4. Initialize SHAP Tree Explainer
explainer = shap.TreeExplainer(rf_model)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# 5. Generate the Advanced SHAP Summary Plot
plt.figure(figsize=(12, 8))
plt.title('SHAP Analysis: The Bottleneck of AI Education', fontsize=16, fontweight='bold', pad=20)

# The summary_plot automatically generates a beautiful beeswarm plot
shap.summary_plot(shap_values, X_test, show=False)

# Save and show the plot in the main project folder
output_img = os.path.join(project_root, 'Figure_4_SHAP_Analysis.png')
plt.tight_layout()
plt.savefig(output_img, bbox_inches='tight', dpi=300)
plt.show()

print(f"✅ SHAP Analysis complete! Image saved as {output_img}")