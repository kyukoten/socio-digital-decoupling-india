import pandas as pd
import numpy as np
import os

# --- 1. LOAD DATASETS FROM DATA FOLDER ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Go up to main folder
data_dir = os.path.join(project_root, 'data')

trai_path = os.path.join(data_dir, 'TRAI_Lean_Data_2025.csv')
students_path = os.path.join(data_dir, 'Students.csv')

df_trai = pd.read_csv(trai_path)
df_students = pd.read_csv(students_path)

# --- 2. AGGREGATE TRAI DATA ---
# We calculate the average speed for each LSA (Telecom Circle) 
# so we can give every student a "Regional Infrastructure Score"
lsa_performance = df_trai.groupby('lsa').agg({
    'speed_mbps': 'mean'
}).reset_index().rename(columns={'speed_mbps': 'Avg_Regional_Speed_Mbps'})

print(f"✅ Aggregated performance for {len(lsa_performance)} Telecom Circles.")

# --- 3. DEFINE THE BRIDGE (State to LSA Mapping) ---
# This dictionary maps the 'State' names in your survey to TRAI 'lsa' names
mapping = {
    'Uttar pradesh': 'UP West', 'Chhattisgarh': 'Madhya Pradesh', 'Uttarakhand': 'UP West',
    'Delhi ncr': 'Delhi', 'Punjab': 'Punjab', 'Chandigarh': 'Punjab', 'Puducherry': 'Tamil Nadu',
    'Rajasthan': 'Rajasthan', 'Meghalaya': 'North East', 'Jharkhand': 'Bihar', 'Maharashtra': 'Maharashtra',
    'Tamil nadu': 'Tamil Nadu', 'Manipur': 'North East', 'Arunachal pradesh': 'North East',
    'Telangana': 'Andhra Pradesh', 'Orissa': 'Orissa', 'Bihar': 'Bihar', 'Madhya pradesh': 'Madhya Pradesh',
    'Himachal pradesh': 'Himachal Pradesh', 'Kerala': 'Kerala', 'Gujarat': 'Gujarat', 'Assam': 'Assam',
    'Jammu': 'Jammu & Kashmir', 'Nagaland': 'North East', 'Karnataka': 'Karnataka', 'Haryana': 'Haryana',
    'West bengal': 'West Bengal', 'Andhra pradesh': 'Andhra Pradesh', 'Goa': 'Maharashtra',
    'Tripura': 'North East', 'Sikkim': 'West Bengal', 'Mizoram': 'North East', 'Andaman': 'West Bengal'
}

# --- 4. MERGE DATA ---
df_students['lsa'] = df_students['State'].map(mapping)
final_df = df_students.merge(lsa_performance, on='lsa', how='left')

# Handle cases where a state doesn't have TRAI data yet (fill with national average)
national_avg = lsa_performance['Avg_Regional_Speed_Mbps'].mean()
final_df['Avg_Regional_Speed_Mbps'] = final_df['Avg_Regional_Speed_Mbps'].fillna(national_avg)

# --- 5. FEATURE ENGINEERING (Advanced Step) ---
# Convert 'Internet_Access' categorical labels into a numeric 'Access_Quality' score
access_map = {'Poor': 1, 'Medium': 2, 'High': 3}
final_df['Subjective_Access_Score'] = final_df['Internet_Access'].map(access_map)

# Create a "Discrepancy Score": Difference between real speed and student's perception
# (Higher means the infrastructure is good but the student's personal connection is poor)
final_df['Infrastructure_Efficiency'] = final_df['Avg_Regional_Speed_Mbps'] * final_df['Subjective_Access_Score']

# --- 6. SAVE THE AUDITOR DATASET TO DATA FOLDER ---
output_path = os.path.join(data_dir, 'Integrated_AI_Auditor_Data.csv')
final_df.to_csv(output_path, index=False)

print("\n" + "="*40)
print("🎊 INTEGRATION SUCCESSFUL!")
print(f"📊 Final Dataset Shape: {final_df.shape}")
print(f"📂 File Saved: {output_path}")
print("="*40)

# Quick Preview of the Digital Divide
print("\nTop 5 Regions by Avg 5G/4G Speed in your Dataset:")
print(lsa_performance.sort_values(by='Avg_Regional_Speed_Mbps', ascending=False).head())