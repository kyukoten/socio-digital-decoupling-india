import requests
import pandas as pd
import time
import os

# --- CONFIGURATION ---
API_KEY = "YOUR_API_KEY_HERE"
RESOURCE_ID = "ade6e644-91b8-4d27-97ba-e8c42c48f278"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

# NEW: Limit records per month to keep the dataset lean but statistically valid
RECORDS_PER_MONTH = 2000 

def smart_harvest_2025(api_key):
    master_records = []
    
    for month in range(1, 13):
        print(f"📅 Sampling Month: {month}/2025")
        
        # We only need enough data to get a good average, so we fetch in 2 batches of 1000
        for offset in [0, 1000]: 
            params = {
                "api-key": api_key,
                "format": "json",
                "offset": offset,
                "limit": 1000,
                "filters[year]": "2025",
                "filters[month]": str(month)
            }
            
            try:
                response = requests.get(BASE_URL, params=params)
                if response.status_code == 200:
                    data = response.json()
                    records = data.get('records', [])
                    if records:
                        master_records.extend(records)
                
                # Brief pause to be safe
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Error fetching month {month}: {e}")
                break
        
        print(f"   Done. Total records now: {len(master_records)}")

    return pd.DataFrame(master_records)

# --- EXECUTION ---
print("🚀 Starting Optimized Data Harvest (Statistical Sampling)...")
df_raw = smart_harvest_2025(API_KEY)

if not df_raw.empty:
    # 1. KEEP ONLY THE IMPORTANT COLUMNS
    # This reduces file size significantly
    important_columns = ['lsa', 'operator', 'technology', 'speed_kbps', 'month', 'year']
    df_lean = df_raw[important_columns]

    # 2. CLEANING: Convert speed to Mbps immediately
    df_lean['speed_mbps'] = pd.to_numeric(df_lean['speed_kbps'], errors='coerce') / 1024
    df_lean = df_lean.drop(columns=['speed_kbps'])

    # 3. SAVE TO THE DATA FOLDER
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # Go up to main folder
    output_path = os.path.join(project_root, "data", "TRAI_Lean_Data_2025.csv")
    
    try:
        df_lean.to_csv(output_path, index=False)
        print("\n" + "="*30)
        print(f"✅ LEAN HARVEST COMPLETE!")
        print(f"📊 Total Rows: {len(df_lean)}")
        print(f"📂 Saved as: {output_path}")
        print("="*30)
    except Exception as e:
        print(f"Failed to save file: {e}")