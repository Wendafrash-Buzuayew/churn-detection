import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the advanced feature extractor and the pipeline execution logic
from churn_advanced_features import extract_advanced_features
from churn_pipeline_v2_3 import run_pipeline  # Adjust if your script entry point differs

print("Step 1: Generating mock raw transactional data...")

# Create 500 fake subscribers (MSISDNs)
np.random.seed(42)
msisdns = [f"251911{str(i).zfill(6)}" for i in range(500)]

# Generate a list of raw voice calls over a 90-day period
call_records = []
base_date = datetime(2026, 3, 1)

for msisdn in msisdns:
    # Randomly assign a user type: 1 = healthy, 2 = frustrated/abrupt churner
    user_profile = np.random.choice([1, 2], p=[0.85, 0.15])
    num_calls = np.random.randint(30, 100) if user_profile == 1 else np.random.randint(10, 40)
    
    for _ in range(num_calls):
        # Frustrated users bunch short calls together (dropped call proxy)
        if user_profile == 2:
            duration = np.random.choice([2, 5, 8, 120], p=[0.4, 0.3, 0.2, 0.1])
            days_offset = np.random.randint(1, 45) # Churners stop calling early
        else:
            duration = np.random.randint(15, 300)
            days_offset = np.random.randint(1, 90)
            
        call_time = base_date + timedelta(days=days_offset, hours=np.random.randint(0, 24))
        call_records.append({"msisdn": msisdn, "timestamp": call_time, "duration_sec": duration})

df_calls_raw = pd.DataFrame(call_records)

# Generate a list of raw recharges
recharge_records = []
for msisdn in msisdns:
    user_profile = np.random.choice([1, 2], p=[0.85, 0.15])
    num_recharges = np.random.randint(8, 20) if user_profile == 1 else np.random.randint(2, 6)
    
    last_date = base_date
    for _ in range(num_recharges):
        # Erratic interval gap simulation
        interval = np.random.randint(4, 8) if user_profile == 1 else np.random.randint(1, 25)
        last_date += timedelta(days=int(interval))
        if last_date > base_date + timedelta(days=90):
            break
        amount = np.random.choice([50, 100, 200]) if user_profile == 1 else np.random.choice([5, 10, 25])
        recharge_records.append({"msisdn": msisdn, "timestamp": last_date, "amount": amount})

df_recharges_raw = pd.DataFrame(recharge_records)

# Save raw files to check inputs locally
df_calls_raw.to_csv("mock_raw_calls.csv", index=False)
df_recharges_raw.to_csv("mock_raw_recharges.csv", index=False)
print(" Raw transaction files saved to local storage.")

# ──────────────────────────────────────────────────────────────────────────────
print("\nStep 2: Engineering advanced proxy features...")
# Pass dataframes to your script's feature extraction functions
# Note: Ensure these function/argument names perfectly match your churn_advanced_features.py definitions
df_features = extract_advanced_features(df_calls_raw, df_recharges_raw)

# Generate dummy historical features that churn_pipeline_v2_3 expects
df_features['TENURE_DAYS'] = np.random.randint(30, 1500, size=len(df_features))
df_features['TOTAL_REVENUE_13W'] = np.random.uniform(10, 2000, size=len(df_features))
# Assign ground truth labels based on our generated profile logic
df_features['LABEL_CHURN_90D'] = np.where(df_features['msisdn'].isin(df_calls_raw[df_calls_raw['duration_sec'] < 10]['msisdn'].unique()) & (np.random.rand(len(df_features)) > 0.3), 1, 0)

df_features.to_csv("processed_features_for_pipeline.csv", index=False)
print(" Formatted dataset created with proxy metrics.")

# ──────────────────────────────────────────────────────────────────────────────
print("\nStep 3: Executing Churn Modeling Pipeline...")
# This invokes the dual-floor threshold model training
# Modify this function call parameters based on the precise signature in churn_pipeline_v2_3.py
run_pipeline(data_path="processed_features_for_pipeline.csv")