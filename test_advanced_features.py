import pandas as pd
import numpy as np
import logging

# Setup basic logging to see the outputs clearly
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-6s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("MockTest")

# 1. Simulate a mock subscriber dataset matching your CSV's exact weekly prefix schema
log.info("Generating mock subscriber dataset containing sequential weekly metrics...")
n_subs = 10

data = {
    "MSISDN": [f"251911{str(i).zfill(6)}" for i in range(n_subs)],
    "SNAPSHOT_DATE": ["2026-01-15"] * n_subs,
    "AON": np.random.randint(30, 1500, size=n_subs),
    "DATASET_TYPE": ["TRAIN"] * 7 + ["TEST"] * 3,
    "LABEL_CHURN_90D": [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
}

# Add sequential weekly data (W10, W11, W12, W13) for the engineer to track trends
# We will deliberately build high usage collapse in sub 0 and 2 to test the math
for i, prefix in enumerate(["DATA_MB", "OG_VOICE_MIN", "TOTAL_SMS_COUNT", "BUNDLE_CNT", "TOTAL_REVENUE"]):
    data[f"{prefix}_W10"] = [5000, 200, 30, 4, 150] * 2
    data[f"{prefix}_W11"] = [3000, 150, 20, 3, 100] * 2
    data[f"{prefix}_W12"] = [800,  40,  5,  1, 30]  * 2
    data[f"{prefix}_W13"] = [0,    0,   0,  0, 0]   * 2 # Dropped to zero usage in W13!

raw_df = pd.DataFrame(data)

# 2. Simulate the secondary raw recharge ledger with timestamps 
log.info("Generating secondary raw recharge transactional table...")
recharge_records = []
for sub in data["MSISDN"]:
    # Add multiple timestamped recharges per subscriber to compute variance (CV)
    recharge_records.append({"MSISDN": sub, "RECHARGE_DATE": "2025-12-01 10:30:00", "RECHARGE_AMT": 50})
    recharge_records.append({"MSISDN": sub, "RECHARGE_DATE": "2025-12-08 14:15:00", "RECHARGE_AMT": 40})
    recharge_records.append({"MSISDN": sub, "RECHARGE_DATE": "2025-12-22 09:00:00", "RECHARGE_AMT": 15}) # Irregular interval

recharge_df = pd.DataFrame(recharge_records)

# 3. Trigger the Drop-in Advanced Feature Engineer using BOTH tables
log.info("Passing mock datasets into AdvancedFeatureEngineer...")
try:
    from churn_advanced_features import AdvancedFeatureEngineer
    
    # Passing both data frames into the orchestrator initialization step
    enriched_df = AdvancedFeatureEngineer(df=raw_df, recharge_df=recharge_df).build_all()
    
    print("\n" + "="*70)
    print(" 🎉 ENRICHMENT SUCCESSFUL!")
    print("="*70)
    print(f"Original shape : {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
    print(f"Enriched shape : {enriched_df.shape[0]} rows × {enriched_df.shape[1]} columns")
    
    # Print a list of some of the high-signal features that were successfully computed
    new_cols = [c for c in enriched_df.columns if c not in raw_df.columns]
    print(f"\nAdded {len(new_cols)} Advanced Features:")
    for col in new_cols[:10]:
        print(f"  • {col}")
    print("="*70)
    
except ImportError:
    log.error("Could not import AdvancedFeatureEngineer. Check file names in folder.")