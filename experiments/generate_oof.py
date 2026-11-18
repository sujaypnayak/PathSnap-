import pandas as pd
import json

print("Loading preprocessed data...")
df = pd.read_parquet("data/processed/candidates.parquet")
with open("data/processed/splits.json") as f:
    splits = json.load(f)

# Get one student trip ID
student_trip = splits["student_test"][0]

# Take 10 rows from the real training data, rename them to the student trip ID, and set label to None
dummy_st = df.sample(10).copy()
dummy_st["trip_id"] = student_trip
dummy_st["label"] = None 

# Save as a temporary file
dummy_st.to_parquet("data/processed/temp_student.parquet", index=False)
print("Created temp_student.parquet")
