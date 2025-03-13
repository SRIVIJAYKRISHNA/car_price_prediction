import pandas as pd
import json

# Load the dataset
df = pd.read_csv("D:\\Projects\\Linear Regression\\Dataset\\cars_dataset.csv")

# Convert column names to lowercase for consistency
df.columns = df.columns.str.lower()

# Ensure 'make' and 'model' columns exist
if "make" in df.columns and "model" in df.columns:
    # Create a dictionary to store make-model mapping
    car_data = {}

    # Get unique makes
    unique_makes = df["make"].unique()

    for make in unique_makes:
        # Get unique models for each make
        models = df[df["make"].str.lower() == make.lower()]["model"].unique().tolist()
        car_data[make] = models

    # Save data to JSON for easy UI integration
    with open("makes_models.json", "w") as json_file:
        json.dump(car_data, json_file, indent=4)

    print("✅ Car makes and models saved to 'makes_models.json'")
else:
    print("❌ Error: 'make' or 'model' column not found in dataset!")
