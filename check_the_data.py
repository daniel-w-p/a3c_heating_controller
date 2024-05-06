import pandas as pd
import numpy as np
import os

directory_path = 'data'

MIN_VALUE = -120
MAX_VALUE = 120

# find and check all csv files
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        print(f"File processing: {filename}")

        data = pd.read_csv(file_path)

        # check for NaN, inf, -inf
        problematic_values = data.isin([np.nan, np.inf, -np.inf])

        # check range
        out_of_range = (data < MIN_VALUE) | (data > MAX_VALUE)

        if problematic_values.any().any():
            print("Find problematic values:")
            # print problematic columns and rows
            for col in data.columns:
                if problematic_values[col].any():
                    print(f"  Column: {col}")
                    rows = problematic_values[col][problematic_values[col] == True].index.tolist()
                    print(f"    Rows: {rows}")
        else:
            print("There are no problematic values.")

        if out_of_range.any().any():
            print("Find out-of-range values:")
            for col in data.columns:
                if out_of_range[col].any():
                    print(f"  Column: {col}")
                    rows = out_of_range[col][out_of_range[col] == True].index.tolist()
                    print(f"    Row: {rows}")
        else:
            print("All values are in range [-120,120].")

        print("\n")
