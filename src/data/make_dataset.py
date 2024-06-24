import pandas as pd

def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    
    print("Printing first 5 rows:")
    print(df.head())
    print("Shape of dataset is:",df.shape)
    print("Check dtypes of dataset:")
    print(df.dtypes)
    return df