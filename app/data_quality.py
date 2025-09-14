import pandas as pd

def check_missing(data):
    return data.isnull().sum().to_dict()

def check_duplicates(data):
    return data.duplicated().sum()

def check_schema(data):
    return {col: str(dtype) for col, dtype in data.dtypes.items()}
