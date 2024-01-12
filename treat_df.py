import pandas as pd

def remove_rows_with_nan(df, columns):
    return df.dropna(subset=columns)

def remove_columns_with_many_nans(df,nan_threshold=0.25):
    nan_percentage = df.isna().sum() / len(df)
    columns_to_drop = nan_percentage[nan_percentage > nan_threshold].index
    return df.drop(columns=columns_to_drop)

def replace_nan_with_average(df,column_name):
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    average = df[column_name].mean()
    df[column_name].fillna(average, inplace=True)
    return df

def replace_nan_with_min(df,column_name):
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    minimum = df[column_name].min()
    df[column_name].fillna(minimum, inplace=True)
    return df

def remove_missing_data_columns(df, missing_threshold=0.25):
    missing_percent = df.isnull().mean()
    columns_to_drop = missing_percent[missing_percent > missing_threshold].index
    df.drop(columns_to_drop, axis=1)
    return df

def replace_columns_with_average(df,columns):
    for column in columns:
        df = replace_nan_with_average(df,column)
    return df

def replace_columns_with_min(df,columns):
    for column in columns:
        df = replace_nan_with_min(df,column)
    return df

def clean_dataset(df):

    df = remove_rows_with_nan(df,df.columns)
    df = remove_columns_with_many_nans(df,nan_threshold=0)

    return df