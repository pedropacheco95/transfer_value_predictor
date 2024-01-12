import pandas as pd

def remove_rows_with_nan(df, columns):
    """
    Remove rows with NaN values in specific columns.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.
    columns (list): List of column names to check for NaN values.

    Returns:
    pandas.DataFrame: The DataFrame with rows containing NaN values in specified columns removed.
    """
    return df.dropna(subset=columns)

def remove_columns_with_many_nans(df, nan_threshold=0.25):
    """
    Remove columns from the DataFrame that have a high percentage of NaN values.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.
    nan_threshold (float): The threshold percentage of NaN values in a column for it to be removed.

    Returns:
    pandas.DataFrame: The DataFrame with columns having NaN values above the threshold removed.
    """
    nan_percentage = df.isna().sum() / len(df)
    columns_to_drop = nan_percentage[nan_percentage > nan_threshold].index
    return df.drop(columns=columns_to_drop)

def replace_nan_with_average(df, column_name):
    """
    Replace NaN values in a specified column with the column's average.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.
    column_name (str): The name of the column for NaN replacement.

    Returns:
    pandas.DataFrame: The DataFrame with NaN values in the specified column replaced by the average.
    """
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    average = df[column_name].mean()
    df[column_name].fillna(average, inplace=True)
    return df

def replace_nan_with_min(df, column_name):
    """
    Replace NaN values in a specified column with the column's minimum value.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.
    column_name (str): The name of the column for NaN replacement.

    Returns:
    pandas.DataFrame: The DataFrame with NaN values in the specified column replaced by the minimum value.
    """
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    minimum = df[column_name].min()
    df[column_name].fillna(minimum, inplace=True)
    return df

def replace_columns_with_average(df, columns):
    """
    Replace NaN values in specified columns with their respective averages.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.
    columns (list of str): List of column names for NaN replacement with average.

    Returns:
    pandas.DataFrame: The DataFrame with NaN values in specified columns replaced by their averages.
    """
    for column in columns:
        df = replace_nan_with_average(df, column)
    return df

def replace_columns_with_min(df, columns):
    """
    Replace NaN values in specified columns with their respective minimum values.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.
    columns (list of str): List of column names for NaN replacement with minimum values.

    Returns:
    pandas.DataFrame: The DataFrame with NaN values in specified columns replaced by their minimum values.
    """
    for column in columns:
        df = replace_nan_with_min(df, column)
    return df

def clean_dataset(df):
    """
    Clean the dataset by removing rows and columns with NaN values and replacing NaNs in remaining columns.

    Parameters:
    df (pandas.DataFrame): The DataFrame to clean.

    Returns:
    pandas.DataFrame: The cleaned DataFrame.
    """
    df = remove_rows_with_nan(df, df.columns)
    df = remove_columns_with_many_nans(df, nan_threshold=0)
    return df