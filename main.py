import pandas as pd

# Importing custom modules for data cleaning, dimensionality reduction, and modeling
from treat_df import clean_dataset
from dimensionality_reduction import reduce_dimensionality
from model import FNNAM

def main():
    """
    Main function to execute the data processing and model training workflow.

    This function performs the following steps:
    1. Reads an Excel file into a pandas DataFrame.
    2. Sorts the DataFrame based on the 'Transfer Date' column.
    3. Cleans the dataset using the `clean_dataset` function.
    4. Reduces the dimensionality of the dataset using the `reduce_dimensionality` function.
    5. Initializes the FNNAM model, trains it with the processed data, and evaluates it.

    The evaluation metrics of the model are printed at the end.
    """
    # Read the data from an Excel file
    df = pd.read_excel('original_xlsx/original.xlsx')
    
    # Sort the DataFrame by 'Transfer Date' in ascending order
    df = df.sort_values(by='Transfer Date', ascending=True)

    # Clean the dataset using the clean_dataset function from treat_df module
    df = clean_dataset(df)

    # Reduce the dimensionality of the dataset using reduce_dimensionality function
    df = reduce_dimensionality(df)

    # Initialize the FNNAM model
    model = FNNAM()

    # Train the model and get evaluation results
    results = model.get_trained_model(df, 'Transfer Value')

    # Print the evaluation metrics of the model
    print(f'The metrics of evaluation for the model are: {results}')

# Ensure that the main function runs only when the script is executed directly
if __name__ == '__main__':
    main()