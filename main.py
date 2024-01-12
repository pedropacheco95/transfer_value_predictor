import pandas as pd

from treat_df import clean_dataset
from dimensionality_reduction import reduce_dimensionality
from model import FNNAM

if __name__=='__main__':
    df = pd.read_excel('original_xlsx/original.xlsx')
    df = df.sort_values(by='Transfer Date', ascending=True)

    df = clean_dataset(df)
    df = reduce_dimensionality(df)

    model = FNNAM()
    results = model.get_trained_model(df,'Transfer Value')
    print(f'The metrics of evaluation for the model are: {results}')
    