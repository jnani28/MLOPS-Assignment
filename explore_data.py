import pandas as pd

df = pd.read_csv(r'c:\Users\DLP-I516-75\OneDrive - Indian School of Business\Term 2\CT_bharani Sir\Assignment\bank-additional.csv', sep=',')
print('Shape:', df.shape)
print('\nColumns:', list(df.columns))
print('\nDtypes:')
print(df.dtypes)
print('\nFirst 3 rows:')
print(df.head(3).to_string())
print('\nTarget distribution:')
print(df['y'].value_counts())
print('\nMissing values:')
print(df.isnull().sum())
print('\nInfo about unknowns:')
for col in df.select_dtypes(include='object').columns:
    vals = df[col].unique()
    if 'unknown' in vals:
        count = (df[col] == 'unknown').sum()
        print(f'{col}: {count} unknowns')
print('\nUnique values per categorical column:')
for col in df.select_dtypes(include='object').columns:
    print(f'{col}: {df[col].unique()}')
