import pandas as pd

# Change the file location before running code.
data_2018 = "C:/Users/Prin/Desktop/dissertation/Datashare/juno_2018.csv"
data_2019 = "C:/Users/Prin/Desktop/dissertation/Datashare/juno_2019.csv"

df_2018 = pd.read_csv(data_2018, encoding='latin1')
df_2019 = pd.read_csv(data_2019)

# Drop the error rows
df_2018 = df_2018.drop(df_2018.index[1112133:1112150])

# Concatenate 2018 and 2019 dataset
df = pd.concat([df_2018, df_2019], ignore_index=True)

# Split Date and Time
df[['Date', 'Time']] = df.TIMESTAMP.str.split(" ", expand=True)

# Add O'clcok column
df['Oclock'] = df.Time.str.split(":",expand=True)[0]

# Generate numeric value for IN and OUT direction.
inn = []
out = []
for d in df['Direction']:
    # For IN column, count 1 if direction is 'I' and count 0 if direction is 'O'.
    if d == 'I':
        inn.append(1)
        out.append(0)
    # For OUT column, count 0 if direction is 'I' and count 1 if direction is 'O'.
    elif d == 'O':
        inn.append(0)
        out.append(1)

# Add IN and OUT column to dataframe.
df['IN'] = inn
df['OUT'] = out

