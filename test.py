import pandas as pd



# def test(df):
#     filt = (df['time'] == 60) & (df['start'] == 6081092172) & (df['end'] == 5551903399)
#     print(df[filt])
#
#
#
# df = pd.read_csv('./data/node_data.csv')
#
# # print(df)
#
# # filt = (df['time'] == 60) & (df['start'] == 6081092172) & (df['end'] == 5551903399)
# # print(df[filt])
#
# test(df)

# df = pd.DataFrame([' ])

df = pd.DataFrame({'name': ['Alice','Bob','Charlie','Dave','Ellen','Frank'],
                   'age': [24,42,18,68,24,30],
                   'state': ['NY','CA','CA','TX','CA','NY'],
                   'point': [64,24,70,70,88,57]}
                  )

filt = (df['name'] == 'Alice') & (df['age'] == 24)
df[filt].replace()
print(df[filt])