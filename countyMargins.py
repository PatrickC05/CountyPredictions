import pandas as pd

pd.options.display.max_columns = None
df = pd.read_csv('countypres_2000-2016.csv')
print('File loaded')
years = [str(year)+p for year in range(2000,2020,4) for p in ['D','R','O']]
columns = ['state','county','2000G'] + years
counties_df = pd.DataFrame(columns = columns)
stateToA = {'Alaska': 'AK', 'Maine': 'ME', 'Connecticut': 'CT', 'Rhode Island':'RI'}
for year,full_name,state,county,party,votes in zip(df['year'],df['state'],df['state_po'],df['county'],df['party'],df['candidatevotes']):
    if pd.isnull(state):
        state = stateToA[full_name]
    row = counties_df[(counties_df['state']==state)&(counties_df['county'] == county)]
    if party == 'democrat':
        s = str(year)+'D'
    elif party == 'republican':
        s = str(year)+'R'
    elif party == 'green':
        s = '2000G'
    else:
        s = str(year)+'O'
    if row.empty:
        counties_df = counties_df.append({'state':state,'county':county,s:votes},ignore_index=True)
    else:
        ind = row.index[0]
        counties_df.at[ind,s] = votes
print('Initial Data loaded')
counties_df = counties_df.dropna(axis=0,how='all',subset=years)
counties_df['2000O'] += counties_df['2000G']
counties_df = counties_df.drop(columns=['2000G'])
print('Saving')
counties_df.to_csv('counties.csv',index=False)
# print(counties_df)
