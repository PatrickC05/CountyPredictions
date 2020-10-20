import pandas as pd
states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
df = pd.read_csv('predictions.csv')
state_predictions = []
for state in states:
    rows = df[df['state']==state]
    state_predictions.append([rows['2020D'].sum(),rows['2020R'].sum(),rows['2020O'].sum()])
state_df = pd.DataFrame(data=state_predictions,columns=['D','R','O'])
state_df.insert(0,'State',states)
state_df.loc[-1]=['Total',state_df['D'].sum(),state_df['R'].sum(),state_df['O'].sum()]
state_df['T'] = state_df['D']+state_df['R']+state_df['O']
for p in ['D','R','O']:
    state_df[p+'%'] = state_df[p]/state_df['T']*100
state_df.to_csv('states.csv',index=False)