import tensorflow as tf
import pandas as pd
import numpy as np

states = sorted(["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"])

def getDataset(data):
    data = np.expand_dims(data,axis=-1)
    final = np.empty((data.shape[0]*3,4,1))
    # final[::3] = data[:,:-3:3]/20000
    # final[1::3] = data[:,1:-3:3]/20000
    # final[2::3] = data[:,2:-3:3]/250
    final[::3] = data[:,3::3]/20000
    final[1::3] = data[:,4::3]/20000
    final[2::3] = data[:,5::3]/2500
    return final


model = tf.keras.models.load_model('model.h5')
print(model.summary())
df = pd.read_csv('counties.csv')
df.dropna(subset=[str(year)+p for year in [2004,2008,2012,2016] for p in ['D','R']],inplace=True)
data = df.drop(columns=['state','county'])
data = data.to_numpy(dtype=np.int64)
examples = data.shape[0]
ds = getDataset(data)
print('All data loaded')
predictions = model.predict(ds)
print('Finished predicting')
predictions[::3]*=20000
predictions[1::3]*=20000
predictions[2::3]*=2500
predictions = np.reshape(predictions,(examples,3))
new_df = pd.DataFrame(predictions,dtype=np.int64,columns=['D','R','O'])
new_df[new_df<0]=0
new_df.insert(0,'county',df['county'])
new_df.insert(0,'state',df['state'])
print('Loading to files')
new_df.to_csv('predictions.csv',index=False)
state_predictions = []
for state in states:
    rows = new_df[new_df['state']==state]
    state_predictions.append([rows['D'].sum(),rows['R'].sum(),rows['O'].sum()])
state_df = pd.DataFrame(data=state_predictions,columns=['D','R','O'])
state_df.insert(0,'State',states)
state_df.loc[-1]=['Total',state_df['D'].sum(),state_df['R'].sum(),state_df['O'].sum()]
state_df['T'] = state_df['D']+state_df['R']+state_df['O']
for p in ['D','R','O']:
    state_df[p+'%'] = np.round(state_df[p]/state_df['T']*100,2)
state_df.to_csv('states.csv',index=False)