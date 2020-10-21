import tensorflow as tf
import pandas as pd
import numpy as np

def getDataset(data):
    data = np.expand_dims(data,axis=-1)
    final = np.empty((data.shape[0]*3,4,1))
    final[::3] = data[:,3::3]/20000
    final[1::3] = data[:,4::3]/20000
    final[2::3] = data[:,5::3]/250
    return final


model = tf.keras.models.load_model('model.h5')
print(model.summary())
df = pd.read_csv('counties.csv')
data = df.drop(columns=['state','county'])
data = data.to_numpy(dtype=np.int64)
examples = data.shape[0]
ds = getDataset(data)
print('All data loaded')
predictions = model.predict(ds)
print('Finished predicting')
predictions[::3]*=20000
predictions[1::3]*=20000
predictions[2::3]*=250
predictions = np.reshape(predictions,(examples,12))
new_df = pd.DataFrame(predictions,dtype=np.int64,columns=[str(year)+s for s in ['D','R','O'] for year in range(2008,2024,4)])
new_df[new_df<0]=0
new_df.insert(0,'county',df['county'])
new_df.insert(0,'state',df['state'])
print('Loading to file')
new_df.to_csv('predictions.csv',index=False)