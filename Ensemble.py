import pandas as pd
import numpy as np

df_1 = pd.read_csv('submission10.csv')
df_2 = pd.read_csv('submission12.csv')

loss = [70.83141, 70.96506]
loss_inverse = [np.exp(1/item) for item in loss]
sum_loss_inverse = np.sum(loss_inverse)
weights = [loss_inverse[i] / sum_loss_inverse for i in range(len(loss_inverse))]

print(weights)

df_1['bandwidth'], df_1['user'] = df_1['label'].str.split(' ', 1).str
df_1.bandwidth = df_1.bandwidth.astype(float)
df_1.user = df_1.user.astype(float)

df_2['bandwidth'], df_2['user'] = df_2['label'].str.split(' ', 1).str
df_2.bandwidth = df_2.bandwidth.astype(float)
df_2.user = df_2.user.astype(float)

print(df_1)
print(df_2)

df_ensemble = pd.DataFrame()
df_ensemble['id'] = df_1.id
df_ensemble['bandwidth'] = weights[0] * df_1.bandwidth + weights[1] * df_2.bandwidth
df_ensemble['user'] = (weights[0] * df_1.user + weights[1] * df_2.user).astype(int)
df_ensemble = df_ensemble.round({'bandwidth': 2})
df_ensemble['label'] = df_ensemble.bandwidth.astype(str) + ' ' + df_ensemble.user.astype(str)
print(df_ensemble)

df_ensemble[['id', 'label']].to_csv('submission_ensemble.csv', index=False)
