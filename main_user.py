from Naive24 import Naive24
import pandas as pd
import numpy as np
from Loss import MAPE, sMAPE
import math
import matplotlib.pyplot as plt
from Naive24_user import Naive24_user


df = pd.read_csv('train-user_last6000_cut.csv')
server_list = list(df.SERVER_NAME.unique())[:]

predict_len = 744
train_len = 1000
loss_list = []

model = Naive24_user(seasonal=24)

list_day = ['10/3/2019', '11/3/2019', '12/3/2019', '13-03-19', '14-03-19', '15-03-19', '16-03-19', '17-03-19', '18-03-19',
            '19-03-19', '20-03-19', '21-03-19', '22-03-19', '23-03-19', '24-03-19', '25-03-19', '26-03-19', '27-03-19',
            '28-03-19', '29-03-19', '30-03-19', '31-03-19', '1/4/2019', '2/4/2019', '3/4/2019', '4/4/2019', '5/4/2019',
            '6/4/2019', '7/4/2019', '8/4/2019', '9/4/2019']

list_hour = [i for i in range(24)]

output = []
min_list = []
high_server_list = ['SERVER_ZONE01_023']
data_dict = {}

for server in server_list:
    data_cur = df[df.SERVER_NAME == server]
    for i in range(2000, data_cur.shape[1]):
        if math.isnan(data_cur.iloc[0, i]):
            data_cur.iloc[0, i] = data_cur.iloc[0, i - 24]
    data_cur = data_cur.values.tolist()[0][2:]
    data_dict[server] = data_cur
    train = data_cur[-train_len-predict_len:-predict_len]
    test = data_cur[-predict_len:]
    if server in high_server_list:
        prediction = [np.min(train)] * predict_len
        prediction_mean = [np.min(train)] * predict_len
    else:
        prediction = model.forward_min(train, predict_len)
        prediction_mean = model.forward_mean(train, predict_len)

    loss = MAPE(test, prediction)
    loss_mean = MAPE(test, prediction_mean)
    if loss > 85 and loss_mean > 85:
        high_server_list.append(server)
        loss_list.append(100)
    elif loss < 85 < loss_mean:
        min_list.append(server)
        loss_list.append(loss)
    else:
        loss_list.append(loss_mean)

    print(server, loss, loss_mean)

print(high_server_list)
print(np.mean(loss_list))
print('Done!')

for server in server_list:
    print(server)
    data_cur = data_dict[server]
    if server in high_server_list:
        prediction = [0] * predict_len
    elif server in min_list:
        train = data_cur[-train_len:]
        prediction = model.forward_min(train, predict_len)
    else:
        train = data_cur[-train_len:]
        prediction = model.forward_mean(train, predict_len)
    for i in range(predict_len):
        quotient = int(i/24)
        remainder = i % 24
        output.append([server, list_day[quotient], remainder, prediction[i]])

df_output = pd.DataFrame(output)
df_output.columns = ['SERVER_NAME', 'UPDATE_TIME', 'HOUR_ID', 'user']
df_output.to_csv('output_user.csv')
