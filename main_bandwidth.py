import pandas as pd
import numpy as np
from Loss import MAPE, sMAPE
import math
import matplotlib.pyplot as plt
from Naive24_bandwidth import Naive24_bandwidth


df = pd.read_csv('train-bandwidth_last6000_cut.csv')
server_list = list(df.SERVER_NAME.unique())[:]

predict_len = 744
train_len = 2000
loss_list = []

model = Naive24_bandwidth(seasonal=24)

list_day = ['10/3/2019', '11/3/2019', '12/3/2019', '13-03-19', '14-03-19', '15-03-19', '16-03-19', '17-03-19', '18-03-19',
            '19-03-19', '20-03-19', '21-03-19', '22-03-19', '23-03-19', '24-03-19', '25-03-19', '26-03-19', '27-03-19',
            '28-03-19', '29-03-19', '30-03-19', '31-03-19', '1/4/2019', '2/4/2019', '3/4/2019', '4/4/2019', '5/4/2019',
            '6/4/2019', '7/4/2019', '8/4/2019', '9/4/2019']

list_hour = [i for i in range(24)]

output = []
min_list = []
mean_list = []
high_server_list = []
too_high_server_list = ['SERVER_ZONE01_023']
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
    if 200 > loss > 90 and loss_mean > 90:
        high_server_list.append(server)
        loss_list.append(loss)
    elif loss > 200:
        too_high_server_list.append(server)
        loss_list.append(100)
    elif loss < 90 < loss_mean:
        min_list.append(server)
        loss_list.append(loss)
    elif loss > 90 > loss_mean:
        mean_list.append(server)
        loss_list.append(loss_mean)
    elif loss < loss_mean < 90:
        mean_list.append(server)
        loss_list.append(loss_mean)
    else:
        loss_list.append(loss_mean)
    print(server, loss, loss_mean)

# print(min_list)
# print(high_server_list)
# print(too_high_server_list)
# print(np.mean(loss_list))
# print('Done!')

for server in server_list:
    print(server)
    data_cur = data_dict[server]
    if server in too_high_server_list:
        prediction = [0] * predict_len
    elif server in high_server_list:
        train = data_cur[-train_len:]
        remainder_train = 24 - len(train) % 24
        prediction = model.forward_min(train, predict_len)
        for i in range(predict_len):
            if prediction[i] < 10:
                prediction[i] = 0
            remainder_cur = (train_len + i) % 24
            if remainder_cur in [(3 + remainder_train) % 24, (4 + remainder_train) % 24, (5 + remainder_train) % 24]:
                prediction[i] = 1.5 * prediction[i]
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
df_output.columns = ['SERVER_NAME', 'UPDATE_TIME', 'HOUR_ID', 'bandwidth']
df_output.to_csv('output_bandwidth.csv')
