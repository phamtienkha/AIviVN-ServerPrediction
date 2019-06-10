import numpy as np
import pandas as pd
import math
from Loss import MAPE
import matplotlib.pyplot as plt


class Naive24_user():

    def __init__(self, seasonal):
        self.seasonal = seasonal

    def forward_min(self, data, predict_len):
        min_list = [0] * self.seasonal
        data_len = len(data)
        for remainder in range(24):
            min_list[remainder] = np.min([data[i] for i in range(data_len) if i % 24 == remainder])

        prediction = [0] * predict_len
        for i in range(predict_len):
            remainder_cur = (data_len + i) % 24
            # prediction[i] = min_list[remainder_cur]
            # if remainder_cur in [3 + remainder_data, 4 + remainder_data, 5 + remainder_data]:
            if remainder_cur in [19, 20, 21]:
                prediction[i] = 1.5 * min_list[remainder_cur]
            else:
                prediction[i] = min_list[remainder_cur]

        return prediction

    def forward_mean(self, data, predict_len):
        min_list = [0] * self.seasonal
        data_len = len(data)
        remainder_data = self.seasonal - data_len % self.seasonal
        convert_list = [0] * self.seasonal
        for i in range(self.seasonal):
            convert_list[i] = (i - remainder_data) % self.seasonal
        for remainder in range(24):
            if remainder in [convert_list[i] for i in [15, 16, 17, 18, 19, 20, 21, 22, 6, 7, 8, 9, 10]]:
                noise = np.random.normal(0.9, 0.1)
                min_list[remainder] = int(np.quantile([data[i] for i in range(data_len) if int(i % 24) == remainder], 0.45))
            else:
                noise = np.random.normal(1.2, 0.1)
                min_list[remainder] = int(np.quantile([data[i] for i in range(data_len) if int(i % 24) == remainder], 0.05))

        prediction = [0] * predict_len
        for i in range(predict_len):
            remainder_cur = (data_len + i) % 24
            # prediction[i] = min_list[remainder_cur]
            # if remainder_cur in [3 + remainder_data, 4 + remainder_data, 5 + remainder_data]:
            if remainder_cur in [19, 20, 21]:
                prediction[i] = 1.5 * min_list[remainder_cur]
            else:
                prediction[i] = min_list[remainder_cur]

        return prediction
