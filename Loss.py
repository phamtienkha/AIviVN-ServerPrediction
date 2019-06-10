def sMAPE(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    loss = 0
    for i in range(len(y_true)):
        loss += 200 * abs(y_true[i] - y_pred[i]) / (abs(y_true[i]) + abs(y_pred[i]))
    return loss / len(y_true)


def MAPE(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    loss = 0
    for i in range(len(y_true)):
        loss += 100 * abs(y_true[i] - y_pred[i]) / (abs(y_true[i]) + 1e-6)
    return loss / len(y_true)
