import pandas as pd

pd.set_option('display.expand_frame_repr', False)
df_test = pd.read_csv('test_id.csv')
df_bandwidth = pd.read_csv('output_bandwidth.csv')
df_user = pd.read_csv('output_user.csv')

df_output = pd.merge(df_test, df_bandwidth, how='left', left_on=['SERVER_NAME', 'UPDATE_TIME', 'HOUR_ID'], right_on=['SERVER_NAME', 'UPDATE_TIME', 'HOUR_ID'])
df_output = df_output.drop(labels=['Unnamed: 0'], axis=1)
df_output = pd.merge(df_output, df_user, how='left', left_on=['SERVER_NAME', 'UPDATE_TIME', 'HOUR_ID'], right_on=['SERVER_NAME', 'UPDATE_TIME', 'HOUR_ID'])
df_output = df_output.drop(labels=['Unnamed: 0'], axis=1)
df_output.user = df_output.user.astype(int)
df_output = df_output.round({'bandwidth': 2})
df_output['label'] = df_output.bandwidth.astype(str) + ' ' + df_output.user.astype(str)
submission = df_output[['id', 'label']]
print(submission)
submission.to_csv('submission.csv', index=False)