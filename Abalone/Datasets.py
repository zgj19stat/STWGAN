import pandas as pd
from sklearn.model_selection import train_test_split

# 读入数据
data = pd.read_csv('abalone.csv')

# 将 sex 列转化为 0, 1, 2 编码
data['Sex'] = data['Sex'].map({'M': 0, 'F': 1, 'I': 2})

# 拆分成三个数据集，分别对应 sex 为 0, 1, 2 的情况
data_sex_0 = data[data['Sex'] == 0].drop(columns=['Sex'])
data_sex_1 = data[data['Sex'] == 1].drop(columns=['Sex'])
data_sex_2 = data[data['Sex'] == 2].drop(columns=['Sex'])

# 分别将三个数据集拆分为训练集和测试集
X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(data_sex_0.drop(columns=['Rings']), data_sex_0['Rings'], test_size=0.1, random_state=42)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(data_sex_1.drop(columns=['Rings']), data_sex_1['Rings'], test_size=0.1, random_state=42)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(data_sex_2.drop(columns=['Rings']), data_sex_2['Rings'], test_size=0.1, random_state=42)

# 保存处理后的数据
X_train_0.to_csv('X_train_sex_0.csv', index=False, header=False)
X_test_0.to_csv('X_test_sex_0.csv', index=False, header=False)
y_train_0.to_csv('y_train_sex_0.csv', index=False, header=False)
y_test_0.to_csv('y_test_sex_0.csv', index=False, header=False)

X_train_1.to_csv('X_train_sex_1.csv', index=False, header=False)
X_test_1.to_csv('X_test_sex_1.csv', index=False, header=False)
y_train_1.to_csv('y_train_sex_1.csv', index=False, header=False)
y_test_1.to_csv('y_test_sex_1.csv', index=False, header=False)

X_train_2.to_csv('X_train_sex_2.csv', index=False, header=False)
X_test_2.to_csv('X_test_sex_2.csv', index=False, header=False)
y_train_2.to_csv('y_train_sex_2.csv', index=False, header=False)
y_test_2.to_csv('y_test_sex_2.csv', index=False, header=False)

