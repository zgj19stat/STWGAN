import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def abalone_data(sources=2, source_num=10000, target_num=10000, valid_num=2000, args=None):
    # 读取数据
    X_train_sex_0 = pd.read_csv('X_train_sex_0.csv', header=None)
    X_train_sex_1 = pd.read_csv('X_train_sex_1.csv', header=None)
    X_train_sex_2 = pd.read_csv('X_train_sex_2.csv', header=None)

    y_train_sex_0 = pd.read_csv('y_train_sex_0.csv', header=None)
    y_train_sex_1 = pd.read_csv('y_train_sex_1.csv', header=None)
    y_train_sex_2 = pd.read_csv('y_train_sex_2.csv', header=None)

    X_test_sex_2 = pd.read_csv('X_test_sex_2.csv', header=None)
    y_test_sex_2 = pd.read_csv('y_test_sex_2.csv', header=None)

    sources_X = torch.zeros((sources, source_num, args.d))
    sources_Y = torch.zeros((sources, source_num, args.q))
    sources_Eta = torch.zeros((sources, source_num, args.m))

    # 对于 sex=0 和 sex=1 的数据集赋值，只取前1000个样本
    sources_X[0] = torch.tensor(X_train_sex_0.iloc[:source_num].values, dtype=torch.float32)
    sources_X[1] = torch.tensor(X_train_sex_1.iloc[:source_num].values, dtype=torch.float32)

    sources_Y[0] = torch.tensor(y_train_sex_0.iloc[:source_num].values, dtype=torch.float32).view(-1, 1)
    sources_Y[1] = torch.tensor(y_train_sex_1.iloc[:source_num].values, dtype=torch.float32).view(-1, 1)

    # Eta 是从 N(0, 1) 分布中采样
    sources_Eta[0] = torch.tensor(np.random.normal(0, 1, (source_num, args.m)), dtype=torch.float32)
    sources_Eta[1] = torch.tensor(np.random.normal(0, 1, (source_num, args.m)), dtype=torch.float32)

    if args.MSSG:
        regressor_f1 = LinearRegression()
        regressor_f2 = LinearRegression()

        f1_input = sources_X[0][:, args.share]  # 1000x1
        f1_output = sources_X[0][:, args.S1hidd]  # 1000x1
        regressor_f1.fit(f1_input.detach().numpy(), f1_output.detach().numpy())  

        f2_input = sources_X[1][:, args.share]  # 1000x1
        f2_output = sources_X[1][:, args.S2hidd]  # 1000x1
        regressor_f2.fit(f2_input.detach().numpy(), f2_output.detach().numpy())  

        sources_X[0][:, args.S1hidd] = torch.tensor(regressor_f1.predict(f1_input.detach().numpy()), dtype=torch.float32)
        sources_X[1][:, args.S2hidd] = torch.tensor(regressor_f2.predict(f2_input.detach().numpy()), dtype=torch.float32)

    target_X = torch.tensor(X_train_sex_2.iloc[:target_num].values, dtype=torch.float32)
    target_Y = torch.tensor(y_train_sex_2.iloc[:target_num].values, dtype=torch.float32).view(-1, 1)

    target_Eta = torch.tensor(np.random.normal(0, 1, (target_num, args.m)), dtype=torch.float32)

    test_X = torch.tensor(X_test_sex_2.iloc[:valid_num].values, dtype=torch.float32)
    test_Y = torch.tensor(y_test_sex_2.iloc[:valid_num].values, dtype=torch.float32).view(-1, 1)

    test_Eta = torch.tensor(np.random.normal(0, 1, (args.j, args.m)), dtype=torch.float32)

    return [sources_X, sources_Y, sources_Eta, target_X, target_Y, target_Eta], [test_X, test_Y], test_Eta
