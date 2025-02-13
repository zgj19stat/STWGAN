import torch

def synthetic_data_M1(sources=5, source_num=10000, target_num=10000, valid_num=2000, args=None):
    num = sources * source_num + target_num
    X, Eta = torch.randn(num, args.d), torch.randn(num, args.m)
    # synthetic source data: 0:source_num for source; source_num:num for target
    # target_X
    X[sources * source_num:num,0] = X[sources * source_num:num,0] + 2.
    X[sources * source_num:num,1] = X[sources * source_num:num,1] + 1.
    # source_1_X vaild (0,0,0,0,0,...)
    # source_2_X invaild
    X[source_num:2*source_num,:] = X[source_num:2*source_num,:] + 5.
    # source_3_X invaild 
    X[2*source_num:3*source_num,:] = X[2*source_num:3*source_num,:] - 5.
    # source_4_X invaild Y|X shift
    X[3*source_num:4*source_num,0] = X[3*source_num:4*source_num,0] + 2.
    X[3*source_num:4*source_num,1] = X[3*source_num:4*source_num,1] + 1.
    # source_5_X invaild Y|X shift 
    X[4*source_num:5*source_num,0] = X[4*source_num:5*source_num,0] + 2.
    X[4*source_num:5*source_num,1] = X[4*source_num:5*source_num,1] + 1.
    
    # valid's distribution as target's
    valid_X, valid_Eta = torch.randn(valid_num, args.d), torch.randn(args.j, args.m)
    valid_X[:,0] = valid_X[:,0] + 2.
    valid_X[:,1] = valid_X[:,1] + 1.
    test_X, test_Eta = torch.randn(valid_num, args.d), torch.randn(args.j, args.m)
    test_X[:,0] = test_X[:,0] + 2.
    test_X[:,1] = test_X[:,1] + 1.
    
    eps = torch.randn(num, 1) * X[:,0].view(-1,1)
    # source_Y and target_Y
    Y = X[:,0].view(-1,1) + torch.exp(X[:,1].view(-1,1)+X[:,2].view(-1,1)/3) + torch.sin(X[:,3].view(-1,1)+X[:,4].view(-1,1)) + eps
    # source_4_Y invaild
    Y[3*source_num:4*source_num] = X[3*source_num:4*source_num,0].view(-1,1)*5 +\
        torch.exp(X[3*source_num:4*source_num,1].view(-1,1)+X[3*source_num:4*source_num,2].view(-1,1)/3 + 2) + torch.cos(X[3*source_num:4*source_num,3].view(-1,1)+X[3*source_num:4*source_num,4].view(-1,1)) + eps[3*source_num:4*source_num,0].view(-1,1) + 5
    # source_5_Y invaild
    Y[4*source_num:5*source_num] = X[4*source_num:5*source_num,0].view(-1,1)/5 +\
        torch.exp(X[4*source_num:5*source_num,1].view(-1,1)+X[4*source_num:5*source_num,2].view(-1,1)/3 - 2) + torch.cos(X[4*source_num:5*source_num,3].view(-1,1)+X[4*source_num:5*source_num,4].view(-1,1)) + eps[4*source_num:5*source_num,0].view(-1,1) - 5
    
    valid_Y_mean = valid_X[:,0].view(-1,1) + torch.exp(valid_X[:,1].view(-1,1)+valid_X[:,2].view(-1,1)/3) + torch.sin(valid_X[:,3].view(-1,1)+valid_X[:,4].view(-1,1))
    valid_Y_sd = torch.ones(valid_num, 1) * torch.abs(valid_X[:,0].view(-1,1))
    test_Y_mean = test_X[:,0].view(-1,1) + torch.exp(test_X[:,1].view(-1,1)+test_X[:,2].view(-1,1)/3) + torch.sin(test_X[:,3].view(-1,1)+test_X[:,4].view(-1,1))
    test_Y_sd = torch.ones(valid_num, 1) * torch.abs(test_X[:,0].view(-1,1))

    sources_X = torch.zeros(sources, source_num, args.d)
    sources_Y = torch.zeros(sources, source_num, args.q)
    sources_Eta = torch.zeros(sources, source_num, args.m)
    for i in range(sources):
        sources_X[i] = X[i*source_num:(i+1)*source_num]
        sources_Y[i] = Y[i*source_num:(i+1)*source_num]
        sources_Eta[i] = Eta[i*source_num:(i+1)*source_num]

    
    return [sources_X, sources_Y, sources_Eta, X[sources * source_num:num], Y[sources * source_num:num], Eta[sources * source_num:num]], \
           [valid_X, valid_Y_mean, valid_Y_sd], valid_Eta, [test_X, test_Y_mean, test_Y_sd], test_Eta

def synthetic_data_M2(sources=5, source_num=10000, target_num=10000, valid_num=2000, args=None):
    num = sources * source_num + target_num
    X, Eta = torch.randn(num, args.d), torch.randn(num, args.m)
    # synthetic source data: 0:source_num for source; source_num:num for target
    # target_X
    X[sources * source_num:num,0] = X[sources * source_num:num,0] + 2.
    X[sources * source_num:num,1] = X[sources * source_num:num,1] + 1.
    # source_1_X vaild (0,0,0,0,0,...)
    # source_2_X invaild
    X[source_num:2*source_num,:] = X[source_num:2*source_num,:] + 5.
    # source_3_X invaild 
    X[2*source_num:3*source_num,:] = X[2*source_num:3*source_num,:] - 5.
    # source_4_X invaild Y|X shift
    X[3*source_num:4*source_num,0] = X[3*source_num:4*source_num,0] + 2.
    X[3*source_num:4*source_num,1] = X[3*source_num:4*source_num,1] + 1.
    # source_5_X invaild Y|X shift 
    X[4*source_num:5*source_num,0] = X[4*source_num:5*source_num,0] + 2.
    X[4*source_num:5*source_num,1] = X[4*source_num:5*source_num,1] + 1.
    
    # test's distribution as target's
    valid_X, valid_Eta = torch.randn(valid_num, args.d), torch.randn(args.j, args.m)
    valid_X[:,0] = valid_X[:,0] + 2.
    valid_X[:,1] = valid_X[:,1] + 1.
    test_X, test_Eta = torch.randn(valid_num, args.d), torch.randn(args.j, args.m)
    test_X[:,0] = test_X[:,0] + 2.
    test_X[:,1] = test_X[:,1] + 1.
    
    eps = torch.randn(num, 1) + X[:,2].view(-1,1)
    # source_Y and target_Y
    Y = (2 + X[:,0].view(-1,1)**2/3 + X[:,1].view(-1,1)**2 + X[:,2].view(-1,1)**2 + X[:,3].view(-1,1)**2 + X[:,4].view(-1,1)**2)/3. * eps
    # source_4_Y invaild
    Y[3*source_num:4*source_num] = (7+(X[3*source_num:4*source_num,0].view(-1,1)**2)**3/3 + X[3*source_num:4*source_num,1].view(-1,1)**3 + X[3*source_num:4*source_num,2].view(-1,1)**3 + X[3*source_num:4*source_num,3].view(-1,1)**3 + X[3*source_num:4*source_num,4].view(-1,1)**3) * eps[3*source_num:4*source_num].view(-1,1) + 5
    # source_5_Y invaild
    Y[4*source_num:5*source_num] = (-3+(X[4*source_num:5*source_num,0].view(-1,1)**2) + X[4*source_num:5*source_num,1].view(-1,1) + X[4*source_num:5*source_num,2].view(-1,1) + X[4*source_num:5*source_num,3].view(-1,1) + X[4*source_num:5*source_num,4].view(-1,1)) * eps[4*source_num:5*source_num].view(-1,1) - 5
    
    valid_Y_mean = valid_X[:,2].view(-1,1)
    valid_Y_sd = (2 + valid_X[:,0].view(-1,1)**2/3 + valid_X[:,1].view(-1,1)**2 + valid_X[:,2].view(-1,1)**2 + valid_X[:,3].view(-1,1)**2 + valid_X[:,4].view(-1,1)**2)/3.
    test_Y_mean = test_X[:,2].view(-1,1)
    test_Y_sd = (2 + test_X[:,0].view(-1,1)**2/3 + test_X[:,1].view(-1,1)**2 + test_X[:,2].view(-1,1)**2 + test_X[:,3].view(-1,1)**2 + test_X[:,4].view(-1,1)**2)/3.
    
    sources_X = torch.zeros(sources, source_num, args.d)
    sources_Y = torch.zeros(sources, source_num, args.q)
    sources_Eta = torch.zeros(sources, source_num, args.m)
    for i in range(sources):
        sources_X[i] = X[i*source_num:(i+1)*source_num]
        sources_Y[i] = Y[i*source_num:(i+1)*source_num]
        sources_Eta[i] = Eta[i*source_num:(i+1)*source_num]
    
    return [sources_X, sources_Y, sources_Eta, X[sources * source_num:num], Y[sources * source_num:num], Eta[sources * source_num:num]], \
           [valid_X, valid_Y_mean, valid_Y_sd], valid_Eta, [test_X, test_Y_mean, test_Y_sd], test_Eta

def synthetic_data_M3(sources=5, source_num=10000, target_num=10000, valid_num=2000, args=None):
    num = sources * source_num + target_num
    X, Eta = torch.randn(num, args.d), torch.randn(num, args.m)
    # synthetic source data: 0:source_num for source; source_num:num for target
    # target_X
    X[sources * source_num:num,0] = X[sources * source_num:num,0] + 2.
    X[sources * source_num:num,1] = X[sources * source_num:num,1] + 1.
    # source_1_X vaild (0,0,0,0,0,...)
    # source_2_X invaild
    X[source_num:2*source_num,:] = X[source_num:2*source_num,:] + 10.
    # source_3_X invaild 
    X[2*source_num:3*source_num,:] = X[2*source_num:3*source_num,:] - 10.
    # source_4_X invaild Y|X shift
    X[3*source_num:4*source_num,0] = X[3*source_num:4*source_num,0] + 2.
    X[3*source_num:4*source_num,1] = X[3*source_num:4*source_num,1] + 1.
    # source_5_X invaild Y|X shift 
    X[4*source_num:5*source_num,0] = X[4*source_num:5*source_num,0] + 2.
    X[4*source_num:5*source_num,1] = X[4*source_num:5*source_num,1] + 1.
    
    # test's distribution as target's
    valid_X, valid_Eta = torch.randn(valid_num, args.d), torch.randn(args.j, args.m)
    valid_X[:,0] = valid_X[:,0] + 2.
    valid_X[:,1] = valid_X[:,1] + 1.
    test_X, test_Eta = torch.randn(valid_num, args.d), torch.randn(args.j, args.m)
    test_X[:,0] = test_X[:,0] + 2.
    test_X[:,1] = test_X[:,1] + 1.
    
    U = torch.rand(num, 1)
    Y = torch.where(U <= 1./3.,
                    torch.normal(mean=-3-X[:,0].view(-1,1)/3-X[:,1].view(-1,1)**2, std=0.5),
                    torch.normal(mean=3+X[:,0].view(-1,1)/3+X[:,1].view(-1,1)**2, std=1.0))
    # source_4_Y invaild
    Y[3*source_num:4*source_num] = torch.where(U[3*source_num:4*source_num].view(-1,1) <= 1/3,
                                                torch.normal(mean=-8-X[3*source_num:4*source_num,0].view(-1,1)**3-X[3*source_num:4*source_num,1].view(-1,1), std=0.5),
                                                torch.normal(mean=8+X[3*source_num:4*source_num,0].view(-1,1)**3+X[3*source_num:4*source_num,1].view(-1,1), std=1.0))
    # source_5_Y invaild
    Y[4*source_num:5*source_num] = torch.where(U[4*source_num:5*source_num].view(-1,1) <= 1/3,
                                                torch.normal(mean=2-X[4*source_num:5*source_num,0].view(-1,1)-X[4*source_num:5*source_num,1].view(-1,1), std=0.5),
                                                torch.normal(mean=-2+X[4*source_num:5*source_num,0].view(-1,1)+X[4*source_num:5*source_num,1].view(-1,1), std=1.0))

    valid_Y_mean = 1./3. * (3+valid_X[:,0].view(-1,1)/3+valid_X[:,1].view(-1,1)**2)
    valid_Y_sd = torch.sqrt((8./9. * (3+valid_X[:,0].view(-1,1)/3+valid_X[:,1].view(-1,1))**2) + 3./4.)
    test_Y_mean = 1./3. * (3+test_X[:,0].view(-1,1)/3+test_X[:,1].view(-1,1)**2)
    test_Y_sd = torch.sqrt((8./9. * (3+test_X[:,0].view(-1,1)/3+test_X[:,1].view(-1,1))**2) + 3./4.)
    
    sources_X = torch.zeros(sources, source_num, args.d)
    sources_Y = torch.zeros(sources, source_num, args.q)
    sources_Eta = torch.zeros(sources, source_num, args.m)
    for i in range(sources):
        sources_X[i] = X[i*source_num:(i+1)*source_num]
        sources_Y[i] = Y[i*source_num:(i+1)*source_num]
        sources_Eta[i] = Eta[i*source_num:(i+1)*source_num]
    
    return [sources_X, sources_Y, sources_Eta, X[sources * source_num:num], Y[sources * source_num:num], Eta[sources * source_num:num]], \
           [valid_X, valid_Y_mean, valid_Y_sd], valid_Eta, [test_X, test_Y_mean, test_Y_sd], test_Eta
           
def synthetic_data(args):
    if args.datasets == 'M1':
        return synthetic_data_M1(args.sources, args.source_num_examples, args.target_num_examples, args.valid_num_examples, args)
    elif args.datasets == 'M2':
        return synthetic_data_M2(args.sources, args.source_num_examples, args.target_num_examples, args.valid_num_examples, args)
    elif args.datasets == 'M3':
        return synthetic_data_M3(args.sources, args.source_num_examples, args.target_num_examples, args.valid_num_examples, args)
    
    