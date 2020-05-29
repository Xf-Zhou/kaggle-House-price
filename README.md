# kaggle-House-price

**学习Dive into deep learning第三章最后一节，实战房价预测比赛。该比赛的网页地址是[kaggle house price]( https://www.kaggle.com/c/house-prices-advanced-regression-techniques )。**

本项比赛中，数据预处理采用了以下策略：
1. 对连续数值的特征做均值方差标准化（standardization），缺失值替换为该特征的均值。
2. 将离散特征进行one-hot 编码。
3. 第一次测试，模型选择了简单的线性回归模型，损失函数使用平方损失函数（MSELoss），模型评价方法使用对数均方根误差（log_rmse），训练优化算法使用了Adam算法，并且采用了K折交叉验证法。
4. 第一次测试的超参数：k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64

2020.05.29第二次submission：
```
  def get_net2(num_inputs, num_hidden1):

    net = nn.Sequential(
        nn.Linear(num_inputs, num_hidden1),
        nn.ReLU(),
        nn.Linear(num_hidden1, 1)
    )

    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net
    
  k,num_hidden1,num_epochs, lr, weight_decay, batch_size = 4,128,50, 0.03, 80, 64
```
调参的一些想法：
1. 先用小的epochs找到平滑的loss下降。
2. 学习率先从一个比较大的数开始（如10），之后不断二分减小，找到合适的值。
3. weight_decay可选一个较小的值开始（如0.001），不断二分倍增，知道找到合适的值

