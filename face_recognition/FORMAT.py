import os
import torch
import torchvision
from tqdm import tqdm
import matplotlib

# 检测运算设备是GPU还是CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 获取到的数据集做转换
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),                          # 把获取到数据转换成Tesnor（张量），Pytorch计算的数据类
                                                                # 具体得到是FloatTensor，结构是（通道数 × 图高 × 图宽）
    torchvision.transforms.Normalize(mean = [0.5],std = [0.5])  # 通过减均值并除以标准差将数据转换为均值为0、标准差为1的分布
])

DOWNLOAD_MNIST = False
# 判断是否要下载数据集
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # 如果已存在数据集，就不再下载
    DOWNLOAD_MNIST = True

# 创建训练数据集
trainData = torchvision.datasets.MNIST(
    root='./mnist/',                                # 数据集所在路径
    train=True,                                     # 用于训练
    transform=transform,
    download=DOWNLOAD_MNIST,
)

# 创建测试数据集
testData = torchvision.datasets.MNIST(
    root='./mnist/',                                # 数据集所在路径
    train=True,                                     # 用于训练
    transform=transform                             # 用上面的转换器
)

# 训练批次，每forward一次训练多少条数据
BATCH_SIZE = 256
# 迭代次数，训练完全部数据，一共几次
EPOCHS = 1

# 创建测试数据集的加载器
# DataLoader
# 参数1：dataset，传入的数据集（如自定义的Dataset子类实例或内置数据集）
# 参数2：batch_size，将数据集分割成多个批次，每个批次的数据量是batch_size，从而批量加载数据
# 参数3：shuffle，是否在每个epoch（轮次）前打乱数据顺序（训练时通常设为True）
# 参数4：num_workers，用于加载数据的进程数（默认值为 0，表示仅主线程加载），非0时需大于0。可以提高加载数据的速度
# 参数5：collate_fn，自定义的批量处理函数，用于将多个样本拼接成一个批次（默认自动处理）
# 参数6：pin_memory，只适用于GPU，可以将数据直接传输到GPU内存
# 锁页（pined）内存，特殊的内存区域，可以让硬件直接访问，避免数据从虚拟内存（硬盘）->物理内存（内存条）->GPU内存，的2次复制
# 参数7：drop_last，若数据集长度不能被batch_size整除，是否丢弃最后一个不完整的批次（默认False）。
# 参数8：sampler，自定义采样器，用于控制样本的选取规则（如按权重采样、分层采样等）。
trainDataLoader = torch.utils.data.DataLoader(dataset = trainData, batch_size = BATCH_SIZE,shuffle = True)
testDataLoader = torch.utils.data.DataLoader(dataset = testData, batch_size = BATCH_SIZE)

# 定义MLP
class Net(torch.nn.Module):
    # 定义模型结构，创建初始化时，会执行
    def __init__(self):
        super(Net,self).__init__()
        self.model = torch.nn.Sequential(
            # 输入28×28 -> 输出100
            torch.nn.Linear(28 * 28, 50),
            # 50 -> 10（因为数据集中有10种数，所以最后的线性层输出是10，表示10分类）
            torch.nn.Linear(50, 10),
            # 归一化 softmax_xi = exp(xi) / sum(exp(xi,j,,))
            # 将10种分类的可能性转换成概率，配合交叉熵计算损失
            # 例：[1, 2, 3, 4, 1, 2, 3] -> [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
            # dim: 指明维度：dim=0表示按列计算；dim=1表示按行计算，不要错认成：输入或输出维度！！！
            torch.nn.Softmax(dim=1)
        )
    # 模型推理运行
    def forward(self,input):
        output = self.model(input)
        return output

# 创建MLP实例
net = Net()
# Pytorch的to函数是用于将数据或模型迁移到指定的计算设备上
# 计算设备包括：
# CPU：中央处理单元，通常是机器的默认设备
# GPU：图形处理单元，用于加速计算。通过 cuda 设备来指定，通常用于深度学习训练
# 尤其是在使用 GPU 进行训练时，确保数据和模型都在 GPU 上计算，以提高效率
print(net.to(device))


# 创建loss损失函数，衡量预测值与真实值的差异程度、
# y真实值，y^预测值，N表示y的个数
# 均方误差MSE = sum(y - y^)^2 / N，适用回归预测任务
# 交叉熵CE = sum(−y × log(y^) − (1 − y) × log(1 − y^))，适用分类任务
# 例：
# y = 1，y^ = 0.9 二值差异小 CE = 0.105，损失小，预测准确
# y = 1，y^ = 0.1 二值差异大 CE = 2.303，损失达，预测错误
# 所以，loss（损失）值越小，说明预测越准，即loss值越小越好
lossF = torch.nn.CrossEntropyLoss()

# 优化器，凸优化中，寻找最小点（理解成最低处），通俗理解就是一步步的寻找
# 怎么做：需要求导，即梯度下降
# 前置概念：
# 当前权重W = 上次权重W^ - 步长v × 梯度g
# 梯度g = 损失L对权重W的求导结果，可以理解梯度为W寻找最小点的方向
# 传统的梯度下降，是以一个固定的步长寻找这个最小点
# 步长就是学习率v，学习率固定，处理不同参数的时候，可能就会过快或过慢
# 过快，就是跨过了最小点；过慢，就是步子迈的太小，距离最小点还很远
optimizer = torch.optim.Adam(net.parameters())

history = {'Test Loss': [], 'Test Accuracy': []}
for epoch in range(1, EPOCHS + 1):
    # tqdm是tqdm是一个快速、可扩展的Python进度条，可以在Python长循环中添加一个进度提示信息
    # 用户只需要封装任意的迭代器tqdm（iterator）
    # 它可以帮助我们监测程序运行的进度，估计运行的时长，甚至可以协助debug
    # *对于在长时间运行的任务中显示进度很有用，因为它可以让用户知道任务正在进行
    # tqdm可以与dataset、dataloader共同使用，for loop语法依然生效
    processBar = tqdm(trainDataLoader, unit='step')
    # 将模型切换为训练模式
    # *若需切换到评估模式（如验证或测试），应调用model.eval()关闭训练相关功能
    net.train(True)
    for step, (trainImgs, labels) in enumerate(processBar):
        # 训练数据集（图像与标签）迁移到计算设备上
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)
        # zero_grad函数是将模型net的梯度参数置0
        # 如果不执行zero_grad，每次迭代时，求得的梯度是当前梯度加上之前的梯度
        # 有什么好处？保证模型在迭代训练时，尽可能独立，不受上次训练的影响，这样经过多次迭代，可以获得更泛化的模型
        net.zero_grad()
        # trainImgs进行张量形状重塑，因为trainImgs目前的shape是28 × 28，是一张二维单通道的黑白图像
        # 但模型net的输入是28 × 28，是一个一维度数据
        # 所以要将每张图像的长宽维度，都重塑到一个维度，即二维转一维
        # 再考虑到batch_size，则是（batch_size，28，28） --> （batch_size，28 × 28）
        train_x = trainImgs.view(-1, 28 * 28)
        # 向net输入trainImgs，返回训练结果outputs，根据模型结构outputs是batch_size × 10，10表示这张图片是10种数分别的概率
        outputs = net(train_x)
        # 将输出结果与真实标签，进行交叉熵损失函数计算，参考上文的交叉熵科普
        loss = lossF(outputs, labels)
        # argmax求当前列表中，最大值所在的索引
        # 如[0.3, 0.1, 0.9]，argmax返回结果是2，因为最大值0.9的索引是2
        # dim=1，表示按行计算，即求每行各列的argmax
        # 因为输出batch_size × 10，10分类在第2维度（dim=0为第1维度，视为行，按列算行；dim=1为第2维度，视为列，按行算列）
        predictions = torch.argmax(outputs, dim=1)
        # 求预测类别与真实类别相同的数量和，然后除以数据总量（batch_size），得出准确度
        accuracy = torch.sum(predictions == labels) / labels.shape[0]
        # 反向传播计算梯度，损失函数对权重求导，求凸优化方向
        loss.backward()
        # 更新参数，梯度被反向计算之后，既然知道往哪个方向寻最优解了，就可以调用函数进行所有参数更新
        optimizer.step()
        # 设置进度条的显示格式
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                   (epoch, EPOCHS, loss.item(), accuracy.item()))
        # 判断是否到当前轮次的最后一个批次
        if step == len(processBar) - 1:
            # correct表示准确的个数，totalLoss表示损失值
            correct, totalLoss = 0, 0
            # net.train(False)将模型设置为推理模式，用处：
            # 1关闭随机性的层，如Dropout层，以确保每次推理结果都是确定的
            # 2不再更新梯度，不会计算保存梯度值，也就是模型不会训练，这是最重要的
            # 有同学会问，.train(False)与.eval()有什么区别？
            # 回答：二者功能上，没有区别，就是在表达上，.eval()更正式，意指明确告知代码阅读者当前为评估模式
            net.train(False)
            # torch.no_grad()是PyTorch中的一个上下文管理器（context manager）
            # 用于指定在其内部的代码块中不进行梯度计算
            # 当你不需要计算梯度时，可以使用该上下文管理器来提高代码的执行效率
            # 尤其是在推断（inference）阶段和梯度裁剪（grad clip）阶段的时候
            with torch.no_grad():
                # 加载测试数据集
                for testImgs, labels in testDataLoader:
                    # 将测试数据集的图像与标签迁移到计算设备上
                    testImgs = testImgs.to(device)
                    labels = labels.to(device)
                    # 调用模型net计算输出结果outputs
                    test_x = testImgs.view(-1, 28 * 28)
                    outputs = net(test_x)
                    # 计算模型net的损失值（只用于统计，不会进行梯度更新）
                    loss = lossF(outputs, labels)
                    # 获得预测结果
                    predictions = torch.argmax(outputs, dim=1)
                    # 对totalLoss累加
                    totalLoss += loss
                    # 对预测正确的数量累加
                    correct += torch.sum(predictions == labels)
                # 计算测试集的准确度
                testAccuracy = correct / (BATCH_SIZE * len(testDataLoader))
                testLoss = totalLoss / len(testDataLoader)
                history['Test Loss'].append(testLoss.item())
                history['Test Accuracy'].append(testAccuracy.item())

    processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                               (epoch, EPOCHS, loss.item(), accuracy.item(), testLoss.item(), testAccuracy.item()))
    # 关闭tqdm进度条
    processBar.close()

# 保存模型的结构与参数
torch.save(net,'./model.pth')

import matplotlib.pyplot as plt
plt.plot(history['Test Loss'],label = 'Test Loss')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(history['Test Accuracy'],color = 'red',label = 'Test Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()