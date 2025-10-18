import torch


# reference code: https://zhuanlan.zhihu.com/p/150675648
def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    对源域数据和目标域数据转化为核矩阵
    :param source: 源域数据
    :param target: 目标域数据
    :param kernel_mul: ？？？
    :param kernel_num: 高斯核的数量
    :param fix_sigma: 不同高斯核的sigma值
    :return:
    """
    n_samples = source.shape[0] + target.shape[0]
    total = torch.cat([source, target], dim=0)
    total = torch.unsqueeze(total, dim=1)
    # note: 扩维
    total_0 = total.unsqueeze(0).expand(total.shape[0], total.shape[0], total.shape[1])
    total_1 = total.unsqueeze(1).expand(total.shape[0], total.shape[0], total.shape[1])
    # note: 按特征维进行求和
    L2_distance = ((total_0 - total_1) ** 2).sum(2)
    # note: 不同的核函数（默认高斯核）
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # note: 以 fix_sigma 为中值，以 kernel_mul 为倍数取 kernel_num 个值
    # note: 例如：若 fix_sigma = 1, kernel_mul = 2.0, kernel_num = 5, bandwidth_list = [0.25, 0.5, 1., 2., 4.]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # note: 高斯核的表达式（对每一个sigma分别取核函数，然后求和作为最后的核函数）
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

# reference code: https://blog.csdn.net/zkq_1986/article/details/86747841
def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算源域数据和目标域数据的MMD距离
    :param source: n * len(x)
    :param target: m * len(y)
    :param kernel_mul:
    :param kernel_num: 高斯核的数量
    :param fix_sigma: 不同高斯核的sigma值
    :return:
    """
    # note: 默认源域和目标域的batch_size一致
    source = torch.from_numpy(source)
    target = torch.from_numpy(target)

    batch_size = source.shape[0]
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss
