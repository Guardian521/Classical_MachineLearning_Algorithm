此项目实现了PLA算法以及“优化”后的Pocket算法


感知机算法（PLA）
PLA 是一种简单的线性二分类算法。它的基本思想是通过不断调整权重向量和偏置。在每次迭代中，如果发现有样本被错误分类，即，就按照一定的规则更新权重和偏置。例如，更新规则通常是和。它的目标是找到一个超平面将两类数据分开。
口袋算法（Pocket）
Pocket 算法是对感知机算法的一种改进。它在感知机算法的基础上，不仅要找到一个能够正确分类所有样本的超平面（如果存在的话），而且在迭代过程中会记录下目前为止使错误分类样本数量最少的超平面（即所谓的 “口袋” 中的超平面）。因为感知机算法在某些情况下可能会出现无限循环而无法收敛到一个最优解，口袋算法试图找到一个相对较好的解。

感知机算法实现

从原始数据集中提取标签数据y。

将处理后的特征数据和标签数据转换为 NumPy 数组X和y。

调用pla函数实现感知机算法，该函数接收特征数据X和标签数据y以及最大迭代次数作为参数，返回感知机的权重向量w和偏置项b。
