# 按卡点选择讲解例子

以下为教练备课与核对接口的材料。写项目讲义时选取能解决眼前卡点的小例子。若某段恰好实现了本轮作业，用独立的小输入演示其中一个操作，再给学员目标输入及可检查的预期。代码运行后再标为实测。

## 点云中心化短讲义

下面是一份独立的 NumPy 练习，用来示范完整模块的讲解粒度。练习从内存里的坐标开始。

### 一、原理与作用

假设读取器交来一组点，形状描述模块需要知道每个点相对物体中心的位置。同一物体整体向右移动后，坐标会变，点之间的距离仍然相同。可以取所有点的平均位置作为中心，再将每个点写成相对这个中心的坐标。

取两个三维点 `(1, 0, 2)` 和 `(3, 4, 2)`。分别对 x、y、z 求平均，得到中心 `(2, 2, 2)`。第一个点相对中心是 `(-1, -2, 0)`，第二个是 `(1, 2, 0)`。如果两个点的 x 都增加 10，中心的 x 也增加 10，相减后得到相同的两行坐标。这就是要传给下游的结果。

### 二、函数用法

本节要写的函数签名是 `center_points(xyz)`。调用方用 `result = center_points(xyz)` 取得新数组。`xyz` 是形状为 `(N, 3)` 的浮点数组，每行是一个点，三列依次是 x、y、z，本练习约定至少有一个点。返回值和输入形状相同，行的顺序保持一致，原坐标还要供读取器使用。

先单独试一下求中心需要的函数。下面每行是一次测量，两列是两个测量项目。

```python
import numpy as np

readings = np.array([[2, 8], [6, 4]], dtype=float)
averages = np.mean(readings, axis=0, keepdims=True)
print(averages)  # [[4. 6.]]
```

`np` 是 NumPy 的导入别名。`np.array` 的第一个参数给出各行的数据，`dtype=float` 指定元素为浮点数，返回一个数组。`np.mean` 的第一个参数是待求平均的数组。`axis=0` 把多行合起来，第一列算 `(2+6)/2`，第二列算 `(8+4)/2`。`keepdims=True` 使行轴长度变为 1，得到形状为 `(1, 2)` 的新数组。用于三维坐标时，这一行就有三个均值。NumPy 的数组减法可以把形状为 `(1, 3)` 的这一行对齐到 `(N, 3)` 的每一行，逐列相减。

### 三、实现步骤

1. 在练习目录新建 `preprocess.py`，导入 NumPy，定义 `center_points(xyz)`。调用者已经传入坐标数组，函数从这个参数开始处理。
2. 对所有点按列求均值，把结果存作中心。用第一节的两行输入，在这里应看到一行 `[[2., 2., 2.]]`。若出现两行、每行一个数，说明你把每个点自己的三个坐标求了平均，回查 `axis`。
3. 按第一节的相减顺序求出每个点的相对坐标，将结果作为新数组返回。若第一个点的 x 变成了正 1，回查相减顺序。
4. 写完后在 `preprocess.py` 所在目录启动 Python，运行以下片段，模拟下游调用。结果的两行应与手算一致，原数组的第一行应仍是 `[1., 0., 2.]`。

```python
import numpy as np
from preprocess import center_points

xyz = np.array([[1, 0, 2], [3, 4, 2]], dtype=float)
print(center_points(xyz))
print(xyz[0])
```

## RoPE：配对、外积与广播

适用于用户已看懂单个二维旋转，但写不出位置表函数，或怀疑每两项如何配对。

- 先确认实现布局。分半布局下，最后一维为 4 时，配对索引是 `(0,2)` 与 `(1,3)`。若旋转角分别为 `π/2` 与 `0`，输入 `[1,2,3,4]` 旋转后为 `[-3,2,1,4]`。从旋转矩阵推到 `x * cos + rotate_half(x) * sin`，其中分半交换与取负得到 `[-3,-4,1,2]`。交错布局应按其自身配对和扩展规则解释。
- 对常见频率定义 `base ** (-2*i/d)`，用 `i = torch.arange(d // 2)`。若索引用 `torch.arange(0, d, 2)`，指数对应 `-i/d`。位置表用 `position[:, None] * frequency[None, :]` 得到 `[L,d/2]`；分半布局沿末轴拼接余弦表自身与正弦表自身，分别得到 `[L,d]`。
- 输入为 `[B,L,H,d]` 时，位置表用 `[1,L,1,d]` 对齐。验证时用 `L=3,d=4` 让逐元素乘法暴露维度问题，再用不同 Q、K 头数检查头轴广播。位置偏移、dtype 恢复及设备对齐按当前接口具体演示。用户问相对位置时，从单个配对的旋转矩阵推导点积中的 `R(n-m)`，再回到 Q、K 的位置与维度。

## 文本训练样本：对象、标签与移位

适用于 MiniMind 一类项目的分词和训练样本构造。

- 用同一短文本说明 token、token id 和 tokenizer。先创建实际使用的分词器，展示调用得到的返回对象，再单独取 `input_ids`。编号从该分词器取；手写教学编号标为示意。
- 例示正文编号 `[17,23]`，假定 BOS、EOS、PAD 为 `1,2,0`，输入可写成 `[1,17,23,2,0]`。按当前实现说明给边界预留长度，追到移位发生在数据集、整理函数还是损失中。若先把补齐位置的标签设为忽略值，再以位置 `t` 的 logits 预测位置 `t+1` 的标签，有效对齐是输入位置 `[1,17,23,2]` 对应目标 `[17,23,2,-100]`，其中 `-100` 取决于实际损失的 ignore_index。检查同一移位没有做两次，且只遮掉约定忽略的位置。
- 用户卡在调用时，提供对象创建、调用、返回字段与实际取值的连续示例。用户卡在标签与输入共享时，演示复制后只修改标签，验证输入保持原值。

## 扩散策略与 PPO：分清时间轴与概率对象

适用于逐节读扩散策略优化论文时，用户先问高斯策略、去噪和 PPO 的关系。

- 从具体轨迹解释环境步、动作块位置和去噪步。画同一环境观测下的去噪链，再指出哪段动作送入环境、哪些中间量为训练保留。变量和索引以所读版本为准。
- 高斯例子用同一个采样值，比较旧策略与当前策略对它给出的 log probability。若旧概率密度为 `0.2`、新概率密度为 `0.3`，则 `exp(log(0.3)-log(0.2))=1.5`。讲清概率对象、条件、以及哪些动作维或步已归约。在「把每个随机去噪转移视为一次内部决策」的设定下，可推导单次转移比率：新旧分布给旧轨迹中同一个去噪结果打分。区分单步条件密度比、整条路径联合密度比和最终动作边缘密度比。引用具体论文或代码行为前，核对实际评分对象、动作维和步维的归约、优势与裁剪的位置。

## 注意力与 DataLoader：演示实际入口

- 注意力先用一份小 Q、K、V 连续算出点积、缩放、遮罩、沿 key 轴的 softmax 和加权 V。若本轮要实现多头版本，再演示投影、拆头、合头和输出投影。用两枚可访问性不同的 key 验证遮罩实际改变了可见范围。
- 用户问 `DataLoader(generator=...)` 时，用同一小数据集展示 `torch.Generator().manual_seed(...)` 如何接入 DataLoader 和实际迭代。启用 `shuffle=True` 后，分别构造同种子的两个生成器，展示首轮顺序可比较。说明同一个生成器在迭代时会推进状态；需要讨论多进程时，再区分采样顺序与 worker 的随机种子。

## Open3D 刚体配准

适用于从 NumPy 坐标转向 Open3D 的练习。先讲一个点如何得到可用于匹配的特征，再讲两份点云如何利用它。以下片段对应 Real3D-AD 的 `ransac_position.py` 中的 `preprocess_point_cloud` 和 `execute_global_registration`。

**看一个点。** 把空间分成边长为 `d` 的小立方体（体素），每个非空体素用其中各点的平均位置代表。取其中一个代表点，想象给它附近的表面贴一张小平面，垂直于平面的方向就是估计的法向量。FPFH（Fast Point Feature Histograms，快速点特征直方图）用点之间的相对方向和法向量计算角度特征，把三种特征各分成 11 档统计，并汇入邻点的统计量。这个点因此有 33 个数描述周围表面的形状。在源云和模板中寻找描述相似的点，就有了候选点对。平坦区域的多个点可能很相似，所以匹配还要经过几何检验。

**把这个点找回代码里。** `xyz` 是每行一个点的 NumPy 坐标数组，`d` 与坐标同单位。`o3d` 是下面这行导入给 Open3D 起的名字。`Vector3dVector` 把三列坐标包装成 Open3D 可用的点集合，`PointCloud` 把它作为点云的 `.points`。下面用的是 `o3d.geometry` 点云接口。

```python
import numpy as np
import open3d as o3d

reg = o3d.pipelines.registration

def describe_points(xyz, d):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    down = cloud.voxel_down_sample(voxel_size=d)
    normal_search = o3d.geometry.KDTreeSearchParamHybrid(radius=2*d, max_nn=30)
    down.estimate_normals(search_param=normal_search)
    feature_search = o3d.geometry.KDTreeSearchParamHybrid(radius=5*d, max_nn=100)
    feature = reg.compute_fpfh_feature(down, feature_search)
    return down, feature
```

`voxel_down_sample` 返回新的点云。`KDTreeSearchParamHybrid` 指定怎样找邻点，例如 `radius=2*d, max_nn=30` 表示在两倍体素边长的半径内最多取 30 个点。`estimate_normals` 返回 `None`，同时把每个点的法向量写入 `down.normals`。`compute_fpfh_feature` 返回 `Feature` 对象。若有 K 个下采样点，`np.asarray(down.points)[i]`、`np.asarray(down.normals)[i]` 和 `feature.data[:, i]` 依次是同一点的坐标、法向量和描述向量，形状分别为 `(3,)`、`(3,)`、`(33,)`。

**用两份描述做配准。** 设 `source_xyz` 已完成项目要求的轴置换，`template_xyz` 是模板坐标。RANSAC（Random Sample Consensus，随机采样一致性）反复从候选点对里抽样，试算一个旋转和平移，再看有多少点能被同一变换对齐。下面保留项目配置，用关键字标出两个点云、两组特征和筛选条件的对应关系。

```python
source_down, source_feature = describe_points(source_xyz, d)
target_down, target_feature = describe_points(template_xyz, d)
distance = 1.5 * d
result = reg.registration_ransac_based_on_feature_matching(
    source=source_down, target=target_down,
    source_feature=source_feature, target_feature=target_feature,
    mutual_filter=True, max_correspondence_distance=distance,
    estimation_method=reg.TransformationEstimationPointToPoint(with_scaling=False),
    ransac_n=3,
    checkers=[reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
              reg.CorrespondenceCheckerBasedOnDistance(distance)],
    criteria=reg.RANSACConvergenceCriteria(max_iteration=100000, confidence=0.999),
)
```

`mutual_filter` 启用双向特征匹配筛选。每次抽 3 对点，检查对应边的长度比例是否接近 1，再检查变换后的点对距离；固定尺度，只估计旋转和平移。迭代上限与置信度控制何时停止。`result.transformation` 是从这里传入的源云到模板的 4×4 矩阵。`result.fitness` 表示变换后在距离阈值内找到对应的源点占下采样源点的比例，`result.inlier_rmse` 是这些对应点距离的均方根，单位与坐标相同。可以把同一份有起伏、形状不对称的坐标传给两端，检查特征列数与各自点数相等、变换接近单位矩阵，再让学员自己把轴置换和整云变换接回练习。
