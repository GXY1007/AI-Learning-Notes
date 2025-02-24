# PyTorch 张量（Tensors）学习笔记

## 1. 关键概念
**张量（Tensor）** 是 PyTorch 的核心数据结构，类似于 NumPy 的 ndarray，但提供以下额外特性：
- **支持 GPU 加速计算**
- **支持自动微分（Autograd）**
- **可与 NumPy 共享内存**
- **支持丰富的数学运算（矩阵操作、索引、拼接等）**

**张量 vs. NumPy ndarray**
| 特性 | NumPy ndarray | PyTorch Tensor |
|------|--------------|---------------|
| 计算设备 | 仅支持 CPU | 支持 CPU 和 GPU |
| 自动微分 | 不支持 | 支持（用于训练深度学习模型） |
| 共享内存 | 不能直接共享 | 可直接共享（减少数据复制） |

## 2. 代码示例

### **2.1 张量的初始化**
```python
import torch
import numpy as np

# 直接从数据创建
x_data = torch.tensor([[1, 2], [3, 4]])

# 由 NumPy 数组创建
np_array = np.array([[1, 2], [3, 4]])
x_np = torch.from_numpy(np_array)

# 从现有张量创建（保留形状和类型）
x_ones = torch.ones_like(x_data)

# 指定数据类型创建
x_rand = torch.rand_like(x_data, dtype=torch.float)
```
📌 **面试考点：**
- **`torch.tensor()` vs. `torch.from_numpy()`**（前者创建新张量，后者共享 NumPy 内存）
- **如何确保数据在 GPU 上运行？**（使用 `.to('cuda')`）

### **2.2 生成随机或常量张量**
```python
shape = (2,3)
rand_tensor = torch.rand(shape)  # 随机值
ones_tensor = torch.ones(shape)  # 全 1
zeros_tensor = torch.zeros(shape)  # 全 0
```
🎯 **学习目标：**
- 熟练使用 `torch.rand()`, `torch.ones()`, `torch.zeros()` 生成不同形状张量。

### **2.3 张量属性**
```python
tensor = torch.rand(3,4)
print(f"Shape: {tensor.shape}")
print(f"Datatype: {tensor.dtype}")
print(f"Device: {tensor.device}")
```
📌 **面试考点：**
- 如何查询张量形状？（`.shape`）
- 如何检查张量存储在哪个设备？（`.device`）

### **2.4 张量操作（索引 & 切片）**
```python
tensor = torch.ones(4, 4)
print(tensor[0])  # 访问第一行
print(tensor[:, 0])  # 访问第一列
print(tensor[..., -1])  # 访问最后一列
tensor[:,1] = 0  # 修改第二列所有元素
```
🎯 **学习目标：**
- 熟练掌握 PyTorch 的索引和切片操作（与 NumPy 类似）。

### **2.5 张量拼接（合并）**
```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)  # 按列拼接
```
📌 **面试考点：**
- `torch.cat()` vs. `torch.stack()`（`cat` 沿维度拼接，`stack` 创建新维度）

### **2.6 张量运算（矩阵乘法 & 逐元素运算）**
```python
# 矩阵乘法
y1 = tensor @ tensor.T  # 矩阵相乘
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# 逐元素乘法
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```
📌 **面试考点：**
- `@` vs. `matmul()` vs. `*`（`@` 进行矩阵乘法，`*` 进行逐元素乘法）
- `out` 参数如何减少额外内存分配？（通过指定输出张量，避免创建新张量，减少内存分配）

### **2.7 单元素张量转换**
```python
agg = tensor.sum()
agg_item = agg.item()  # 转换为 Python 数值
```
📌 **面试考点：**
- `tensor.item()` 何时使用？（用于提取单个标量值）

### **2.8 原地操作（in-place）**
```python
tensor.add_(5)  # 所有元素加 5
```
📌 **面试考点：**
- PyTorch 中 **`_` 后缀表示原地操作**（如 `add_()` 会直接修改 `tensor`）
- 为什么原地操作影响梯度计算？（会丢失计算图）

## 3. NumPy & PyTorch 互操作
```python
# 张量转 NumPy
n = torch.ones(5)
n_np = n.numpy()

# NumPy 转张量
t = np.ones(5)
t_tensor = torch.from_numpy(t)
```
🎯 **学习目标：**
- 了解 NumPy 和 PyTorch 如何共享数据（修改一个会影响另一个）。

## 4. 面试点总结
✅ **张量 vs. NumPy ndarray**（区别、相互转换）
✅ **张量初始化方法**（`torch.tensor()`, `torch.from_numpy()`）
✅ **GPU 计算**（如何检查 GPU，如何把数据移动到 GPU）
✅ **重要操作**（索引切片、拼接、矩阵运算、逐元素运算）
✅ **原地操作对梯度的影响**
✅ **NumPy 互操作**（数据共享，减少内存复制）


🗣️ **面试话术**
> 在 PyTorch 中，张量是核心数据结构，类似 NumPy 的 ndarray，但支持 GPU 计算和自动微分。我们可以使用 `torch.tensor()` 或 `torch.from_numpy()` 来创建张量，并通过 `.to('cuda')` 将其移动到 GPU。对于计算，我们可以使用 `matmul()` 进行矩阵乘法，`*` 进行逐元素乘法，同时 `cat()` 用于拼接张量。如果需要获取单个值，可以使用 `.item()`，但要避免原地操作（如 `add_()`），以防止影响计算图。
