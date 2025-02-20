# PyTorch 数据集与数据加载器（Datasets & DataLoaders）学习笔记

## 1. 核心概念
**Dataset** 和 **DataLoader** 是 PyTorch 数据管道的核心组件：
- **Dataset**：存储样本及其标签，提供标准化访问接口
- **DataLoader**：包装 Dataset 实现批量加载、数据混洗和多进程加速

**核心功能/优势**  
- 解耦数据预处理与模型训练代码
- 支持批量处理、自动混洗和多进程加载
- 提供预置数据集快速原型开发（FashionMNIST等）
- 灵活支持自定义数据集格式

---

## 2. 基础操作与代码实现

### 2.1 预置数据集加载
```python
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",         # 数据存储路径
    train=True,          # 加载训练集
    download=True,       # 自动下载缺失数据
    transform=ToTensor() # 图像转换为张量
)
```
📌 **面试考点：**
- `transform` 和 `target_transform` 的区别（前者处理特征，后者处理标签）
- 下载失败常见原因（网络连接/存储权限问题）

🎯 **学习目标：**
- 掌握 TorchVision/TorchText 等域库的预置数据集使用方法

### 2.2 自定义数据集实现
```python
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)  # 读取图像为张量
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
```
📌 **面试考点：**
- 自定义 Dataset 必须实现的三个方法（`__init__`, `__len__`, `__getitem__`）
- `read_image` 与 PIL.Image.open 的区别（直接返回张量 vs 返回 PIL 对象）

### 2.3 数据加载器配置
```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    training_data,
    batch_size=64,    # 每批样本数
    shuffle=True,     # 训练集混洗顺序
    num_workers=4     # 使用4个子进程加载数据
)
```
📌 **面试考点：**
- `shuffle=True` 的作用（防止模型记忆样本顺序）
- `num_workers` 设置原则（不超过CPU核心数）

### 2.4 数据可视化与迭代
```python
# 单样本可视化
img, label = training_data[0]
plt.imshow(img.squeeze(), cmap="gray")

# 批量数据遍历
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch {batch} shape: {X.shape}")
```
🎯 **学习目标：**
- 掌握张量数据与可视化库（matplotlib）的交互
- 理解数据加载器的迭代机制

---

## 3. 高级功能与最佳实践

### 3.1 数据预处理管道
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 数据增强
    transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
])
```
📌 **面试考点：**
- 为什么要在 DataLoader 中做数据增强？（增加样本多样性）

### 3.2 性能优化技巧
- 设置合理的 `num_workers`（通常为 CPU 核心数 2-4 倍）
- 使用 `pin_memory=True` 加速 GPU 数据传输
- 避免在 `__getitem__` 中进行复杂计算

---

## 4. 面试与实战聚焦

✅ **高频考点列表**  
1. Dataset 与 DataLoader 的协作机制  
2. 自定义数据集实现要点  
3. 批量训练的数据流处理  

🗣️ **面试话术模板**  
> 当被问及数据加载流程时，可回答："PyTorch 通过 Dataset 定义数据访问规范，DataLoader 负责批量组织和迭代。自定义数据集需要实现三个核心方法：初始化方法配置数据源、`__len__` 返回数据总量、`__getitem__` 实现按索引加载。通过设置 shuffle=True 避免模型过拟合，num_workers 参数则可提升数据加载效率。"

