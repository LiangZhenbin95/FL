import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy.stats import ttest_ind
import logging
from typing import List, Tuple, Dict, Any
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# 尝试导入syft库
try:
    import syft as sy
except ImportError:
    print("PySyft not installed or incompatible version. Running without PySyft.")
    sy = None

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置设备(GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义客户端类
class Client:
    def __init__(self, name):
        self.name = name
        self.data = None
        self.dataloader = None
        self.model = None
    
    def __str__(self):
        return f"Client({self.name})"

# 创建客户端
def create_clients(num_clients: int) -> List[Client]:
    return [Client(f"client_{i}") for i in range(num_clients)]

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 数据集分割函数
def split_dataset(dataset, num_clients):
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)
    
    # 按标签组织数据索引以确保平衡分布
    label_indices = {}
    for idx in indices:
        label = dataset[idx][1]
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(idx)
    
    # 为每个客户端分配样本
    client_indices = [[] for _ in range(num_clients)]
    for label, label_idx in label_indices.items():
        label_per_client = len(label_idx) // num_clients
        for i, idx in enumerate(label_idx):
            client_indices[i % num_clients].append(idx)
    
    # 创建数据子集
    split_datasets = []
    for indices in client_indices:
        subset = torch.utils.data.Subset(dataset, indices)
        split_datasets.append(subset)
    
    return split_datasets

# 加载数据集并分发给客户端
def load_datasets(data_root: str = './data', num_clients: int = 2) -> Tuple[Dict[str, torch.utils.data.DataLoader], torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    try:
        # 加载MNIST训练和测试数据
        trainset = torchvision.datasets.MNIST(root=data_root, train=True,
                                              download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=data_root, train=False,
                                             download=True, transform=transform)
        
        # 全局训练数据加载器
        global_train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=64, shuffle=True)
        
        # 创建客户端
        clients = create_clients(num_clients)
        
        # 分割训练集
        split_trainsets = split_dataset(trainset, num_clients)
        
        # 为每个客户端创建数据加载器
        client_dataloaders = {}
        for i, client in enumerate(clients):
            client.data = split_trainsets[i]
            client.dataloader = torch.utils.data.DataLoader(
                split_trainsets[i], batch_size=64, shuffle=True)
            client_dataloaders[client.name] = client.dataloader
        
        # 测试数据加载器
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=64, shuffle=False)
        
        return client_dataloaders, global_train_loader, test_loader
    except Exception as e:
        logging.error(f"Failed to load datasets: {e}")
        raise

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout层防止过拟合
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # 全连接层
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

# 训练参考模型作为良性基准
def train_reference_model(train_loader, test_loader, epochs=5, lr=0.01):
    print("Training reference model...")
    reference_model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(reference_model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        reference_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in tqdm(train_loader, desc=f'Reference Model - Epoch {epoch+1}'):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = reference_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        accuracy, precision, recall, f1 = evaluate_model(reference_model, test_loader)
        print(f"Reference model - Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {accuracy*100:.2f}%")
    
    return reference_model

# 噪声衰减函数 - 用于差分隐私
def noise_decay_function(initial_noise: float, loss_change_rate: float, decay_factor: float = 0.7) -> float:
    noise = initial_noise * (decay_factor ** loss_change_rate)
    return max(noise, 0.01)

# 计算模型参数余弦相似度
def cosine_similarity(model1: nn.Module, model2: nn.Module) -> float:
    params1 = torch.cat([param.view(-1) for param in model1.parameters()])
    params2 = torch.cat([param.view(-1) for param in model2.parameters()])
    return torch.nn.functional.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0)).item()

# 计算梯度方向一致性
def gradient_direction_consistency(model: nn.Module, global_model: nn.Module) -> float:
    model_grads = []
    global_grads = []
    
    for param, global_param in zip(model.parameters(), global_model.parameters()):
        if param.grad is not None and global_param.grad is not None:
            model_grads.append(param.grad.view(-1))
            global_grads.append(global_param.grad.view(-1))
    
    if len(model_grads) == 0:
        return 0.0
    
    model_grads = torch.cat(model_grads)
    global_grads = torch.cat(global_grads)
    
    cos_sim = torch.nn.functional.cosine_similarity(model_grads.unsqueeze(0), global_grads.unsqueeze(0)).item()
    return cos_sim

# 使用T检验比较参数分布
def parameter_distribution_statistics(model: nn.Module, benign_model: nn.Module) -> float:
    p_values = []
    for param1, param2 in zip(model.parameters(), benign_model.parameters()):
        try:
            param1_np = param1.detach().cpu().numpy().flatten()
            param2_np = param2.detach().cpu().numpy().flatten()
            
            if len(param1_np) > 1 and len(param2_np) > 1:
                _, p = ttest_ind(param1_np, param2_np, nan_policy='omit')
                
                if not np.isnan(p):
                    p_values.append(p)
        except ValueError as e:
            logging.warning(f"T-test failed due to invalid data: {e}. Skipping this parameter.")
    
    return np.mean(p_values) if p_values else 0.0

# 多指标检测评估客户端模型
def multi_metric_detection(models: List[nn.Module], global_model: nn.Module, benign_model: nn.Module) -> List[float]:
    scores = []
    for model in models:
        # 计算三个指标并取平均值
        cos_sim_score = cosine_similarity(model, global_model)
        grad_consistency_score = gradient_direction_consistency(model, global_model)
        p_value_score = parameter_distribution_statistics(model, benign_model)
        score = (abs(cos_sim_score) + abs(grad_consistency_score) + p_value_score) / 3
        scores.append(score)
    return scores

# 根据模型评分进行加权聚合
def weighted_aggregation(models: List[nn.Module], scores: List[float]) -> nn.Module:
    total_score = sum(scores)
    aggregated_model = Net().to(device)
    for param in aggregated_model.parameters():
        param.data.zero_()
    
    # 规范化分数
    if total_score > 0:
        normalized_scores = [s / total_score for s in scores]
    else:
        normalized_scores = [1.0 / len(models) for _ in models]
    
    # 检测异常值 - 可能是恶意模型
    mean_score = np.mean(normalized_scores)
    std_score = np.std(normalized_scores)
    adjusted_scores = []
    
    for score in normalized_scores:
        # 如果得分低于平均值2个标准差，认为是异常值
        if score < mean_score - 2 * std_score:
            adjusted_score = 0.1 * score  # 降低其影响
        else:
            adjusted_score = score
        adjusted_scores.append(adjusted_score)
    
    # 重新归一化调整后的得分
    total_adjusted = sum(adjusted_scores)
    if total_adjusted > 0:
        final_weights = [s / total_adjusted for s in adjusted_scores]
    else:
        final_weights = [1.0 / len(models) for _ in models]
    
    # 使用最终权重进行模型聚合
    for model, weight in zip(models, final_weights):
        for aggregated_param, model_param in zip(aggregated_model.parameters(), model.parameters()):
            aggregated_param.data += weight * model_param.data
    
    return aggregated_model

# 评估模型性能
def evaluate_model(model: nn.Module, test_loader: torch.utils.data.DataLoader) -> Tuple[float, float, float, float]:
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = correct / total
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')

    return accuracy, precision, recall, f1

# 计算测试损失
def calculate_test_loss(model: nn.Module, test_loader: torch.utils.data.DataLoader) -> float:
    model.eval()
    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
    
    return test_loss / len(test_loader)

# 在客户端上训练模型
def train_on_client(model: nn.Module, 
                    data_loader: torch.utils.data.DataLoader, 
                    epochs: int = 1, 
                    lr: float = 0.01, 
                    noise: float = 0.3) -> nn.Module:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc=f'Client Training Epoch {epoch+1}')):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            
            # 添加噪声实现差分隐私
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad += noise * torch.randn_like(param.grad)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(data_loader)
        epoch_acc = 100. * correct / total
        print(f'Client Training - Epoch: {epoch+1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    
    return model

# 训练联邦学习模型
def train_federated_model(client_dataloaders: Dict[str, torch.utils.data.DataLoader],
                         test_loader: torch.utils.data.DataLoader,
                         benign_model: nn.Module,
                         num_rounds: int = 10,
                         local_epochs: int = 2,
                         initial_noise: float = 0.3,
                         lr: float = 0.01,
                         lr_decay: float = 0.95,
                         early_stopping_patience: int = 5) -> nn.Module:
    # 初始化全局模型
    global_model = Net().to(device)
    
    # 学习率和噪声初始化
    current_lr = lr
    current_noise = initial_noise
    
    # 用于存储训练指标
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    # 早停相关变量
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 联邦学习主循环
    for round_idx in range(num_rounds):
        print(f"Round {round_idx+1}/{num_rounds}")
        
        local_models = []
        local_losses = []
        
        # 在每个客户端上训练本地模型
        for client_name, dataloader in client_dataloaders.items():
            print(f"Training on {client_name}")
            
            # 复制全局模型到本地
            local_model = copy.deepcopy(global_model)
            
            # 在本地训练模型
            local_model = train_on_client(
                model=local_model,
                data_loader=dataloader,
                epochs=local_epochs,
                lr=current_lr,
                noise=current_noise
            )
            
            # 计算本地损失
            local_loss = calculate_test_loss(local_model, dataloader)
            local_losses.append(local_loss)
            
            local_models.append(local_model)
        
        # 计算平均训练损失
        avg_train_loss = np.mean(local_losses)
        train_losses.append(avg_train_loss)
        
        # 多指标检测评估每个模型的贡献
        scores = multi_metric_detection(local_models, global_model, benign_model)
        print(f"Client contribution scores: {scores}")
        
        # 根据评分进行加权聚合
        global_model = weighted_aggregation(local_models, scores)
        
        # 在测试集上评估全局模型
        test_loss = calculate_test_loss(global_model, test_loader)
        test_losses.append(test_loss)
        
        test_accuracy, precision, recall, f1 = evaluate_model(global_model, test_loader)
        test_accuracies.append(test_accuracy)
        
        print(f"Round {round_idx+1} Results:")
        print(f"Avg Train Loss: {avg_train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy*100:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # 早停策略
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(global_model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at round {round_idx+1}")
                break
        
        # 更新学习率
        current_lr *= lr_decay
        
        # 根据损失变化动态调整噪声
        if len(test_losses) > 1:
            loss_change_rate = abs(test_losses[-1] - test_losses[-2]) / test_losses[-2]
            current_noise = noise_decay_function(initial_noise, loss_change_rate)
            print(f"Updated noise level: {current_noise:.6f}")
    
    # 加载最佳模型状态
    if best_model_state is not None:
        global_model.load_state_dict(best_model_state)
        print("Loaded best model state based on test loss")
    
    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('federated_learning_results.png')
    plt.show()
    
    return global_model

# 主函数
def main(num_clients: int = 3, num_rounds: int = 15, local_epochs: int = 2):
    # 设置随机种子以确保可复现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Loading datasets...")
    client_dataloaders, global_train_loader, test_loader = load_datasets(num_clients=num_clients)
    
    # 训练良性参考模型
    benign_model = train_reference_model(global_train_loader, test_loader, epochs=3)
    
    print(f"Starting federated learning with {num_clients} clients...")
    model = train_federated_model(
        client_dataloaders=client_dataloaders,
        test_loader=test_loader,
        benign_model=benign_model,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        initial_noise=0.3,
        lr=0.01,
        lr_decay=0.95,
        early_stopping_patience=5
    )
    
    # 最终评估
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader)
    print("\nFinal Model Evaluation:")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), 'federated_model.pth')
    print("Model saved as 'federated_model.pth'")
    
    return model

if __name__ == "__main__":
    main(num_clients=3, num_rounds=15, local_epochs=2)