import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import Counter
from models.lstm_irfpca_fusion import FusionClassifier

# ==========================
# Focal Loss 自定义类
# ==========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# ==========================
# 平衡采样函数（复制小类别样本）
# ==========================
def balance_samples(X_seq, X_feat, Y):
    from collections import defaultdict
    class_indices = defaultdict(list)
for idx, label in enumerate(Y):
    label = int(label)  # ✅ 强制转换为纯 int
    class_indices[label].append(idx)


    max_count = max(len(idxs) for idxs in class_indices.values())

    new_X_seq, new_X_feat, new_Y = [], [], []
    for label, idxs in class_indices.items():
        reps = max_count // len(idxs)
        remainder = max_count % len(idxs)
        full_indices = idxs * reps + idxs[:remainder]
        new_X_seq.append(X_seq[full_indices])
        new_X_feat.append(X_feat[full_indices])
        new_Y.extend([label] * len(full_indices))

    return np.vstack(new_X_seq), np.vstack(new_X_feat), np.array(new_Y)

# ==========================
# 数据定义
# ==========================
class FusionDataset(Dataset):
    def __init__(self, X_seq, X_feat, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_feat = torch.tensor(X_feat, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_feat[idx], self.y[idx]

# ==========================
# 加载 & 平衡数据
# ==========================
X_seq = np.load('data/X_seq.npy')     # [208, 1000, 3]
X_feat = np.load('data/X_feat.npy')   # [208, 20]
Y = np.load('data/Y.npy')             # [208]

X_seq_train, X_seq_test, X_feat_train, X_feat_test, Y_train, Y_test = train_test_split(
    X_seq, X_feat, Y, test_size=0.2, stratify=Y, random_state=0)

# 对训练集做平衡采样
X_seq_train_bal, X_feat_train_bal, Y_train_bal = balance_samples(X_seq_train, X_feat_train, Y_train)

train_ds = FusionDataset(X_seq_train_bal, X_feat_train_bal, Y_train_bal)
test_ds  = FusionDataset(X_seq_test, X_feat_test, Y_test)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=16)

# ==========================
# 初始化模型 + 损失函数
# ==========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FusionClassifier(lstm_hidden=64, num_classes=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train_bal), y=Y_train_bal)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)

# ==========================
# 模型训练
# ==========================
print("开始训练...")
for epoch in range(10):
    model.train()
    total_loss = 0
    for x_seq, x_feat, y in train_loader:
        x_seq, x_feat, y = x_seq.to(device), x_feat.to(device), y.to(device).squeeze()
        out = model(x_seq, x_feat)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:03d} | Loss: {total_loss:.4f}")

# ==========================
# 模型评估
# ==========================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for x_seq, x_feat, y in test_loader:
        x_seq, x_feat = x_seq.to(device), x_feat.to(device)
        y = y.to(device).squeeze()
        out = model(x_seq, x_feat)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

# ==========================
# 报告与可视化
# ==========================
print("\n📊 分类报告：")
print(classification_report(all_labels, all_preds, target_names=["Normal", "Offset", "Vibration", "OutCtrl"]))

print("📌 模型预测类别分布：", Counter(all_preds))

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal","Offset","Vibration","OutCtrl"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
