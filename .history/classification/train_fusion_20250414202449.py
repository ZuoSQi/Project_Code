#训练脚本
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from models.lstm_irfpca_fusion import FusionClassifier

# ----------------------------
# 数据定义
# ----------------------------
class FusionDataset(Dataset):
    def __init__(self, X_seq, X_feat, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_feat = torch.tensor(X_feat, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_feat[idx], self.y[idx]

# ----------------------------
# 数据加载
# ----------------------------
X_seq = np.load('data/X_seq.npy')     # [208, 1000, 3]
X_feat = np.load('data/X_feat.npy')   # [208, 20]
Y = np.load('data/Y.npy')             # [208]

# 划分训练/测试
X_seq_train, X_seq_test, X_feat_train, X_feat_test, Y_train, Y_test = train_test_split(
    X_seq, X_feat, Y, test_size=0.2, stratify=Y, random_state=0)

train_ds = FusionDataset(X_seq_train, X_feat_train, Y_train)
test_ds  = FusionDataset(X_seq_test, X_feat_test, Y_test)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=16)

# ----------------------------
# 模型训练
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FusionClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("开始训练...")
for epoch in range(30):
    model.train()
    total_loss = 0
    for x_seq, x_feat, y in train_loader:
        x_seq, x_feat, y = x_seq.to(device), x_feat.to(device), y.to(device).squ
        out = model(x_seq, x_feat)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

# ----------------------------
# 模型评估
# ----------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for x_seq, x_feat, y in test_loader:
        x_seq, x_feat = x_seq.to(device), x_feat.to(device)
        out = model(x_seq, x_feat)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

print("\n📊 分类报告：")
print(classification_report(all_labels, all_preds, target_names=["正常", "偏移", "振动", "失控"]))
