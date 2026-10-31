import torch
import torch.nn as nn
import torch.optim as optim

#重新创建模型结构
model = nn.Linear(1,1)
#加载之前保存的模型参数
model.load_state_dict(torch.load("model.pth",weights_only = True))
#切换到评估模式
model.eval()

#准备新的数据
new_x = torch.tensor([6.0])

#进行预测
with torch.no_grad():
    pred = model(new_x)

print("输入：",new_x.item())
print("预测结果：",pred.item())