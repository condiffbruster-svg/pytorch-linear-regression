import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


x = torch.tensor([1,2,3,4,5],dtype=torch.float32)
y = torch.tensor([5,7,9,11,13],dtype=torch.float32)
x = x.reshape(-1,1)
y = y.reshape(-1,1)
#print(x.shape)
#print(y.shape)

model = nn.Linear(1,1)
print(model.weight)
print(model.bias)

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01)

epochs = 1000

losses = []
for epoch in range(epochs):
    #1.向前传播得到预测值
    pred = model(x)

    #2.计算损失
    loss = loss_fn(pred,y)

    #3.清空上一轮梯度
    optimizer.zero_grad()

    #4.反向传播计算梯度
    loss.backward()

    #5.更新参数
    optimizer.step()
    losses.append(loss.item())
print("训练后的weight:",model.weight)
print("训练后的bias:",model.bias)
print("最终损失：",loss.item())
mae = torch.mean(torch.abs(pred - y))
print("MAE:",mae.item())

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("loss.png")
plt.show()

#new_x = torch.tensor([6.0])

with torch.no_grad():
    pred = model(x)

#print("输入：",new_x.item())
#print("预测结果：",pred.item())

plt.scatter(x.numpy(), y.numpy(), label = "Original Data")

plt.plot(x,pred,label = "Prediction")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression")
plt.legend()
plt.savefig("prediction.png")
plt.show()

torch.save(model.state_dict(), "model.pth")
print("模型保存成功！")


