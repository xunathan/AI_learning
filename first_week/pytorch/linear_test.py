import torch
from torch import nn
import matplotlib.pyplot as plt

#print(torch.__version__)
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

#create
X = torch.arange(start, end, step)
y = weight * X + bias

#split data 
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(X_train.shape, X_train.ndim)

#LinearRegressionModelV2模型必须要下面几行，因为线性模型要求必须是2维
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(X_train.shape, X_train.ndim)

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):

    #Plots training data, test data and compares predictions.
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
      # Plot the predictions in red (predictions were made on the test data)
      plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})
    plt.show()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1,dtype=torch.float), requires_grad=True)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)

#这里可以用自定义的参数或者用线性层（V2）的实现，如果更改这两个模型，模型要重新存储。
model_0 = LinearRegressionModelV2()

#model_0.load_state_dict(torch.load(f="./model_linear.pth"))
#print(list(model_0.parameters()))

print(model_0.state_dict())

def plot_pred():
    with torch.inference_mode():
        y_pred = model_0(X_test)
    plot_predictions(predictions=y_pred)

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

epoch_count = []
train_loss_values = []
test_loss_values = []

def train_loop(test_model):
    torch.manual_seed(42)

    epochs = 100
    for epoch in range(epochs):
        test_model.train()

        y_pred = test_model(X_train)

        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        test_model.eval()

        with torch.inference_mode():
            test_pred = test_model(X_test)

            test_loss = loss_fn(test_pred, y_test.type(torch.float))

            if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch:{epoch}, MAE train loss: {loss}, MAE test Loss:{test_loss}")


train_loop(model_0)

def print_loss_curves():
    plt.plot(epoch_count, train_loss_values, label="train_loss")
    plt.plot(epoch_count, test_loss_values, label="test_loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

print_loss_curves()

print(model_0.state_dict())
plot_pred()
torch.save(model_0.state_dict(), f="./model_linear.pth")