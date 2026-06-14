from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from helper_functions import plot_decision_boundary, plot_predictions

n_samples = 1000
X,y = make_circles(n_samples, noise=0.03, random_state=42)
#print(X.shape,y.shape)

#plt.scatter(x=X[:, 0],
 #           y=X[:, 1],
  #          c=y,
   #         cmap=plt.cm.RdYlBu)

#plt.show()

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
#print(X[:5],y[:5],X.shape,y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#print(len(X_train),len(X_test))

device = "cuda" if torch.cuda.is_available() else "cpu"
#print(device)

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=20)
        self.layer2 = nn.Linear(in_features=20, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))


def accuyacy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_true)) * 100

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


def train_model(train_model : nn.Module):
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model_0.parameters(),lr=0.1)

    torch.manual_seed(42)

    epochs = 1000

    for epoch in range(epochs):
        train_model.train()

        y_logits = train_model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        
        loss = loss_fn(y_logits, y_train)
        acc = accuyacy_fn(y_true=y_train,y_pred=y_pred)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        train_model.eval()

        with torch.inference_mode():
            test_logits = train_model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuyacy_fn(y_test, test_pred)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")




def plot_train_test_boundry(model):
    # Plot decision boundaries for training and test sets
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, X_test, y_test)
    plt.show()




if __name__ == "__main__":
    model_0 = CircleModelV0().to(device)
    print(model_0)

    train_model(model_0)

    plot_train_test_boundry(model_0)


