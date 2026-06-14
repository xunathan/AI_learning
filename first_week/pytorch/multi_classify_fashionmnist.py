import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from helper_functions import accuracy_fn
from tqdm.auto import tqdm
from timeit import default_timer as timer
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = datasets.FashionMNIST(
    root="fash_data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None)
test_data = datasets.FashionMNIST(
    root="fash_data",
    train=False,
    download=True,
    transform=ToTensor())

class_names = train_data.classes
BATCH_SIZE = 32

train_loader = DataLoader(train_data,batch_size = BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)
train_features_batch, train_labels_batch = next(iter(train_loader))

loss_fn = nn.CrossEntropyLoss()


# Show a sample
def show_sample():
    torch.manual_seed(42)
    random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
    img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(train_data.classes[label])
    plt.axis("Off");
    print(f"Image size: {img.shape}")
    print(f"Label: {label}, label size: {label.shape}")
    plt.show()

def print_train_time(start:float, end : float, device:torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: total time{total_time:.3f} seconds")
    return total_time

class FashionMnistModelV0(nn.Module):
    def __init__(self,input_shape : int, hidden_units : int, output_shape : int):
        super().__init__()
        self.layer_stack = nn.Sequential(nn.Flatten(),
                                           nn.Linear(in_features=input_shape, out_features=hidden_units),
                                           nn.Linear(in_features=hidden_units,out_features=output_shape))
    def forward(self, x : torch.Tensor):
        return self.layer_stack(x)

class FashionMnistModelV1(nn.Module):
    def __init__(self,input_shape : int, hidden_units : int, output_shape : int):
        super().__init__()
        self.layer_stack = nn.Sequential(nn.Flatten(),
                                           nn.Linear(in_features=input_shape, out_features=hidden_units),
                                           nn.ReLU(),
                                           nn.Linear(in_features=hidden_units,out_features=output_shape),
                                           nn.ReLU())
    def forward(self, x : torch.Tensor):
        return self.layer_stack(x)

class FashionMnistModelV2(nn.Module):
    def __init__(self, input_shape : int, hidden_units : int, output_shape : int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3,padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units,hidden_units,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,out_features=output_shape)
        )
    def forward(self, x : torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)
    
def train_step(
        model:torch.nn.Module,
        data_loader:torch.utils.data.DataLoader,
        loss_fn:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        accuracy_fn,
        device:torch.device = device):
    train_loss, train_acc = 0,0
    model.to(device)
    for batch, (X,y) in enumerate(data_loader):
        X,y = X.to(device), y.to(device)
        y_pred = model(X)

        loss=loss_fn(y_pred,y)
        train_loss += loss

        train_acc += accuracy_fn(y_true=y, y_pred = y_pred.argmax(dim=1))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"train loss:{train_loss:.5f} | train_acc:{train_acc:.5f}")

def test_step(data_loader:torch.utils.data.DataLoader,
              model:torch.nn.Module,
              loss_fn:torch.nn.Module,
              accuracy_fn,
              device:torch.device=device):
    test_loss, test_acc = 0,0
    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X,y in data_loader:
            X,y = X.to(device), y.to(device)
            test_pred = model(X)

            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred = test_pred.argmax(dim=1))

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"test loss:{test_loss:.5f} | test acc:{test_acc:.5f}")

def train_test_model(model:torch.nn.Module,
              loss_fn:torch.nn.Module):
    torch.manual_seed(42)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    start_time = timer()
    epochs = 3

    for epoch in tqdm(range(epochs)):
        print(f"model:{model.__class__.__name__}, epoch:{epoch}\n-------")
        train_step(model, train_loader, loss_fn, optimizer, accuracy_fn)

        test_step(test_loader,model,loss_fn,accuracy_fn)
    end_time = timer()

    total_train_time = print_train_time(start_time, end_time, device= device)

def make_predictions(
        model:torch.nn.Module,
        data:list,
        device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            #print(f"before: {sample.shape}")
            sample = torch.unsqueeze(sample, dim=0).to(device)

            pred_logits = model(sample)

            pred_prob = torch.softmax(pred_logits.squeeze(), dim=0)

            #print(f"after: {sample.shape},logits:{pred_logits.shape}, pred_prob:{pred_prob.shape}")

            pred_probs.append(pred_prob)
    return torch.stack(pred_probs)


def get_test_data():
    random.seed(42)
    test_sample = []
    test_label = []

    for sample, label in random.sample(list(test_data), k=9):
        test_sample.append(sample)
        test_label.append(label)
    print(f"Test sample image shape: {test_sample[0].shape}\nTest sample label: {test_label[0]} ({class_names[test_label[0]]})")

    return test_sample, test_label

def plot_pred(test_sample, test_label, pred_class):
    # Plot predictions
    plt.figure(figsize=(9, 9))
    nrows = 3
    ncols = 3
    for i, sample in enumerate(test_sample):
        # Create a subplot
        plt.subplot(nrows, ncols, i+1)

        # Plot the target image
        plt.imshow(sample.squeeze(), cmap="gray")

        # Find the prediction label (in text form, e.g. "Sandal")
        pred_label = class_names[pred_class[i]]

        # Get the truth label (in text form, e.g. "T-shirt")
        truth_label = class_names[test_label[i]] 

        # Create the title text of the plot
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"
        
        # Check for equality and change title colour accordingly
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g") # green text if correct
        else:
            plt.title(title_text, fontsize=10, c="r") # red text if wrong
    plt.axis(False);
    plt.show()

if __name__ == "__main__":
    #image, label = train_data[0]
    #print(image.shape, label, train_data.classes)
    #print(f"DataLoaders:{train_loader, test_loader}")
    #print(f"len of dataloader :{len(train_loader), len(test_loader)}")
    #print(model_0)
    #model_0 = FashionMnistModelV0(input_shape=784, hidden_units=10, output_shape=len(class_names))
    #model_0.to(device)
    #train_test_model(model_0, loss_fn)

    #model_1 = FashionMnistModelV1(input_shape=784, hidden_units=10,output_shape=len(class_names))
    #model_1.to(device)
    #train_test_model(model=model_1,loss_fn=loss_fn)

    model_2 = FashionMnistModelV2(input_shape=1, hidden_units=10,output_shape=len(class_names))
    model_2.load_state_dict(torch.load(f="./data/cnn.pth"))

    model_2.to(device)
    #train_test_model(model=model_2,loss_fn=loss_fn)

    #torch.save(model_2.state_dict(), f="./model/cnn.pth")

    test_sample, test_label = get_test_data()
    pred_probs = make_predictions(model_2, test_sample)
    print(pred_probs[:2])
    pred_class = pred_probs.argmax(dim=1)
    print(pred_class)
    plot_pred(test_sample, test_label, pred_class)


