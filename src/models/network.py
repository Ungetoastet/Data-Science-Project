import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, Subset

# Datensatz Klasse erstelen
class Flights(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        # Fur das Trainieren mit NN muessen die indizes bei 0 anfangen, nicht bei 1
        return torch.tensor(self.X[index].flatten(), dtype=torch.float32), torch.tensor(int(self.y[index])-1, dtype=torch.int64)


def split_data(dataset: Dataset, train=0.7, val=0.15, test=0.15)->tuple[Dataset, Dataset, Dataset]:
    # Parameter überprüfen
    assert train+val + test == 1.0
    assert train > 0 and train < 1
    assert val > 0 and val < 1
    assert test > 0 and test < 1

    # Anzahl der Datenpunkte pro Split berechnen
    n = len(dataset)

    train_n = int(train*n)
    val_n = int(val*n)
    # Test nimmt sich den Rest der Datenpunkte

    # Indices der Datenpunkte auswählen
    indices = np.arange(n, dtype=int)
    np.random.shuffle(indices)

    # Subsets erstellen
    train_indices = indices[0:train_n]
    trainset = Subset(dataset, train_indices)
    val_indices = indices[train_n:train_n+val_n]
    valset = Subset(dataset, val_indices)
    test_indices = indices[train_n+val_n:]
    testset = Subset(dataset, test_indices)

    return trainset, valset, testset

def get_device(mute=True):
    if torch.cuda.is_available():
        device = "cuda"
        if not mute: print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        if not mute: print("Using CPU")
    return device

class DeviceOverSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, device_dataset):
        self.labels = torch.stack([y for _, y in device_dataset]).cpu().numpy()
        class_counts = np.bincount(self.labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[self.labels]
        self.weights = torch.DoubleTensor(sample_weights)
        self.num_samples = len(device_dataset)

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, True))
    
def test_network(model, loader, metric, loss_fn):
    model.eval()
    metric.reset()
    avg_loss = 0.0
    for x, y in loader:
        logits = model(x)
        loss = loss_fn(logits, y)
        metric(logits.argmax(dim=1), y)
        avg_loss += loss.item()

    avg_loss = avg_loss/len(loader)
    metric_result = metric.compute()
    metric.reset()
    return metric_result, avg_loss


def move_and_norm_data(dataset, device):
    x_list, y_list = [], []
    for x, y in dataset:
        x_list.append(x.to(device))
        y_list.append(y.to(device))
    return torch.utils.data.TensorDataset(torch.stack(x_list), torch.stack(y_list))



new_train_features = torch.stack([x for x, _ in train_data_device])
train_mean = new_train_features.mean(dim=0)
train_std = new_train_features.std(dim=0) + 1e-8

print(f"Normalized data stats - Mean: {train_mean.mean():.6f} ± {train_std.mean():.6f}")
print(f"Normalized data range - Min: {new_train_features.min().item():.3f}, Max: {new_train_features.max().item():.3f}")

# Data Loader auf dem Zielgeraet erstellen
train_loader = torch.utils.data.DataLoader(
    train_data_device,
    batch_size=128,
    sampler=DeviceOverSampler(train_data_device)
)

val_loader = torch.utils.data.DataLoader(
    val_data_device,
    batch_size=128,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_data_device,
    batch_size=128,
    shuffle=True
)

trainlabels = np.array([xy[1] for xy in train_data])

print("Training set size: ", len(trainlabels))

class MLP_Block(nn.Module):
    def __init__(self, layer_count, hidden_size, nonlinearity, regularization):
        super().__init__()
        # Anzahl der Layer anpassbar
        self.layers = nn.ModuleList()
        for _ in range(layer_count):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        self.nonlinearity = nonlinearity
        self.reg = regularization

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.reg(x)
            x = self.nonlinearity(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, block1_layers, block2_layers, num_in, num_out, hidden_size=64, skip_connections=True, activ=nn.GELU(), dropout:float=0.3):
        super().__init__()
        self.skip_connections = skip_connections

        self.in_layer = nn.Linear(num_in, hidden_size)
        self.activation = activ

        # Zwei Bloecke mit unterschiedlicher Regularisierung
        self.block1 = MLP_Block(block1_layers, hidden_size, activ, nn.BatchNorm1d(hidden_size))
        self.block2 = MLP_Block(block2_layers, hidden_size, activ, nn.Dropout(p=dropout))

        self.out_layer = nn.Linear(hidden_size, num_out)
        if self.skip_connections:
            self.id = nn.Identity()

        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        hidden_x = self.activation(self.in_layer(x))

        b1 = self.block1(hidden_x)
        if self.skip_connections:
            b1 = b1 + self.id(hidden_x)

        b2 = self.block2(b1)
        if self.skip_connections:
            b2 = b2 + self.id(b1)

        out = self.out_layer(b2)
        return out
    
# Netzwerk erstellen
network = MLP(2, 2, 
             num_in=204, 
             num_out=5,
             hidden_size=256,
             skip_connections=True,
             activ=nn.GELU(),
             dropout=0.2).to(device)

optimizer = torch.optim.AdamW(network.parameters(), lr=0.00005)
loss_fn = nn.CrossEntropyLoss().to(device)
acc = Accuracy(task="multiclass", num_classes=5, average="micro").to(device)

max_epochs = 100

# Early stopping
maxvalacc = 0
minvalloss = np.Inf
bestNetwork = None
patience = 35
current_patience = patience

# Plot Daten vorbereiten
epochs = []
val_losses = []

for epoch in range(max_epochs):
    running_loss = 0.0
    acc.reset() 
    network.train()

    for x, y in train_loader:
        optimizer.zero_grad()
        logits = network(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() / x.size(0)
        acc(logits, y)
        
    train_acc = acc.compute()
    val_acc, val_loss = test_network(network, val_loader, acc, loss_fn)
    maxvalacc = max(val_acc, maxvalacc)

    # Early stopping    
    newbest = False
    if val_loss < minvalloss:
        minvalloss = val_loss
        newbest = True
        bestNetwork = network.state_dict()
        current_patience = patience
    else:
        current_patience -= 1

    print(f"{'*' if newbest else current_patience:>2} Epoch {epoch+1:>3}/{max_epochs}: Train_acc: {train_acc:.3f}   Val_acc: {val_acc:.3f}   Max Val_acc: {maxvalacc:.3f}   Val loss: {val_loss:.5f}   Min val loss: {minvalloss:.5f}")
    
    # Plot daten
    epochs.append(epoch)
    val_losses.append(val_loss)

    if current_patience == 0:
        print("Early stop!")
        break;

print("Loading best Network for testing...")
network.load_state_dict(bestNetwork)
print("Final test...")
test_acc, test_loss = test_network(network, test_loader, acc, loss_fn)
print(f"Final test accuracy: {test_acc:.4f}, loss {test_loss:.4f}")

# Plot
plt.plot(epochs, val_losses)
plt.title("Validation loss over epochs")
plt.xlabel("epochs")
plt.ylabel("Validation loss")
plt.show()