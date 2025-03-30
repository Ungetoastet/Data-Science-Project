import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, Subset

# Datensatz Klasse erstelen
class Flights_Data(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        # Fur das Trainieren mit NN muessen die indizes bei 0 anfangen, nicht bei 1
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.y[index], dtype=torch.int64)

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
    
def test_network(model, data, test_size, metric, loss_fn):
    model.eval()
    metric.reset()
    avg_loss = 0.0
    loader = torch.utils.data.DataLoader(data, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    processed = 0
    for x, y in loader:
        x = x.float()
        y = y.float().view(-1, 1)
        logits = model(x)
        loss = loss_fn(logits, y)
        metric(logits, y)
        avg_loss += loss.item()
        processed += 1
        if processed >= test_size:
            break

    avg_loss = avg_loss/test_size
    metric_result = metric.compute()
    metric.reset()
    return metric_result, avg_loss


def move_data(dataset, device):
    x_list, y_list = [], []
    for x, y in dataset:
        x_list.append(x.to(device))
        y_list.append(y.to(device))
    return torch.utils.data.TensorDataset(torch.stack(x_list), torch.stack(y_list))

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