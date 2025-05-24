import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, dimensions=[128, 64], activations=None):
        """
        Initialize the network.

        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
            dimensions (list of int): hidden layer sizes
            activations (list of callables): activation functions for each hidden layer
        """
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        # Build layers
        layer_sizes = [input_size] + dimensions + [n_classes]
        self.n_layers = len(layer_sizes) - 1
        self.layers = nn.ModuleList() # List of nn.Linear, Conv2d. etc...

         # Use default activations
        if activations is None:
            activations = [torch.relu] * self.n_layers
        self.activations = activations

        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

        
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        # Compute everything but the last 
        for i in range(self.n_layers - 1):
            x = self.activations[i](self.layers[i](x))
       
        preds = self.layers[-1](x)
        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes, dim_x = 28, dim_y = 28, filters = (64, 128, 256), kernel_size = 3, feature = 128):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)

        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super(CNN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size-1)//2
        self.input_channels = input_channels
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_classes = n_classes
        self.feature = feature
        self.conv2d1 = nn.Conv2d(in_channels=self.input_channels,
                                 out_channels=self.filters[0],
                                 kernel_size=self.kernel_size,
                                 padding=self.padding)
        self.conv2d2 = nn.Conv2d(in_channels=self.filters[0],
                                 out_channels=self.filters[1],
                                 kernel_size=self.kernel_size,
                                 padding=self.padding)
        self.conv2d3 = nn.Conv2d(in_channels=self.filters[1],
                                 out_channels=self.filters[2],
                                 kernel_size=self.kernel_size,
                                 padding=self.padding)

        self.dim = self.dim_x // ((self.kernel_size - 1) ** 3)
        self.in_features = self.dim * self.dim * self.filters[2]

        self.fc1 = nn.Linear(in_features=self.in_features, out_features=self.feature)
        self.fc2 = nn.Linear(in_features=self.feature, out_features=self.n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = self.conv2d1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=self.kernel_size - 1)

        x = self.conv2d2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=self.kernel_size - 1)

        x = self.conv2d3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=self.kernel_size - 1)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, device=torch.device("cpu")):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs.

        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep)

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            # Reset the gradients
            self.optimizer.zero_grad()
            # Forward pass
            y_pred = self.model(x)
            # Compute the loss
            loss = self.criterion(y_pred, y.long())
            # Compute gradients (backpropagation)
            loss.backward()
            # Update the parameters
            self.optimizer.step()
            if ep % 10 == 0:
                print(f"Epoch {ep}: loss = {loss.item():.4f}")

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation,
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        self.model.eval()

        pred_labels = []
        with torch.no_grad():
            for x, in dataloader:
                x = x.to(self.device)
                y_pred = self.model(x)
                predictions = torch.argmax(y_pred, dim=1)
                pred_labels.append(predictions)

        # Concatenate all predictions
        pred_labels = torch.cat(pred_labels, dim=0)

        return pred_labels

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(),
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()
