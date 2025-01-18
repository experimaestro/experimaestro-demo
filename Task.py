import torch, torchvision, logging
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from experimaestro import Task, Param, Constant

@torch.no_grad()
def evaluate(model, testloader) -> tuple:
    """Evaluate the model on the test set
    Args:
        model: model to evaluate
        testloader: test data loader
    Returns:
        test_loss: average test loss
        test_accuracy: test accuracy
    """
    
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0

    for images, labels in testloader:
        output = model(images)
        test_loss += criterion(output, labels).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    return (
        test_loss/len(testloader),  # test loss
        correct/(len(testloader)*testloader.batch_size)   # test accuracy
            )

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, n_layers:int = 3, hidden_dim:int = 64, kernel_size:int = 3):
        """Simple CNN module with n_layers hidden layers and hidden_dim hidden units"""
        super(CNN, self).__init__()
        # create a list of hidden CNN layers with ReLU activation
        self.layers = nn.Sequential()
        for i in range(n_layers):
            self.layers.add_module(f'conv{i}', 
                nn.Conv2d(
                    in_channels=1 if i == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size, 
                    padding='same'))
            self.layers.add_module(f'relu{i}', nn.ReLU())

        # pooling layer to reduce the size of the output to 13x13
        self.layers.add_module(f'pool', nn.MaxPool2d(kernel_size=2)) 

        # output layer
        self.output = nn.Linear(hidden_dim * 14 * 14
                                , 10)

    def forward(self, x):
        # apply the CNN layers to the input
        x = self.layers(x) 
        # flatten the output
        x = x.view(x.size(0), -1)
        # apply the output layer
        x = self.output(x)
        return torch.log_softmax(x, dim=1)
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TrainOnMNIST(Task):
    """Main Task that learns a rank r Self Attention layer to perform NER from LLM representations"""
    # experimaestro Task parameters
    ## Model
    n_layers: Param[int] = 2    # number of Hidden layers
    hidden_dim: Param[int] = 64 # number of hidden units
    kernel_size: Param[int] = 3 # kernel size of the CNN

    # Training
    epochs: Param[int] = 1      # number of epochs to train the model
    n_val: Param[int] = 100     # number of steps between validation and logging
    lr: Param[float] = 1e-2     # learning rate
    batch_size: Param[int] = 64 # batch size

    ## Task version, (not mandatory)
    version: Constant[str] = '1.0' # can be changed if needed to rerun the task with same parameters

    def execute(self):
        """Main Task training a CNN on the MNIST dataset with PyTorch"""

        logging.info("Training CNN on MNIST dataset with given parameters")
        logging.info("Loading and preprocessing the MNIST dataset")
        
        # Load and preprocess the MNIST dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        logging.info(f"MNIST loaded\n - Train: {len(trainset)} samples\n - Test: {len(testset)} samples")

        # print shape of the images
        logging.debug(f"Image shape: {trainset[0][0].shape}")

        logging.info("Creating the CNN model")
        # Create the CNN model
        model = CNN(self.n_layers, self.hidden_dim)
        model.train()
        logging.info(f"Model created with {self.n_layers} layers of dimension {self.hidden_dim}: {count_parameters(model)} parameters in total")

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Train the model
        history = {'train_loss': [], 'test_loss': [], 'accuracy': []}
        it = 0
        logging.info("Training the CNN model ...")

        for epoch in range(self.epochs):
            running_loss = 0
            for images, labels in tqdm(trainloader, desc=f'Epoch {epoch+1}'):
                it += 1
                output = model(images)

                # Compute the loss and update the model parameters
                optimizer.zero_grad()
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Evaluate the model every n_val steps
                if it % self.n_val == 0: 
                    test_loss, accuracy = evaluate(model, testloader)
                    history["test_loss"].append(test_loss)
                    history["accuracy"].append(accuracy)
                    history["train_loss"].append(running_loss/self.n_val)
                    logging.info(f'Epoch {epoch+1}, Train loss: {running_loss/self.n_val:.3f}, Test loss: {test_loss:.3f}, Accuracy: {accuracy:.3f}')
                    running_loss = 0

        # Plot training history
        plt.figure(figsize=(10,5))
        plt.title('Training history of the CNN on MNIST')
        plt.plot(history['train_loss'], label='Train loss')
        plt.plot(history['test_loss'], label='Test loss')
        # plot accuracy
        plt.twinx()
        plt.plot(history['accuracy'], label='Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        # save figure to a file
        plt.savefig('TrainingLoss.png')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("Training a CNN on the MNIST dataset")
    # Execute the Task with default parameters 
    TrainOnMNIST().execute()