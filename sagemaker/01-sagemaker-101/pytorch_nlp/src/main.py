"""CNN-based text classification on SageMaker with PyTorch"""

# Python Built-Ins:
import argparse
import os
import io
import logging
import sys

# External Dependencies:
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

###### Define the model ############
class Net(nn.Module):
    def __init__(self, vocab_size=400000, emb_dim=300, num_classes=4):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.conv1 = nn.Conv1d(emb_dim, 128, kernel_size=3)
        self.max_pool1d = nn.MaxPool1d(5)
        self.flatten1 = nn.Flatten()
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(896, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.transpose(x,1,2)
        x = self.flatten1(self.max_pool1d(self.conv1(x)))
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

###### Helper functions ############
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        "Initialization"
        self.labels = labels
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.data)

    def __getitem__(self, index):
        # Load data and get label
        X = torch.as_tensor(self.data[index]).long()
        y = torch.as_tensor(self.labels[index])
        return X, y

def load_training_data(base_dir):
    X_train = np.load(os.path.join(base_dir, "train_X.npy"))
    y_train = np.load(os.path.join(base_dir, "train_Y.npy"))
    return DataLoader(Dataset(X_train, y_train), batch_size=16)

def load_testing_data(base_dir):
    X_test = np.load(os.path.join(base_dir, "test_X.npy"))
    y_test = np.load(os.path.join(base_dir, "test_Y.npy"))
    return DataLoader(Dataset(X_test, y_test), batch_size=1)

def load_embeddings(base_dir):
    embedding_matrix = np.load(os.path.join(base_dir, "docs-embedding-matrix.npy"))
    return embedding_matrix

def parse_args():
    """Acquire hyperparameters and directory locations passed by SageMaker"""
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=40)

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--embeddings", type=str, default=os.environ.get("SM_CHANNEL_EMBEDDINGS"))

    return parser.parse_known_args()

def test(model, test_loader, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            target_index = target.max(1, keepdim=True)[1]
            correct += pred.eq(target_index).sum().item()

    test_loss /= len(test_loader.dataset)  # Average loss over dataset samples
    print(f"val_loss: {test_loss:.4f}, val_acc: {correct/len(test_loader.dataset):.4f}")

def train(args):
    ###### Load data from input channels ############
    train_loader = load_training_data(args.train)
    test_loader = load_testing_data(args.test)
    embedding_matrix = load_embeddings(args.embeddings)

    ###### Setup model architecture ############
    model = Net(
        vocab_size=embedding_matrix.shape[0],
        emb_dim=embedding_matrix.shape[1],
        num_classes=args.num_classes,
    )
    model.embedding.weight = torch.nn.parameter.Parameter(torch.FloatTensor(embedding_matrix), False)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    model.to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch_idx, (X_train, y_train) in enumerate(train_loader, 1):
            data, target = X_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        print(f"epoch: {epoch}, train_loss: {running_loss / n_batches:.6f}")  # (Avg over batches)
        print("Evaluating model")
        test(model, test_loader, device)
    save_model(model, args.model_dir, args.max_seq_len)

def save_model(model, model_dir, max_seq_len):
    path = os.path.join(model_dir, "model.pth")
    x = torch.randint(0, 10, (1, max_seq_len))
    model = model.cpu()
    model.eval()
    m = torch.jit.trace(model, x)
    torch.jit.save(m, path)

def model_fn(model_dir):
    """Customized model loading function for inference

    https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#load-a-model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(os.path.join(model_dir, "model.pth")).to(device)
    return model

###### Main application  ############
if __name__ == "__main__":

    ###### Parse input arguments ############
    args, unknown = parse_args()

    train(args)
