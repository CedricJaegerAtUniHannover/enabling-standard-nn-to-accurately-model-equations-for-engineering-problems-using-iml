# src/ann.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import os
import json
import pandas as pd
import time
from src.utils import get_short_model_name

class SimpleNN(nn.Module):
    """
    A simple feedforward neural network with configurable depth and width.
    """
    def __init__(self, input_size, hidden_sizes, output_size=1):
        """
        Initializes the neural network.

        Args:
            input_size (int): The number of input features.
            hidden_sizes (list of int): A list where each element is the number of
                                        neurons in a hidden layer.
            output_size (int): The number of output neurons (default is 1 for regression).
        """
        super(SimpleNN, self).__init__()
        
        layers = []
        # Input layer
        in_size = input_size
        
        # Hidden layers
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            in_size = h_size
            
        # Output layer
        layers.append(nn.Linear(in_size, output_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Defines the forward pass of the neural network."""
        return self.network(x)

def create_dataloaders(train_path, val_path, batch_size):
    """
    Loads data from CSV files and creates PyTorch DataLoaders.

    Args:
        train_path (str): Path to the training data CSV file.
        val_path (str): Path to the validation data CSV file.
        batch_size (int): The batch size for the DataLoaders.

    Returns:
        tuple: A tuple containing:
            - DataLoader: The DataLoader for the training set.
            - DataLoader: The DataLoader for the validation set.
            - int: The number of input features.
    """

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df.drop('y', axis=1).values
    y_train = train_df['y'].values
    X_val = val_df.drop('y', axis=1).values
    y_val = val_df['y'].values
    
    input_size = X_train.shape[1]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, input_size

def train_model(model, train_loader, val_loader, epochs, learning_rate, patience, device):
    """
    A standardized training loop for the neural network.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        epochs (int): The maximum number of epochs to train.
        learning_rate (float): The learning rate for the optimizer.
        patience (int): The number of epochs to wait for improvement before stopping early.
        device (torch.device): The device (CPU or GPU) to train on.

    Returns:
        tuple: A tuple containing:
            - nn.Module: The best trained model (based on validation loss).
            - list: A history of training losses per epoch.
            - list: A history of validation losses per epoch.
            - float: The total time taken for training in seconds.
    """
    start_time = time.time()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {epoch_train_loss:.4f}.. "
              f"Val loss: {epoch_val_loss:.4f}")

        # --- Early Stopping Check ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save the best model state
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    # Load the best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training finished in {training_time:.2f} seconds.")
        
    return model, train_loss_history, val_loss_history, training_time

def save_model_and_history(model, history, base_path, model_name):
    """
    Saves the trained model's state dictionary and its training history.

    Args:
        model (nn.Module): The trained PyTorch model.
        history (dict): A dictionary containing lists of training and validation losses.
        base_path (str): The base directory to save the artifacts.
        model_name (str): The base name for the model and history files (without extension).
    """
    # Ensure the target directory exists
    os.makedirs(base_path, exist_ok=True)

    # --- Save Model State ---
    model_path = os.path.join(base_path, f"{model_name}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    # --- Save Training History ---
    history_path = os.path.join(base_path, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history to: {history_path}")

def get_model_details_str(hidden_sizes):
    """
    Creates a standardized string for model architecture details.
    Example: [64, 64, 64] -> "layers-3_neurons-64-64-64"
    """
    layers = len(hidden_sizes)
    neurons = "-".join(map(str, hidden_sizes))
    return f"layers-{layers}_neurons-{neurons}"

def save_artifacts(model, train_hist, val_hist, training_time, model_details, dataset_name, save_path, seed):
    """
    Generates model name, creates save path, and saves model and history.
    """
    model_name = get_short_model_name(dataset_name, model_details, seed)
    history = {
        "train_loss": train_hist, 
        "validation_loss": val_hist,
        "training_time": training_time,
        "model_details": model_details
        }
    save_model_and_history(model, history, save_path, model_name)


if __name__ == '__main__':
    # For module tessting purposes

    # --- 1. Configuration ---
    INPUT_FEATURES = 2
    # A shallow but sufficiently large NN as a starting point
    HIDDEN_LAYERS = [64, 64, 64] 
    OUTPUT_FEATURES = 1
    
    EPOCHS = 200
    LEARNING_RATE = 0.001
    PATIENCE = 15 # For early stopping
    BATCH_SIZE = 32

    # --- 2. Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Dummy Data Generation (to be replaced by Phase 1) ---
    # y = 2 * x1^2 - 3 * x2 + 5
    X_train = torch.rand(1000, 2) * 4 - 2 # 1000 samples, 2 features, range [-2, 2]
    y_train = 2 * X_train[:, 0]**2 - 3 * X_train[:, 1] + 5 + torch.randn(1000) * 0.1 # with noise
    y_train = y_train.unsqueeze(1)

    X_val = torch.rand(200, 2) * 4 - 2 # 200 validation samples
    y_val = 2 * X_val[:, 0]**2 - 3 * X_val[:, 1] + 5 + torch.randn(200) * 0.1
    y_val = y_val.unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # --- 4. Model Initialization and Training ---
    # To assess stability, you can run this section in a loop with different random seeds.
    model = SimpleNN(input_size=INPUT_FEATURES, hidden_sizes=HIDDEN_LAYERS, output_size=OUTPUT_FEATURES)
    print("\nModel Architecture:")
    print(model)

    print("\nStarting Training...")
    trained_model, train_hist, val_hist, training_time = train_model(
        model, train_loader, val_loader, EPOCHS, LEARNING_RATE, PATIENCE, device
    )

    print(f"\nTraining finished in {training_time:.2f} seconds.")
    # The 'trained_model' is now ready for Phase 3: iML Analysis.