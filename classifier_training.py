import os
import argparse
import pickle
import numpy as np
import json
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb
from sklearn.metrics import accuracy_score

from car_quality_estimator.individual_image_classifier import IndividualImageClassifier, IndividualImageTransformer


class IndividualImageCarDataset(Dataset):
    """
    A PyTorch Dataset that reshapes sequence embeddings into individual samples at initialization.
    
    This dataset flattens the input embeddings from [n_objects, seq_len, embed_dim] to 
    [n_objects*seq_len, embed_dim] and replicates labels and UIDs for each image in the sequence.
    
    Args:
        embeddings (numpy.ndarray): Embeddings of shape [n_objects, seq_len, embed_dim]
        labels (numpy.ndarray): Labels of shape [n_objects]
        uids (numpy.ndarray): UIDs of shape [n_objects]
    """

    def __init__(self, embeddings, labels, uids):
        # Store original shapes for reference
        self.n_objects = embeddings.shape[0]
        self.seq_len = embeddings.shape[1]
        self.embed_dim = embeddings.shape[2]

        # Reshape embeddings from [n_objects, seq_len, embed_dim] to [n_objects*seq_len, embed_dim]
        self.embeddings = embeddings.reshape(-1, self.embed_dim)

        # Duplicate labels for each image in the sequence
        # From [n_objects] to [n_objects*seq_len]
        expanded_labels = []
        expanded_uids = []
        expanded_object_indices = []
        expanded_seq_indices = []

        for obj_idx in range(self.n_objects):
            for seq_idx in range(self.seq_len):
                expanded_labels.append(labels[obj_idx])
                expanded_uids.append(uids[obj_idx])
                expanded_object_indices.append(obj_idx)
                expanded_seq_indices.append(seq_idx)

        self.labels = np.array(expanded_labels)
        self.uids = np.array(expanded_uids)
        self.object_indices = np.array(expanded_object_indices)
        self.seq_indices = np.array(expanded_seq_indices)

        print(
            f"Reshaped dataset from {self.n_objects} objects with {self.seq_len} views each")
        print(
            f"to {len(self.embeddings)} individual image embeddings of dimension {self.embed_dim}")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        """
        Get an individual image embedding with its label, object index, and sequence index.
        
        Args:
            idx (int): Index of the image embedding
            
        Returns:
            tuple: (embedding, label, object_idx, seq_idx)
                - embedding: The image embedding vector
                - label: The class label
                - object_idx: The index of the original object
                - seq_idx: The index of this view in the original sequence
        """
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        object_idx = self.object_indices[idx]
        seq_idx = self.seq_indices[idx]

        return embedding, label, object_idx, seq_idx


def evaluate_model(model, val_loader, criterion, device):
    """
    Evaluate the model on validation data with multi-view averaging.
    
    Args:
        model (nn.Module): The model to evaluate
        val_loader (DataLoader): DataLoader for validation data
        criterion (nn.Module): Loss function
        device (str): Device to run evaluation on
        
    Returns:
        tuple: (validation loss, validation accuracy)
    """
    model.eval()
    running_loss = 0.0

    # Dictionary to store predictions for each object
    object_predictions = {}
    object_labels = {}

    with torch.no_grad():
        for data in val_loader:
            embeddings, labels, object_idxs, seq_idxs = data
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            # Forward pass on individual images
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Store predictions for each object and image
            for i, (obj_idx, seq_idx) in enumerate(zip(object_idxs, seq_idxs)):
                obj_idx = obj_idx.item()
                seq_idx = seq_idx.item()

                if obj_idx not in object_predictions:
                    object_predictions[obj_idx] = {}
                    object_labels[obj_idx] = labels[i].item()

                object_predictions[obj_idx][seq_idx] = outputs[i].detach(
                ).cpu()

    # Average predictions for each object
    all_preds = []
    all_labels = []

    for obj_idx, seq_preds in object_predictions.items():
        # Average predictions across all images for this object
        avg_pred = torch.mean(torch.stack(list(seq_preds.values())), dim=0)
        predicted_class = torch.argmax(avg_pred).item()

        all_preds.append(predicted_class)
        all_labels.append(object_labels[obj_idx])

    val_accuracy = accuracy_score(all_labels, all_preds)
    val_loss = running_loss / len(val_loader)

    return val_loss, val_accuracy

def main(args):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"  # Use MPS for Apple Silicon if available

    # create output path if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Initialize wandb
    wandb.init(
        project="car-quality-classifier",
        config={
            "model_type": args.model_type,
            "embedding_model": args.embedding_model,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.n_classes,
            "dropout": args.dropout,
            "num_layers": args.num_layers,
            "nhead": args.nhead,
            "transformer_reshape_dim": args.transformer_reshape_dim,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "n_epochs": args.n_epochs,
            "use_balanced_sampling": args.use_balanced_sampling,
            "amount_of_embedding_splits_to_keep": args.amount_of_embedding_splits_to_keep,
            "individual_image_processing": True
        }
    )

    # load embeddings from an HDF5 file
    base_folder = './data/'
    ending = '_seq_4'
    if args.embedding_model == "combined":
        dino_filename = base_folder + 'car_model_embedding_' + \
            "DINOv2" + ending + '_reduced.h5'
        with h5py.File(dino_filename, 'r') as f:
            all_dino_embeddings = f['embedding_dataset'][:]
        siglip_filename = base_folder + 'car_model_embedding_' + "siglip" + ending + '.h5'
        with h5py.File(siglip_filename, 'r') as f:
            all_siglip_embeddings = f['embedding_dataset'][:]
        all_embeddings = np.concatenate(
            (all_dino_embeddings, all_siglip_embeddings), axis=2)
        print(f"Combined embeddings shape: {all_embeddings.shape}")
    else:
        if args.embedding_model == "DINOv2":
            filename = base_folder + 'car_model_embedding_' + \
                args.embedding_model + ending + '_reduced.h5'
        else:
            filename = base_folder + 'car_model_embedding_' + \
                args.embedding_model + ending + '.h5'
        with h5py.File(filename, 'r') as f:
            all_embeddings = f['embedding_dataset'][:]
            print(f"Embeddings shape: {all_embeddings.shape}")

    # load the votes from an HDF5 file
    filename = base_folder + 'car_model_votes.h5'
    with h5py.File(filename, 'r') as f:
        votes = f['vote_dataset'][:]
        votes = votes - 1
        print(f"Votes shape: {votes.shape}")

    # load the uids from an HDF5 file
    filename = base_folder + 'car_model_uids.h5'
    with h5py.File(filename, 'r') as f:
        uids = f['uid_dataset'][:]
        print(f"UIDs shape: {uids.shape}")

    # remap the votes according to number of classes
    n_classes = args.n_classes
    if n_classes == 2:
        new_votes = [1 if vote == 4 or vote == 3 else 0 for vote in votes]
    elif n_classes == 3:
        new_votes = [2 if vote == 4 or vote == 3 else 1 if vote ==
                     2 or vote == 1 else 0 for vote in votes]
    elif n_classes == 4:
        new_votes = [3 if (vote == 4 or vote == 3) else 2 if (
            vote == 2) else 1 if (vote == 1) else 0 for vote in votes]
    elif n_classes == 5:
        new_votes = [4 if (vote == 4) else 3 if (vote == 3) else 2 if (
            vote == 2) else 1 if (vote == 1) else 0 for vote in votes]
    else:
        raise ValueError("Output dimension must be 2, 3, 4 or 5")

    # get all the embeddings and labels and split them into training and validation sets
    all_labels = np.array(new_votes)
    n_samples = len(all_embeddings)
    amount_of_embedding_splits = 1
    amount_of_embedding_splits_to_keep = args.amount_of_embedding_splits_to_keep
    if amount_of_embedding_splits_to_keep > amount_of_embedding_splits:
        amount_of_embedding_splits_to_keep = amount_of_embedding_splits

    # sample train and test set
    n_split_samples = n_samples // amount_of_embedding_splits
    indices = np.random.permutation(n_split_samples)
    first_train_indices = indices[:int(0.8 * n_split_samples)]
    first_val_indices = indices[int(0.8 * n_split_samples):]

    # apply the same split to all N times the same data
    train_indices = []
    val_indices = []
    for i in range(amount_of_embedding_splits_to_keep):
        train_indices_tmp = first_train_indices + i * n_split_samples
        val_indices_tmp = first_val_indices + i * n_split_samples
        train_indices.extend(train_indices_tmp)
        val_indices.extend(val_indices_tmp)
    print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")

    train_embeddings = np.array([all_embeddings[i] for i in train_indices])
    train_labels = np.array([all_labels[i] for i in train_indices])
    train_uids = np.array([uids[i] for i in train_indices])
    print(f"Training embeddings shape: {train_embeddings.shape}")

    val_embeddings = np.array([all_embeddings[i] for i in val_indices])
    val_labels = np.array([all_labels[i] for i in val_indices])
    val_uids = np.array([uids[i] for i in val_indices])

    # Create datasets that treat each image individually
    train_dataset = IndividualImageCarDataset(
        train_embeddings, train_labels, train_uids)
    val_dataset = IndividualImageCarDataset(
        val_embeddings, val_labels, val_uids)

    # create dataloaders
    batch_size = args.batch_size
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the model
    input_dim = all_embeddings.shape[2]  # Embedding dimension

    if args.model_type == "transformer":
        model = IndividualImageTransformer(
            input_dim=input_dim,
            dim_feedforward=args.hidden_dim,
            output_dim=n_classes,
            nhead=args.nhead,
            num_layers=args.num_layers,
            reshape_dim=args.transformer_reshape_dim,
            dropout=args.dropout
        ).to(device)
    else:
        model = IndividualImageClassifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=n_classes,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)

    # Save model configuration
    model_config = {
        "model_type": args.model_type,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "output_dim": n_classes,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "nhead": args.nhead if args.model_type == "transformer" else None,
        "transformer_reshape_dim": args.transformer_reshape_dim if args.model_type == "transformer" else None,
        "individual_image_processing": True
    }

    with open(os.path.join(args.output_path, f"car_quality_model_{args.embedding_model}_{args.model_type}_individual.json"), 'w') as f:
        json.dump(model_config, f)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(
        f"Amount of parameters: {sum(p.numel() for p in model.parameters())}")

    # Train the model
    n_epochs = args.n_epochs
    train_losses = []
    val_losses = []
    best_val_accuracy = 0

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for data in train_loader:
            embeddings, labels, _, _ = data
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # Process individual images directly - no need for reshaping
            outputs = model(embeddings)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Evaluate on validation set
        val_loss, val_accuracy = evaluate_model(
            model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Log metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch": epoch
        })

        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(
                args.output_path, f"best_model_individual")
            # Save in PyTorch format (preferred)
            torch.save(model.state_dict(), f"{best_model_path}.pt")

    # Save final model
    final_model_path = os.path.join(
        args.output_path, f"car_quality_model_{args.embedding_model}_{args.model_type}_individual")
    # Save in PyTorch format (preferred)
    torch.save(model.state_dict(), f"{final_model_path}.pt")
    print(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train individual image classifier for vehicle quality assessment")
    parser.add_argument("--model_type", default="transformer",
                        type=str, choices=["standard", "transformer"])
    parser.add_argument("--output_path", default="./Experiments", type=str)
    parser.add_argument("--embedding_model", default="combined", type=str)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--n_classes", default=2,
                        type=int, choices=[2, 3, 4, 5])
    parser.add_argument("--dropout", default=0.369, type=float)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--nhead", default=4, type=int)
    parser.add_argument("--transformer_reshape_dim", default=32, type=int)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=200, type=int)
    parser.add_argument("--learning_rate", default=0.00046, type=float)
    parser.add_argument("--weight_decay", default=0.00029, type=float)
    parser.add_argument("--use_balanced_sampling",
                        default=False, action="store_true")
    parser.add_argument(
        "--amount_of_embedding_splits_to_keep", default=1, type=int)
    args = parser.parse_args()

    main(args)
