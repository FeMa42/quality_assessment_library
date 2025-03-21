import math
import json
import pickle
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from car_quality_estimator.embedding_models import generate_siglip_embedding_model, generate_dino_embedding_model_for_individual_images


class IndividualImageClassifier(nn.Module):
    """
    A classifier that processes individual images and outputs class probabilities.
    
    This model is designed to evaluate a single image at a time, rather than a sequence.
    During inference, scores from multiple views can be averaged externally.
    
    Args:
        input_dim (int): Dimension of input embeddings
        hidden_dim (int): Dimension of hidden layers
        output_dim (int): Number of output classes
        num_layers (int): Number of layers in the network
        dropout (float, optional): Dropout probability. Defaults to 0.25
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.25):
        super(IndividualImageClassifier, self).__init__()
        self.input_layer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        self.layers = nn.ModuleList(
            [self._create_layer(hidden_dim, dropout)
             for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def _create_layer(self, hidden_dim, dropout):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )

    def forward(self, x):
        # x shape: [batch_size, embed_dim]
        x = self.input_layer(x)
        # Apply layers with residual connections
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x


class IndividualImageTransformer(nn.Module):
    """
    A lightweight transformer model for individual image classification.
    
    This model uses self-attention to process features within a single image embedding,
    rather than across a sequence of images.
    
    Args:
        input_dim (int): Dimension of input embeddings
        dim_feedforward (int): Dimension of feedforward network in transformer layers
        output_dim (int): Number of output classes
        nhead (int): Number of attention heads in transformer layers
        num_layers (int): Number of transformer encoder layers
        dropout (float, optional): Dropout probability. Defaults to 0.1
    """

    def __init__(self, input_dim, dim_feedforward, output_dim, nhead, num_layers, reshape_dim=32, dropout=0.1):
        super(IndividualImageTransformer, self).__init__()

        # Create a reshape layer to view the input as a "sequence" of feature groups
        # This allows self-attention across feature groups within a single image embedding
        self.reshape_dim = reshape_dim  # Number of feature groups
        self.seq_dim = input_dim // self.reshape_dim  # Size of each feature group

        # Create a positional encoding for the feature groups
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.reshape_dim, self.seq_dim))
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)

        # Create transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=self.seq_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.seq_dim)
        )

        # Output layers
        self.norm = nn.LayerNorm(input_dim)
        self.output_layer = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: [batch_size, embed_dim]
        batch_size = x.shape[0]

        # Reshape input to [batch_size, reshape_dim, seq_dim]
        # This treats the embedding as a "sequence" of feature groups
        x = x.view(batch_size, self.reshape_dim, self.seq_dim)

        # Add positional encoding
        x = x + self.positional_encoding

        # Apply transformer encoder
        x = self.encoder(x)

        # Flatten and normalize
        x = x.reshape(batch_size, -1)
        x = self.norm(x)

        # Final classification
        x = self.output_layer(x)
        x = self.softmax(x)
        return x


class IndividualImageEstimator(nn.Module):
    """
    A PyTorch module that generates embeddings for individual images and classifies them.
    
    This class takes individual images, generates embeddings using a provided embedding function,
    and passes them through a classifier for individual scoring.
    
    Args:
        classifier (nn.Module): The model that processes individual image embeddings.
        embedding_fnc (callable): Function that generates embeddings from input images.
        
    Methods:
        forward(image): Forward pass from image to classification output.
        embed_image(image): Generates embeddings from images.
        predict(image): Returns predicted class indices for input images.
    """
    def __init__(self, classifier, embedding_fnc):
        super(IndividualImageEstimator, self).__init__()
        self.classifier = classifier
        self.embedding_fnc = embedding_fnc
    
    def forward(self, image):
        """Process a single image or batch of individual images"""
        embedding = self.embed_image(image)
        with torch.no_grad():
            outputs = self.classifier(embedding)
            return outputs
    
    def embed_image(self, image):
        """Generate embeddings for a single image or batch of individual images"""
        with torch.no_grad():
            embedding = self.embedding_fnc(image)
            # If the embedding function returns a sequence, take just the first item
            if len(embedding.shape) == 3 and embedding.shape[1] == 1:
                embedding = embedding.squeeze(1)
        return embedding
    
    def predict(self, image):
        """Return predicted class indices"""
        outputs = self.forward(image)
        return outputs.argmax(dim=1)
    
    def predict_multiple_views(self, images_list):
        """
        Process multiple views of an object and average their scores.
        
        Args:
            images_list (list): List of images representing different views of the same object
            
        Returns:
            torch.Tensor: Averaged class probabilities
        """
        all_outputs = []
        
        for image in images_list:
            outputs = self.forward(image)
            all_outputs.append(outputs)
        
        # Stack and average
        stacked_outputs = torch.stack(all_outputs, dim=0)
        avg_outputs = torch.mean(stacked_outputs, dim=0)
        
        return avg_outputs


class CombinedIndividualImageEstimator(nn.Module):
    """
    A neural network module that combines embeddings from two models for individual image classification.
    
    This class takes embeddings from two different models (e.g., SigLIP and DINOv2),
    concatenates them, and passes them through a classifier for individual scoring.
    
    Args:
        classifier (nn.Module): The model that processes combined embeddings.
        siglip_embedding_fnc (callable): Function that generates SigLIP embeddings.
        dino_embedding_fnc (callable): Function that generates DINOv2 embeddings.
        
    Methods:
        forward(image): Forward pass from image to classification output.
        predict(image): Returns predicted class indices for input images.
    """
    def __init__(self, classifier, siglip_embedding_fnc, dino_embedding_fnc):
        super(CombinedIndividualImageEstimator, self).__init__()
        self.classifier = classifier
        self.siglip_embedding_fnc = siglip_embedding_fnc
        self.dino_embedding_fnc = dino_embedding_fnc
    
    def forward(self, image):
        """Process a single image or batch of individual images"""
        with torch.no_grad():
            siglip_embedding = self.siglip_embedding_fnc(image)
            dino_embedding = self.dino_embedding_fnc(image)
            
            # Handle sequence outputs if necessary
            if len(siglip_embedding.shape) == 3 and siglip_embedding.shape[1] == 1:
                siglip_embedding = siglip_embedding.squeeze(1)
            if len(dino_embedding.shape) == 3 and dino_embedding.shape[1] == 1:
                dino_embedding = dino_embedding.squeeze(1)
            
            # Concatenate embeddings
            combined_embedding = torch.cat((siglip_embedding, dino_embedding), dim=1)
            
            # Forward through classifier
            outputs = self.classifier(combined_embedding)
            return outputs
    
    def predict(self, image):
        """Return predicted class indices"""
        outputs = self.forward(image)
        return outputs.argmax(dim=1)
    
    def predict_multiple_views(self, images_list):
        """
        Process multiple views of an object and average their scores.
        
        Args:
            images_list (list): List of images representing different views of the same object
            
        Returns:
            torch.Tensor: Averaged class probabilities
        """
        all_outputs = []
        
        for image in images_list:
            outputs = self.forward(image)
            all_outputs.append(outputs)
        
        # Stack and average
        stacked_outputs = torch.stack(all_outputs, dim=0)
        avg_outputs = torch.mean(stacked_outputs, dim=0)
        
        return avg_outputs


def build_individual_classifier(config_path=None, config=None, model_type="standard"):
    """
    Build an individual image classifier from a JSON configuration file or dict.
    
    Args:
        config_path (str, optional): Path to the JSON configuration file. Defaults to None.
        config (dict, optional): Configuration dictionary. Defaults to None.
        model_type (str, optional): Type of model to build. Defaults to "standard".
            Options: "standard", "transformer"
            
    Returns:
        nn.Module: The constructed model
    """
    if config_path is not None:
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config is None:
        raise ValueError("Either config_path or config must be provided")
    
    input_dim = config.get("input_dim")
    hidden_dim = config.get("hidden_dim")
    output_dim = config.get("output_dim")
    num_layers = config.get("num_layers")
    dropout = config.get("dropout")
    
    if model_type == "standard":
        return IndividualImageClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type == "transformer":
        nhead = config.get("nhead", 4)  # Default to 4 heads if not specified
        reshape_dim = config.get("transformer_reshape_dim", 32)
        return IndividualImageTransformer(
            input_dim=input_dim,
            dim_feedforward=hidden_dim,
            output_dim=output_dim,
            nhead=nhead,
            num_layers=num_layers,
            reshape_dim=reshape_dim,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_individual_siglip_estimator(config_path, weights_path, device="cpu", model_type="standard"):
    """
    Creates a SigLIP-based individual image estimator by loading a model from config and weights.

    Args:
        config_path (str): Path to the JSON configuration file
        weights_path (str): Path to the model weights (.pt file)
        device (str, optional): Device to run model on. Defaults to "cpu"
        model_type (str, optional): Type of model to build. Defaults to "standard"

    Returns:
        IndividualImageEstimator: Loaded model wrapped with SigLIP embeddings
    """
    # Build classifier from config
    classifier = build_individual_classifier(config_path, model_type=model_type)
    
    # Load weights
    classifier.load_state_dict(torch.load(weights_path, map_location=device))
    
    classifier.to(device)
    classifier.eval()
    
    # Create embedding function
    embed_siglip = generate_siglip_embedding_model(device=device)
    
    # Create and return estimator
    return IndividualImageEstimator(classifier, embed_siglip)


def generate_individual_dino_estimator(config_path, weights_path, pca_file_name=None, device="cpu", model_type="standard"):
    """
    Creates a DINOv2-based individual image estimator by loading a model from config and weights.

    Args:
        config_path (str): Path to the JSON configuration file
        weights_path (str): Path to the model weights (.pt file)
        pca_file_name (str, optional): Path to PCA model file. Defaults to None
        device (str, optional): Device to run model on. Defaults to "cpu"
        model_type (str, optional): Type of model to build. Defaults to "standard"

    Returns:
        IndividualImageEstimator: Loaded model wrapped with DINOv2 embeddings
    """
    # Build classifier from config
    classifier = build_individual_classifier(config_path, model_type=model_type)
    
    # Load weights
    classifier.load_state_dict(torch.load(weights_path, map_location=device))
    
    classifier.to(device)
    classifier.eval()
    
    # Create embedding function
    embed_dino = generate_dino_embedding_model_for_individual_images(
        pca_file_name=pca_file_name, 
        device=device)
    
    # Create and return estimator
    return IndividualImageEstimator(classifier, embed_dino)


def generate_individual_combined_estimator(config_path, weights_path, pca_file_name=None, device="cpu", model_type="standard"):
    """
    Creates a combined (SigLIP + DINOv2) individual image estimator.

    Args:
        config_path (str): Path to the JSON configuration file
        weights_path (str): Path to the model weights (.pt file)
        pca_file_name (str, optional): Path to PCA model file for DINOv2. Defaults to None
        device (str, optional): Device to run model on. Defaults to "cpu"
        model_type (str, optional): Type of model to build. Defaults to "standard"

    Returns:
        CombinedIndividualImageEstimator: Loaded model with combined embeddings
    """
    # Build classifier from config
    classifier = build_individual_classifier(config_path, model_type=model_type)
    
    # Load weights
    classifier.load_state_dict(torch.load(weights_path, map_location=device))
    
    classifier.to(device)
    classifier.eval()
    
    # Create embedding functions
    embed_siglip = generate_siglip_embedding_model(device=device)
    embed_dino = generate_dino_embedding_model_for_individual_images(
        pca_file_name=pca_file_name, 
        device=device,
    )
    
    # Create and return estimator
    return CombinedIndividualImageEstimator(classifier, embed_siglip, embed_dino)


