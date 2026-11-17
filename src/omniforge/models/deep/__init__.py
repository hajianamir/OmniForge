"""Deep learning architectures: MLP, CNN, RNN/LSTM/GRU, FT-Transformer, TabNet, ResNet-Tabular, GCN/GraphSAGE/GAT/GAE, Autoencoders, Multimodal."""

from omniforge.models.deep.autoencoders import AutoencoderModel
from omniforge.models.deep.cnn import CNN1DModel
from omniforge.models.deep.gnn import GraphAutoencoderLinkModel, GraphNeuralNetworkModel
from omniforge.models.deep.mlp import MLPModel
from omniforge.models.deep.multimodal import MultimodalFusionModel
from omniforge.models.deep.resnet_tabular import ResNetTabularModel
from omniforge.models.deep.rnn_lstm import RecurrentSequenceModel
from omniforge.models.deep.tabnet import TabNetModel
from omniforge.models.deep.transformers import FTTransformerModel, TabularTransformerModel

__all__ = [
    "AutoencoderModel",
    "CNN1DModel",
    "FTTransformerModel",
    "GraphAutoencoderLinkModel",
    "GraphNeuralNetworkModel",
    "MLPModel",
    "MultimodalFusionModel",
    "RecurrentSequenceModel",
    "ResNetTabularModel",
    "TabNetModel",
    "TabularTransformerModel",
]
