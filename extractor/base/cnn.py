import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from policy.layers.building_blocks import MLP


def get_flattened_cnn_output_size(input_shape, conv_layers):
    dummy_input = torch.zeros(1, *input_shape)  # batch size = 1
    dummy_input = dummy_input.permute((0, 3, 2, 1))
    model = nn.Sequential(*conv_layers)
    with torch.no_grad():
        output = model(dummy_input)

    feature_image_dim = output.shape[1:]

    return feature_image_dim, output.view(1, -1).size(1)


class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class Reshape(nn.Module):
    def __init__(self, reshape_dim):
        super(Reshape, self).__init__()
        self.reshape_dim = reshape_dim

    def forward(self, x):
        return x.view(-1, *self.reshape_dim)


class CNN(nn.Module):
    def __init__(
        self,
        state_dim: tuple,
        action_dim: int,
        feature_dim: int,
        activation: nn.Module = nn.Tanh(),
        device: torch.device = torch.device("cpu"),
    ):
        super(CNN, self).__init__()

        # Parameters
        self.state_dim = state_dim
        self.width, self.height, self.in_channel = self.state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.device = device

        self.logstd_range = (-5, 2)

        ### Encoding module
        self.en_pmt = Permute((0, 3, 1, 2))
        self.encoder_cnn = [
            nn.Conv2d(self.in_channel, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        ]

        self.feature_image_dim, self.flatten_dim = get_flattened_cnn_output_size(
            state_dim, self.encoder_cnn
        )

        self.encoder_fc = MLP(
            input_dim=self.flatten_dim,
            hidden_dims=[int(self.flatten_dim / 2), int(self.flatten_dim / 4)],
            output_dim=feature_dim,
            activation=activation,
        )

        self.encoder = nn.Sequential(
            *self.encoder_cnn, nn.Flatten(), self.encoder_fc, nn.Sigmoid()
        )

        # self.mu = nn.Linear(
        #     in_features=self.flatten_dim,
        #     out_features=feature_dim,
        # )
        # self.logstd = nn.Linear(
        #     in_features=self.flatten_dim,
        #     out_features=feature_dim,
        # )

        ### Decoding module
        self.de_latent = MLP(
            input_dim=feature_dim,
            hidden_dims=[int(self.flatten_dim / 2)],
            activation=activation,
        )

        self.de_action = MLP(
            input_dim=action_dim,
            hidden_dims=[int(self.flatten_dim / 2)],
            activation=activation,
        )

        self.decoder_fc = MLP(
            input_dim=self.flatten_dim,
            hidden_dims=[self.flatten_dim, self.flatten_dim],
            output_dim=self.flatten_dim,
            activation=activation,
        )

        self.decoder = nn.Sequential(
            Reshape(reshape_dim=self.feature_image_dim),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, self.in_channel, kernel_size=3, stride=1, padding=1),
            Permute((0, 2, 3, 1)),
        )

        self.to(self.device)

    def forward(self, state: torch.Tensor, deterministic: bool = True):
        if len(state.shape) < 3:
            state = state.view(state.size(0), *self.state_dim)

        state = self.en_pmt(state)
        features = self.encoder(state)

        # mu = self.mu(logits)
        # logstd = torch.clamp(
        #     self.logstd(logits),
        #     min=self.logstd_range[0],
        #     max=self.logstd_range[1],
        # )
        # std = torch.exp(logstd)

        # if deterministic:
        #     feature = mu
        # else:
        # cov = torch.diag_embed(std**2)
        # dist = MultivariateNormal(loc=mu, covariance_matrix=cov)

        # feature = dist.rsample()

        return features, {"loss": torch.tensor(0.0).to(self.device)}

    def decode(self, features: torch.Tensor, actions: torch.Tensor):
        print(features)
        out1 = self.de_latent(features)
        out2 = self.de_action(actions)
        logits = torch.cat((out1, out2), axis=-1)

        logits = self.decoder_fc(logits)
        reconstructed_state = self.decoder(logits)
        return reconstructed_state
