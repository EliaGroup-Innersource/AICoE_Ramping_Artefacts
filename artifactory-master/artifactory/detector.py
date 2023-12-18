from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import grad_norm
from modeling import SinusoidalPositionEmbedding, _convolutions, _linear
from torch.nn import (Dropout, Linear, TransformerEncoder,
                      TransformerEncoderLayer)
from torch.nn.functional import mse_loss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ConvolutionDetector(LightningModule):
    """Use only convolutional layers."""

    def __init__(
        self,
        convolution_features: list[int],
        convolution_width: list[int],
        convolution_dilation: list[int],
        convolution_dropout: float = 0.0,
        convolution_activation: str = "sigmoid",
        loss: str = "mask",
    ):
        super().__init__()
        self.loss = loss
        self.save_hyperparameters()
        self.convolutions = _convolutions(
            [1] + convolution_features,
            convolution_width,
            convolution_dilation,
            convolution_dropout,
            convolution_activation,
            pad=True,
        )

    def forward(self, x):
        """
        Input: (batch, window)
        Output: (batch, window)
        """
        x = x.unsqueeze(1)
        x = self.convolutions(x)
        x = x.max(dim=1)[0]
        return x.squeeze()

    def training_step(self, batch, _):
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss]
        loss = mse_loss(y, m)
        self.log("train", loss)
        return loss

    def validation_step(self, batch, _):
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss]
        loss = mse_loss(y, m)
        loss_fp = mse_loss(y[m == 0], m[m == 0])
        self.log("validation", loss)
        self.log("validation (false positives)", loss_fp)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=1,
            min_lr=1e-6,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1000,
                "name": "learning_rate",
                "strict": True,
                "monitor": "validation",
            },
        }


class WindowTransformerDetector(LightningModule):
    """Use convolutional layers to extract features, then use them
       in a transformer block"""

    def __init__(
        self,
        window: int,
        convolution_features: list[int],
        convolution_width: list[int],
        convolution_dropout: float,
        transformer_heads: int,
        transformer_feedforward: int,
        transformer_layers: int,
        transformer_dropout: float,
        loss: str = "mask",
        loss_boost_fp: float = 0.0,
    ):
        super().__init__()
        self.window = window
        self.loss = loss
        self.loss_boost_fp = loss_boost_fp
        self.save_hyperparameters()
        # convolutional layers to extract features
        self.convolutions = _convolutions(
            convolution_features=[1] + convolution_features,
            convolution_width=convolution_width,
            convolution_dropout=convolution_dropout,
            pad=True,
        )
        # positional encoding for transformer
        self.position = SinusoidalPositionEmbedding(convolution_features[-1], window)
        self.dropout = Dropout(transformer_dropout)
        # transformer block
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                convolution_features[-1],
                transformer_heads,
                dim_feedforward=transformer_feedforward,
                batch_first=True,
            ),
            num_layers=transformer_layers,
        )
        # final linear block to convert each
        # feature to a single value
        self.linear = Linear(convolution_features[-1], 1)

    def forward(self, x):
        """
        Input: (batch, window)
        Output: (batch, window)
        """
        x = x.unsqueeze(1)
        x = self.convolutions(x)
        # transpose to use convolution features
        # as datapoint embeddings
        x = x.transpose(1, 2)
        # apply position and transformer block
        x = self.position(x)
        # apply dropout before transformer
        x = self.dropout(x)
        x = self.transformer(x)
        # convert output tokens back to predictions
        x = self.linear(x)
        return x.squeeze()

    def training_step(self, batch, _):
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss]
        loss = mse_loss(y, m)
        if self.loss_boost_fp > 0 and self.loss_boost_fp <= 1:
            loss_fp = mse_loss(y[m == 0], m[m == 0])
            loss += self.loss_boost_fp * loss_fp
            self.log("train (false positives)", loss_fp.item())
        self.log("train", loss.item())
        return loss

    def validation_step(self, batch, _):
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss]
        loss = mse_loss(y, m)
        loss_fp = mse_loss(y[m == 0], m[m == 0])
        self.log("validation", loss.item())
        self.log("validation (false positives)", loss_fp.item())
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=1,
            min_lr=1e-6,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1000,
                "name": "learning_rate",
                "strict": True,
                "monitor": "validation",
            },
        }
    
    def on_before_optimizer_step(self, _):
        self.log_dict(grad_norm(self, norm_type=2))


class WindowLinearDetector(LightningModule):
    """Use convolutional layers to extract features, then use them
       in a dense block"""

    def __init__(
        self,
        window: int,
        convolution_features: list[int],
        convolution_width: list[int],
        convolution_dropout: float,
        linear_dropout: float = 0,
        linear_layers: list[int] = None,
        loss: str = "mask",
        loss_boost_fp: float = 0.0,
    ):
        super().__init__()
        self.window = window
        self.loss = loss
        self.loss_boost_fp = loss_boost_fp
        self.save_hyperparameters()
        # convolutional layers to extract features
        self.convolutions = _convolutions(
            convolution_features=[1] + convolution_features,
            convolution_width=convolution_width,
            convolution_dropout=convolution_dropout,
            pad=True,
        )
        # dropout layer before dense block
        self.dropout = Dropout(linear_dropout)
        # final linear block to convert each
        # feature to a single value
        self.linear = _linear([convolution_features[-1] * window,
                               *(linear_layers or list()),
                               window])

    def forward(self, x):
        """
        Input: (batch, window)
        Output: (batch, window)
        """
        x = x.unsqueeze(1)
        x = self.convolutions(x)
        x = self.dropout(x.view(x.shape[0], -1))
        x = self.linear(x)
        # convert output tokens back to predictions
        return x.squeeze()

    def training_step(self, batch, _):
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss]
        loss = mse_loss(y, m)
        if self.loss_boost_fp > 0 and self.loss_boost_fp <= 1:
            loss_fp = mse_loss(y[m == 0], m[m == 0])
            loss += self.loss_boost_fp * loss_fp
            self.log("train (false positives)", loss_fp.item())
        self.log("train", loss.item())
        return loss

    def validation_step(self, batch, _):
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss]
        loss = mse_loss(y, m)
        loss_fp = mse_loss(y[m == 0], m[m == 0])
        self.log("validation", loss.item())
        self.log("validation (false positives)", loss_fp.item())
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-2)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=1,
            min_lr=1e-6,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1000,
                "name": "learning_rate",
                "strict": True,
                "monitor": "validation",
            },
        }
    
    def on_before_optimizer_step(self, _):
        self.log_dict(grad_norm(self, norm_type=2))