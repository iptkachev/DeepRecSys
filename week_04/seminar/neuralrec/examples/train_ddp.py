from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from neuralrec.nn.transformer import TransformerEncoder
from neuralrec.data.dataloader import DataLoader
from neuralrec.data.transforms import ToDevice
from neuralrec.ext.yambda.utils import load_yambda_user_split
from neuralrec.ext.yambda.transforms import pad_collate_item_id
from neuralrec.run.train import TrainRunner
from neuralrec.run.callbacks import (
    GradientNormClippingCallback,
    LoggingCallback,
    ValidationCallback,
    TensorBoardCallback,
)
from neuralrec.nn.autocast import AutoCast
from neuralrec.run.distributed import (
    init_process_group,
    destroy_process_group,
    is_chief,
)


@dataclass
class PipelineConfig:
    batch_size: int = 1
    d_model: int = 64
    n_heads: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    num_layers: int = 2
    num_epochs: int = 1
    gradient_clipping: float = 1.0
    logging_every_n_steps: int = 10
    validation_every_n_steps: int = 20
    tensorboard_every_n_steps: int = 10
    learning_rate: float = 1e-3


class RecommenderTransformer(nn.Module):
    def __init__(
        self,
        num_items: int,
        d_model: int,
        encoder: nn.Module,
    ):
        super().__init__()
        self.num_items = num_items
        self.embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.encoder = encoder
        self.to_logits = nn.Linear(d_model, num_items + 1)

    def forward(self, batch) -> dict[str, torch.Tensor]:
        item_id = batch['item_id']
        x = self.embedding(item_id)
        h = self.encoder(x)
        logits = self.to_logits(h)

        logits_pred = logits[:, :-1].contiguous().view(-1, self.num_items + 1)
        target = item_id[:, 1:].contiguous().view(-1)

        loss = nn.functional.cross_entropy(
            logits_pred, target, ignore_index=0, reduction="mean"
        )
        return {"loss": loss}


def main():
    init_process_group()

    config = PipelineConfig()
    train_dataset, valid_dataset, max_item_id = load_yambda_user_split()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=pad_collate_item_id,
        sampler=DistributedSampler(train_dataset),
        transforms=ToDevice(torch.device("cuda"))
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=pad_collate_item_id,
        sampler=DistributedSampler(valid_dataset),
        transforms=ToDevice(torch.device("cuda")),
    )

    model = RecommenderTransformer(
        num_items=max_item_id,
        d_model=config.d_model,
        encoder=TransformerEncoder(
            d_model=config.d_model,
            nhead=config.n_heads,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            causal=True,
        )
    ).cuda()
    model = AutoCast(model)
    model = DDP(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, fused=True)

    TrainRunner(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        callbacks=[
            GradientNormClippingCallback(config.gradient_clipping),
            ValidationCallback(valid_loader).every_n_steps(config.validation_every_n_steps).ignore_if(not is_chief()),
            LoggingCallback().every_n_steps(config.logging_every_n_steps).ignore_if(not is_chief()),
        ],
    ).fit(config.num_epochs)

    destroy_process_group()

if __name__ == "__main__":
    main()
