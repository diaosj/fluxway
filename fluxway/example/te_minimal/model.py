import torch
import torch.nn.functional as F

from fluxway.runtime.te import require_te


class TEMinimalMLP(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        te = require_te()

        self.fc1 = te.Linear(
            hidden_size,
            ffn_hidden_size,
            bias=True,
            params_dtype=params_dtype,
        )
        self.fc2 = te.Linear(
            ffn_hidden_size,
            hidden_size,
            bias=True,
            params_dtype=params_dtype,
        )

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


def compute_mse_loss(pred, target):
    return F.mse_loss(pred.float(), target.float())

