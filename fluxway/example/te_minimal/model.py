import torch
import torch.nn.functional as F

from fluxway.runtime.te import require_te


class TEMinimalMLP(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
        *,
        tp_group=None,
        tp_size: int = 1,
    ):
        super().__init__()
        te = require_te()

        column_mode = "column" if tp_size > 1 else None
        row_mode = "row" if tp_size > 1 else None

        self.fc1 = te.Linear(
            hidden_size,
            ffn_hidden_size,
            bias=True,
            params_dtype=params_dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            parallel_mode=column_mode,
        )
        self.fc2 = te.Linear(
            ffn_hidden_size,
            hidden_size,
            bias=True,
            params_dtype=params_dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            parallel_mode=row_mode,
        )

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


def compute_mse_loss(pred, target):
    return F.mse_loss(pred.float(), target.float())
