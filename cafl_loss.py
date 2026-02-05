import torch
import torch.nn as nn
import torch.nn.functional as F


from asl_focal_loss import create_loss, Cyclical_FocalLoss


class CAFLoss(nn.Module):
    """
    L_CAFL = L_CFL + alpha * L_A + beta * L_SSPCAB

    - L_CFL: Cyclical_FocalLoss / ASLSingleLabel (multi-class)
    - L_A  : adversarial loss (we use a coarse binary view: original vs synthetic)
    - L_SSPCAB: auxiliary loss from model (cost_sspcab)
    """
    def __init__(
        self,
        base_cfl: nn.Module,
        alpha: float = 0.5,
        beta: float = 1e-3,
        tau_w: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.base_cfl = base_cfl
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau_w = float(tau_w)
        self.eps = float(eps)

    def _cfl(self, logits: torch.Tensor, y: torch.Tensor, epoch: int):
        # Cyclical_FocalLoss needs epoch, ASLSingleLabel does not
        if isinstance(self.base_cfl, Cyclical_FocalLoss):
            return self.base_cfl(logits, y, epoch)
        return self.base_cfl(logits, y)

    def _adv(self, logits: torch.Tensor, y: torch.Tensor):
        """
        Adversarial loss in a robust way for CutPaste/CutMask pretext:
        - Treat class 0 (original) as negative
        - Treat all other classes (synthetic views) as positive
        So for 2-way: pos = class1
           for 3-way: pos = class1 or class2 (sum)
        """
        probs = F.softmax(logits, dim=-1)  # (B, C)
        p0 = probs[:, 0].clamp(self.eps, 1.0 - self.eps)           # P(original)
        p_pos = (1.0 - p0).clamp(self.eps, 1.0 - self.eps)         # P(synthetic = not original)

        y_pos = (y != 0)  # synthetic views are positive

        # Separate pos/neg terms then re-weight
        if y_pos.any():
            loss_pos = (-torch.log(p_pos[y_pos])).mean()
        else:
            loss_pos = torch.tensor(0.0, device=logits.device)

        if (~y_pos).any():
            loss_neg = (-torch.log(1.0 - p_pos[~y_pos])).mean()    # = -log(p0)
        else:
            loss_neg = torch.tensor(0.0, device=logits.device)

        return self.tau_w * loss_pos + (1.0 - self.tau_w) * loss_neg

    def forward(self, logits: torch.Tensor, y: torch.Tensor, epoch: int, cost_sspcab: torch.Tensor | None):
        loss_cfl = self._cfl(logits, y, epoch)
        loss_adv = self._adv(logits, y)

        loss_ssp = torch.tensor(0.0, device=logits.device)
        if cost_sspcab is not None:
            # make sure it's a scalar
            loss_ssp = cost_sspcab.mean()

        total = loss_cfl + self.alpha * loss_adv + self.beta * loss_ssp
        return total, loss_cfl.detach(), loss_adv.detach(), loss_ssp.detach()
