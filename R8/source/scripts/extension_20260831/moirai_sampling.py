"""Exact one-step marginal sampling for univariate Moirai 1.1.

The upstream path samples all context tokens and all 128 padded coordinates
for every candidate patch, then discards almost everything. This path keeps
the same validation loss and patch choice, slices the selected one-step law,
and samples only that marginal. It changes the RNG stream, not the law.
"""
import torch
from torch.distributions import Categorical
from uni2ts.distribution._base import AffineTransformed
from uni2ts.distribution.mixture import Mixture


def marginal(distr, token):
    def take(x):
        return torch.broadcast_to(x, distr.batch_shape)[:, token, 0]
    if isinstance(distr, AffineTransformed):
        return AffineTransformed(marginal(distr.base_dist, token), loc=take(distr.loc), scale=take(distr.scale))
    if isinstance(distr, Mixture):
        return Mixture(Categorical(logits=distr.weights.logits[:, token, 0, :]),
                       [marginal(c, token) for c in distr.components], validate_args=False)
    # Use logits rather than the alternative probs parameter for count laws.
    keys = [k for k in distr.arg_constraints if k != 'probs']
    return type(distr)(**{k:take(getattr(distr,k)) for k in keys}, validate_args=False)


def forward(self, past_target, past_observed_target, past_is_pad, **kwargs):
    if self.hparams.prediction_length != 1 or self.hparams.target_dim != 1 or self.hparams.patch_size != 'auto':
        raise ValueError('Optimised sampler only supports the stated scalar one-step auto-patch design')
    if any(v is not None for k,v in kwargs.items() if k != 'num_samples'):
        raise ValueError('Dynamic covariates are outside this scalar forecast design')
    losses = torch.stack([self._val_loss(patch_size=p,
                                        target=past_target[..., :self.past_length, :],
                                        observed_target=past_observed_target[..., :self.past_length, :],
                                        is_pad=past_is_pad[..., :self.past_length]) for p in self.module.patch_sizes])
    chosen = losses.argmin(dim=0)
    n = kwargs.get('num_samples') or self.hparams.num_samples
    out = torch.empty((len(past_target), n, 1), device=past_target.device)
    for j,p in enumerate(self.module.patch_sizes):
        keep = chosen == j
        if not keep.any():
            continue
        d = self._get_distr(p, past_target[keep, -self.hparams.context_length:, :],
                           past_observed_target[keep, -self.hparams.context_length:, :],
                           past_is_pad[keep, -self.hparams.context_length:])
        scalar = marginal(d, self.context_token_length(p))
        out[keep, :, 0] = scalar.sample(torch.Size((n,))).T
    self.last_selected_patches = torch.tensor(self.module.patch_sizes,device=chosen.device)[chosen].cpu().numpy()
    return out
