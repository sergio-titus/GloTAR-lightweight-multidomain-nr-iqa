"""KAN regression head. Wraps efficient-kan if installed; falls back to an MLP with a
loud warning so training never silently breaks if the KAN dependency isn't available.
This also makes the KAN-vs-MLP ablation (mentioned in the proposal) trivial: just force
use_mlp_fallback=True and rerun."""
import warnings
import torch
import torch.nn as nn

try:
    from efficient_kan import KAN as _EfficientKAN
    _HAS_KAN = True
except ImportError:
    _HAS_KAN = False


class KANHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims, out_dim: int, force_mlp_fallback: bool = False):
        super().__init__()
        self.use_kan = _HAS_KAN and not force_mlp_fallback
        layer_dims = [in_dim] + list(hidden_dims) + [out_dim]

        if self.use_kan:
            self.net = _EfficientKAN(layer_dims)
        else:
            if not force_mlp_fallback:
                warnings.warn(
                    "efficient-kan not installed -- falling back to an MLP head. "
                    "Install with `pip install git+https://github.com/Blealtan/efficient-kan.git` "
                    "to use the intended KAN regression head."
                )
            mlp_layers = []
            for a, b in zip(layer_dims[:-1], layer_dims[1:]):
                mlp_layers += [nn.Linear(a, b), nn.SiLU()]
            mlp_layers = mlp_layers[:-1]  # drop trailing activation
            self.net = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.net(x)

    def spline_activations(self, x):
        """Returns per-layer spline activation values for interpretability (Step 6).
        Only meaningful when self.use_kan is True -- raises otherwise so callers notice
        they're looking at an MLP fallback and the explanation isn't a real KAN spline."""
        if not self.use_kan:
            raise RuntimeError("spline_activations() requires the real KAN backend, not the MLP fallback.")
        activations = []
        h = x
        for layer in self.net.layers:
            h = layer(h)
            activations.append(h.detach().cpu())
        return activations
