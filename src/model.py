import torch
from torch import nn

NUM_CLASSES = 10
INPUT_SHAPE_CHW = (1, 128, 173)  # channels, mel bands, time frames (4 s)


class ConvBlock(nn.Sequential):
    """n x (Conv 3x3 -> BatchNorm -> ReLU), then MaxPool 2x2 and optional spatial dropout."""

    def __init__(self, in_channels, out_channels, n_convs=2, dropout=0.0):
        layers = []
        for i in range(n_convs):
            layers += [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.MaxPool2d(2))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class UrbanSoundCNN(nn.Module):
    """VGG-style CNN: conv blocks that double the filters, global average pooling, two FC layers.

    The forward pass returns raw logits of shape (batch, num_classes), to use with CrossEntropyLoss.
    The keyword arguments are the knobs for the hyperparameter experiments.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        in_channels=1,
        n_blocks=4,
        base_filters=32,
        convs_per_block=2,
        conv_dropout=0.1,
        fc_hidden=128,
        dropout=0.3,
        input_shape=INPUT_SHAPE_CHW,
    ):
        super().__init__()
        smallest = min(input_shape[1:])
        if n_blocks < 1 or smallest < 2 ** n_blocks:
            raise ValueError(f"{n_blocks} pooling blocks need inputs of at least {2 ** n_blocks} pixels per side.")
        self.config = dict(
            num_classes=num_classes, in_channels=in_channels, n_blocks=n_blocks, base_filters=base_filters,
            convs_per_block=convs_per_block, conv_dropout=conv_dropout, fc_hidden=fc_hidden, dropout=dropout,
            input_shape=tuple(input_shape),
        )

        channels = [base_filters * 2 ** i for i in range(n_blocks)]
        blocks, previous = [], in_channels
        for out in channels:
            blocks.append(ConvBlock(previous, out, convs_per_block, conv_dropout))
            previous = out
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(previous, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def model_summary(model, input_shape=INPUT_SHAPE_CHW, batch_size=1):
    """Layer table with output shapes, parameters and multiply-accumulates for one input."""
    rows, hooks = [], []

    def hook(name, module, output):
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if isinstance(module, nn.Conv2d):
            k = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
            macs = output[0].numel() * k
        elif isinstance(module, nn.Linear):
            macs = module.in_features * module.out_features
        else:
            macs = 0
        rows.append((name, type(module).__name__, tuple(output.shape[1:]), params, macs))

    for name, module in model.named_modules():
        if name and not list(module.children()):
            hooks.append(module.register_forward_hook(lambda m, i, o, name=name: hook(name, m, o)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            output = model(torch.zeros(batch_size, *input_shape, device=device))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    macs = sum(r[4] for r in rows)
    width = max(len(r[0]) for r in rows) + 2
    lines = [f"{'Layer':<{width}}{'Type':<14}{'Output shape':<22}{'Params':>12}", "-" * (width + 48)]
    lines.append(f"{'input':<{width}}{'':<14}{str(tuple(input_shape)):<22}{'':>12}")
    for name, kind, shape, params, _ in rows:
        lines.append(f"{name:<{width}}{kind:<14}{str(shape):<22}{params:>12,}")
    lines += [
        "-" * (width + 48),
        f"Output shape:          {tuple(output.shape)}",
        f"Total parameters:      {total:,}",
        f"Trainable parameters:  {trainable:,}",
        f"Model size (float32):  {total * 4 / 2 ** 20:.2f} MB",
        f"MACs per spectrogram:  {macs / 1e6:,.1f} M",
    ]
    return "\n".join(lines)