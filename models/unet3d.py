from __future__ import annotations

from monai.networks.nets import UNet


class UNet3D(UNet):
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_channels: int = 16, num_levels: int = 4, norm: str = "instance", activation: str = "relu"):
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if num_levels < 2:
            raise ValueError("num_levels must be at least 2")
        channels = tuple(base_channels * (2 ** level) for level in range(num_levels))
        strides = (2,) * (num_levels - 1)
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=3,
            up_kernel_size=3,
            num_res_units=2,
            act=activation.upper(),
            norm=norm.upper(),
        )
