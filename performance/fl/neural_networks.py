import jax
import jax.numpy as jnp
import flax.linen as nn
import einops


class FCN(nn.Module):
    classes: int
    pw: float = 1.0
    pd: float = 1.0
    scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for l in range(round(10 * self.pd)):
            x = nn.Dense(round(1000 * self.pw), name=f"Dense{l}")(x)
            x *= self.scale  # This goes before activation or batch norms, and is used by heterofl
            x = nn.relu(x)
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        return x


class CNN(nn.Module):
    "A network based on the VGG16 architecture"
    classes: int
    pw: float = 1.0
    pd: float = 1.0
    scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        for l in range(5):
            if l < round(5 * self.pd):
                x = nn.Conv(round(32 * (2**l) * self.pw), kernel_size=(3, 3), name=f"Conv{l}_1")(x)
                x = x * self.scale
                x = nn.relu(x)
                x = nn.Conv(round(32 * (2**l) * self.pw), kernel_size=(3, 3), name=f"Conv{l}_2")(x)
                x = x * self.scale
                x = nn.relu(x)
            x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(round(128 * self.pw), name="Dense1")(x)
        x = x * self.scale
        x = nn.relu(x)
        x = nn.Dense(round(128 * self.pw), name="Dense2")(x)
        x = x * self.scale
        x = nn.relu(x)
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        return x


# DenseNet based architecture
# TODO: Test with TinyImageNet, see performance w/ group norm

class ConvBlock(nn.Module):
    growth_rate: int
    name: str
    scale: float = 1.0

    @nn.compact
    def __call__(self, x, train=True):
        # x1 = nn.BatchNorm(axis=3, epsilon=1.001e-5, name=self.name + '_0_bn', use_running_average=not train)(x)
        x1 = nn.GroupNorm(32, epsilon=1.001e-5, use_bias=False, name=self.name + '_0_gn')(x)
        x1 = nn.relu(x1)
        x1 = nn.Conv(4 * self.growth_rate, (1, 1), padding='VALID', use_bias=False, name=self.name + '_1_conv')(x1)
        x1 *= self.scale
        # x1 = nn.BatchNorm(axis=3, epsilon=1.001e-5, name=self.name + '_1_bn', use_running_average=not train)(x1)
        x1 = nn.GroupNorm(32, epsilon=1.001e-5, use_bias=False, name=self.name + '_1_gn')(x1)
        x1 = nn.relu(x1)
        x1 = nn.Conv(self.growth_rate, (3, 3), padding='SAME', use_bias=False, name=self.name + '_2_conv')(x1)
        x1 *= self.scale
        x = jnp.concatenate((x, x1), axis=3)
        return x


class TransitionBlock(nn.Module):
    reduction: float
    name: str
    scale: float = 1.0

    @nn.compact
    def __call__(self, x, train=True):
        # x = nn.BatchNorm(axis=3, epsilon=1.001e-5, name=self.name + '_bn', use_running_average=not train)(x)
        x = nn.GroupNorm(32, epsilon=1.001e-5, use_bias=False, name=self.name + '_gn')(x)
        x = nn.relu(x)
        x = nn.Conv(int(x.shape[3] * self.reduction), (1, 1), padding='VALID', use_bias=False, name=self.name + '_conv')(x)
        x *= self.scale
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        return x


class DenseBlock(nn.Module):
    blocks: list[int]
    name: str
    pw: float = 1.0
    pd: float = 1.0
    scale: float = 1.0

    @nn.compact
    def __call__(self, x, train=True):
        for i in range(round(self.blocks * self.pd)):
            x = ConvBlock(round(32 * self.pw), name=f"{self.name}_block{i + 1}", scale=self.scale)(x, train)
        return x


class DenseNet121(nn.Module):
    classes: int
    pw: float = 1.0
    pd: float = 1.0
    scale: float = 1.0

    @nn.compact
    def __call__(self, x, train=True):
        x = jnp.pad(x, ((0, 0), (3, 3), (3, 3), (0, 0)))
        x = nn.Conv(64, (7, 7), (2, 2), padding='VALID', use_bias=False, name="conv1/conv")(x)
        x *= self.scale
        # x = nn.BatchNorm(axis=3, epsilon=1.001e-5, name='conv1/bn', use_running_average=not train)(x)
        x = nn.GroupNorm(32, epsilon=1.001e-5, use_bias=False, name="conv1/gn")(x)
        x = nn.relu(x)
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        x = nn.max_pool(x, (3, 3), (2, 2))
        x = DenseBlock(6, self.pw, self.pd, scale=self.scale, name="conv2")(x, train)
        x = TransitionBlock(0.5, name="pool2", scale=self.scale)(x, train)
        x = DenseBlock(12, self.pw, self.pd, scale=self.scale, name="conv3")(x, train)
        x = TransitionBlock(0.5, name="pool3", scale=self.scale)(x, train)
        x = DenseBlock(24, self.pw, self.pd, scale=self.scale, name="conv4")(x, train)
        x = TransitionBlock(0.5, name="pool4", scale=self.scale)(x, train)
        x = DenseBlock(16, self.pw, self.pd, scale=self.scale, name="conv5")(x, train)
        # x = nn.BatchNorm(axis=3, epsilon=1.001e-5, name="bn", use_running_average=not train)(x)
        x = nn.GroupNorm(32, epsilon=1.001e-5, use_bias=False, name="gn")(x)
        x = nn.relu(x)
        x = einops.reduce(x, "b w h d -> b d", "mean")  # Global average pooling
        x = nn.Dense(self.classes, name="predictions")(x)
        x = nn.softmax(x)
        return x


# ConvNext

class ConvNeXt(nn.Module):
    classes: int
    pw: float = 1.0
    pd: float = 1.0
    scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        depths = [3, 3, 27, 3]
        projection_dims = [128, 256, 512, 1024]
        # Stem block.
        stem = nn.Sequential([
            nn.Conv(projection_dims[0], (4, 4), strides=(4, 4), name="convnext_base_stem_conv"),
            nn.LayerNorm(epsilon=1e-6, name="convnext_base_stem_layernorm"),
        ])

        # Downsampling blocks.
        downsample_layers = [stem]

        num_downsample_layers = 3
        for i in range(num_downsample_layers):
            downsample_layer = nn.Sequential([
                nn.LayerNorm(epsilon=1e-6, name=f"convnext_base_downsampling_layernorm_{i}"),
                nn.Conv(projection_dims[i + 1], (2, 2), strides=(2, 2), name=f"convnext_base_downsampling_conv_{i}"),
            ])
            downsample_layers.append(downsample_layer)

        num_convnext_blocks = 4
        for i in range(num_convnext_blocks):
            x = downsample_layers[i](x)
            for j in range(depths[i]):
                x = ConvNeXtBlock(
                    projection_dim=projection_dims[i],
                    layer_scale_init_value=1e-6,
                    name=f"convnext_base_stage_{i}_block_{j}",
                )(x)

        x = einops.reduce(x, 'b h w c -> b c', 'mean')
        x = nn.LayerNorm(epsilon=1e-6, name="convnext_base_head_layernorm")(x)
        x = nn.Dense(self.classes, name="convnext_base_head_dense")(x)
        x = nn.softmax(x)
        return x


class ConvNeXtBlock(nn.Module):
    projection_dim: int
    name: str = None
    layer_scale_init_value: float = 1e-6

    @nn.compact
    def __call__(self, inputs):
        x = inputs

        x = nn.Conv(
            self.projection_dim,
            kernel_size=(7, 7),
            padding="SAME",
            feature_group_count=self.projection_dim,
            name=self.name + "_depthwise_conv",
        )(x)
        x = nn.LayerNorm(epsilon=1e-6, name=self.name + "_layernorm")(x)
        x = nn.Dense(4 * self.projection_dim, name=self.name + "_pointwise_conv_1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.projection_dim, name=self.name + "_pointwise_conv_2")(x)

        if self.layer_scale_init_value is not None:
            x = LayerScale(
                self.layer_scale_init_value,
                self.projection_dim,
                name=self.name + "_layer_scale",
            )(x)

        return inputs + x


class LayerScale(nn.Module):
    init_values: float
    projection_dim: int
    name: str
    param_dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        gamma = self.param(
            'gamma',
            nn.initializers.constant(self.init_values, dtype=self.param_dtype),
            (self.projection_dim,),
            self.param_dtype,
        )
        return x * gamma