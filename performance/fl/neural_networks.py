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
        for l in range(round(5 * self.pd)):
            x = nn.Conv(round(32 * (2**l) * self.pw), kernel_size=(3, 3), name=f"Conv{l}_1")(x)
            x = x * self.scale
            x = nn.relu(x)
            x = nn.Conv(round(32 * (2**l) * self.pw), kernel_size=(3, 3), name=f"Conv{l}_2")(x)
            x = x * self.scale
            x = nn.relu(x)
            x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = jnp.pad(x, [(0, 0)] + [(0, s * 2**round(5 * self.pd) - s) for s in x.shape[1:-1]] + [(0, 0)])
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128, name="Dense1")(x)
        x = x * self.scale
        x = nn.relu(x)
        x = nn.Dense(128, name="Dense2")(x)
        x = x * self.scale
        x = nn.relu(x)
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        return x


# ResNetRS50 based architecture (https://arxiv.org/pdf/2103.07579.pdf)
# adapted from https://github.com/keras-team/keras/blob/master/keras/applications/resnet_rs.py

def fixed_padding(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return jnp.pad(inputs, ((0, 0), (pad_beg, pad_end), (pad_beg, pad_end), (0, 0)))


class Conv2DFixedPadding(nn.Module):
    filters: int
    kernel_size: int
    strides: int
    name: str = None

    @nn.compact
    def __call__(self, x):
        if self.strides > 1:
            x = fixed_padding(x, self.kernel_size)
        return nn.Conv(
            self.filters,
            (self.kernel_size, self.kernel_size),
            self.strides,
            padding="SAME" if self.strides == 1 else "VALID",
            use_bias=False,
            kernel_init=jax.nn.initializers.variance_scaling(2.0, "fan_out", "truncated_normal"),
            name=self.name
        )(x)


class STEM(nn.Module):
    bn_momentum: float = 0.0
    bn_epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x, train=True):
        x = Conv2DFixedPadding(32, kernel_size=3, strides=2, name="stem_conv_1")(x)
        x = nn.BatchNorm(
            name="stem_batch_norm_1",
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(32, kernel_size=3, strides=1, name="stem_conv_2")(x)
        x = nn.BatchNorm(
            name="stem_batch_norm_2",
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(64, kernel_size=3, strides=1, name="stem_conv_3")(x)
        x = nn.BatchNorm(
            name="stem_batch_norm_3",
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(64, kernel_size=3, strides=1, name="stem_conv_4")(x)
        x = nn.BatchNorm(
            name="stem_batch_norm_4",
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train
        )(x)
        x = nn.relu(x)
        return x


class BlockGroup(nn.Module):
    filters: int
    strides: int
    num_repeats: int
    counter: int
    se_ratio: float = 0.25
    bn_epsilon: float = 1e-5
    bn_momentum: float = 0.0
    survival_probability: float = 0.8
    name: str = None

    @nn.compact
    def __call__(self, x, train=True):
        if self.name is None:
            self.name = f"block_group_{self.counter}"
        x = BottleneckBlock(
            self.filters,
            strides=self.strides,
            use_projection=True,
            se_ratio=self.se_ratio,
            bn_epsilon=self.bn_epsilon,
            bn_momentum=self.bn_momentum,
            survival_probability=self.survival_probability,
            name=self.name + "_block_0"
        )(x, train)
        for i in range(1, self.num_repeats):
            x = BottleneckBlock(
                self.filters,
                strides=1,
                use_projection=False,
                se_ratio=self.se_ratio,
                bn_epsilon=self.bn_epsilon,
                bn_momentum=self.bn_momentum,
                survival_probability=self.survival_probability,
                name=self.name + f"_block_{i}_"
            )(x, train)
        return x


class BottleneckBlock(nn.Module):
    filters: int
    strides: int
    use_projection: bool
    bn_momentum: float = 0.0
    bn_epsilon: float = 1e-5
    survival_probability: float = 0.8
    se_ratio: float = 0.25
    name: str

    @nn.compact
    def __call__(self, x, train=True):
        shortcut = x
        if self.use_projection:
            filters_out = self.filters * 4
            if self.strides == 2:
                shortcut = nn.avg_pool(x, (2, 2), (2, 2), padding="SAME")
                shortcut = Conv2DFixedPadding(
                    filters_out,
                    kernel_size=1,
                    strides=1,
                    name=f"{self.name}_projection_conv"
                )(shortcut)
            else:
                shortcut = Conv2DFixedPadding(
                    filters_out,
                    kernel_size=1,
                    strides=self.strides,
                    name=f"{self.name}_projection_conv"
                )(shortcut)
            shortcut = nn.BatchNorm(
                axis=3, 
                momentum=self.bn_momentum,
                epsilon=self.bn_epsilon,
                use_running_average=not train,
                name=f"{self.name}_projection_batch_norm"
            )(shortcut)
        x = Conv2DFixedPadding(self.filters, kernel_size=1, strides=1, name=self.name + "_conv_1")(x)
        x = nn.BatchNorm(
            axis=3,
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon, 
            use_running_average=not train,
            name=f"{self.name}_batch_norm_1"
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(
            self.filters, kernel_size=3, strides=self.strides, name=self.name + "_conv_2"
        )(x)
        x = nn.BatchNorm(
            axis=3,
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train,
            name=f"{self.name}_batch_norm_2"
        )(x)
        x = nn.relu(x)
        x = Conv2DFixedPadding(
            self.filters * 4, kernel_size=1, strides=1, name=self.name + "_conv_3"
        )(x)
        x = nn.BatchNorm(
            axis=3,
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            use_running_average=not train,
            name=f"{self.name}_batch_norm_3"
        )(x)
        if 0 < self.se_ratio < 1:
            x = SE(self.filters, se_ratio=self.se_ratio, name=f"{self.name}_se")(x)
        if self.survival_probability:
            x = nn.Dropout(self.survival_probability, deterministic=train, name=f"{self.name}_drop")(x)
        x = x + shortcut
        x = nn.relu(x)
        return x


class SE(nn.Module):
    in_filters: int
    se_ratio: float = 0.25
    expand_ratio: int = 1
    name: str = "se"

    @nn.compact
    def __call__(self, x):
        inputs = x
        x = einops.reduce(x, 'b h w d -> b 1 1 d', 'mean')  # global average pooling
        num_reduced_filters = max(1, int(self.in_filters * 4 * self.se_ratio))
        x = nn.Conv(
            num_reduced_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            kernel_init=jax.nn.initializers.variance_scaling(2.0, "fan_out", "truncated_normal"),
            name=f"{self.name}_se_reduce"
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            4 * self.in_filters * self.expand_ratio,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            kernel_init=jax.nn.initializers.variance_scaling(2.0, "fan_out", "truncated_normal"),
            name=f"{self.name}_se_expand"
        )(x)
        x = nn.sigmoid(x)
        return inputs * x


class ResNetRS(nn.Module):
    classes: int
    block_args: dict
    drop_connect_rate: float = 0.2
    dropout_rate: float = 0.25
    bn_momentum: float = 0.0
    bn_epsilon: float = 1e-5
    se_ratio: float = 0.25

    @nn.compact
    def __call__(self, x, train=True):
        x = STEM(
            bn_momentum=self.bn_momentum, bn_epsilon=self.bn_epsilon, name="STEM_1"
        )(x, train)
        for i, block_arg in enumerate(self.block_args):
            survival_probability = self.drop_connect_rate * float(i + 2) / (len(self.block_args) + 1)
            x = BlockGroup(
                block_arg["input_filters"],
                strides=(1 if i == 0 else 2),
                num_repeats=block_arg["num_repeats"],
                counter=i,
                se_ratio=self.se_ratio,
                bn_momentum=self.bn_momentum,
                bn_epsilon=self.bn_epsilon,
                survival_probability=survival_probability,
                name=f"BlockGroup{i + 2}"
            )(x, train) 
        x = einops.reduce(x, 'b h w d -> b d', 'mean')  # global average pooling
        x = nn.Dropout(self.dropout_rate, deterministic=train, name="top_dropout")(x)
        x = nn.Dense(1000, name="predictions")(x)
        x = nn.softmax(x)
        return x


class ResNetRS50(nn.Module):
    classes: int = 1000
    pw: float = 1.0
    pd: float = 1.0
    scale: float = 1.0

    @nn.compact
    def __call__(self, x, train=True):
        x = jax.nn.standardize(x, mean=np.array([0.485, 0.456, 0.406]), variance=np.array([0.229**2, 0.224**2, 0.225**2]))
        x = ResNetRS(
            self.classes,
            [
                {
                    "input_filters": 64,
                    "num_repeats": 3
                },
                {
                    "input_filters": 128,
                    "num_repeats": 4
                },
                {
                    "input_filters": 256,
                    "num_repeats": 6
                },
                {
                    "input_filters": 512,
                    "num_repeats": 3
                },
            ],
            drop_connect_rate=0.0,
            dropout_rate=0.25,
        )(x, train)
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        return x


# import math
# import jaxopt
# import optax
# import datasets
# import numpy as np
# from flax import serialization
# from tqdm import trange
# import sklearn.metrics as skm
# import common


# def cifar10():
#     ds = datasets.load_dataset("cifar10")
#     ds = ds.map(
#         lambda e: {
#             'X': np.array(e['img'], dtype=np.float32) / 255,
#             # 'X': np.array(e['img'], dtype=np.float32),
#             'Y': e['label'],
#         },
#         remove_columns=['img', 'label']
#     )
#     features = ds['train'].features
#     features['X'] = datasets.Array3D(shape=(32, 32, 3), dtype='float32')
#     ds['train'] = ds['train'].cast(features)
#     ds['test'] = ds['test'].cast(features)
#     ds.set_format('numpy')
#     return {t: {'X': ds[t]['X'], 'Y': ds[t]['Y']} for t in ['train', 'test']}


# def crossentropy_loss(model):
#     def _apply(params, X, Y):
#         logits, batch_stats = model.apply(params, X, mutable=["batch_stats"])
#         logits = jnp.clip(logits, 1e-15, 1 - 1e-15)
#         one_hot = jax.nn.one_hot(Y, logits.shape[-1])
#         return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits))), batch_stats
#     return _apply

# def predictor(model):
#     @jax.jit
#     def _apply(variables, X):
#         # return jnp.argmax(model.apply(variables, X, rngs={"dropout": jax.random.PRNGKey(42)}, train=False), axis=-1)
#         return jnp.argmax(model.apply(variables, X), axis=-1)
#     return _apply


# if __name__ == "__main__":
#     dataset = cifar10()
#     model = CNN(10)
#     opt = optax.sgd(0.01, momentum=0.9)
#     variables = model.init(jax.random.PRNGKey(42), dataset['train']['X'][:1])

#     # # RESNETRS50 START
#     # model = ResNetRS50(10)
#     # with open("ResNetRS50.variables", 'rb') as f:
#     #     variables = serialization.msgpack_restore(f.read())
#     # init_variables = model.init(jax.random.PRNGKey(42), dataset['train']['X'][:1])
#     # variables['params']['classifier'] = init_variables['params']['classifier']
#     # model_key = list(variables['params'].keys())[0]
#     # opt = optax.masked(optax.adam(0.01), common.map_nested_fn(variables, lambda k, v: k == "classifier"))
#     # # RESNETRS50 END

#     solver = jaxopt.OptaxSolver(common.crossentropy_loss(model), opt)
#     state = solver.init_state(variables, X=dataset['train']['X'][:1], Y=dataset['train']['Y'][:1])
#     solver_step = jax.jit(solver.update)
#     batch_size = 32
#     for _ in (pbar := trange(3000)):
#         idx = np.random.choice(np.arange(len(dataset['train']['Y'])), batch_size, replace=False)
#         variables, state = solver_step(variables, state, X=dataset['train']['X'][idx], Y=dataset['train']['Y'][idx])
#         pbar.set_postfix_str(f"Loss: {state.value:.5f}")
#     idxs = np.array_split(np.arange(len(dataset['test']['Y'])), math.ceil(len(dataset['test']['Y']) / batch_size))
#     predict = predictor(model)
#     preds = np.concatenate([predict(variables, dataset['test']['X'][idx]) for idx in idxs])
#     print(f"Accuracy: {skm.accuracy_score(dataset['test']['Y'], preds):.3%}")

#     # import tensorflow as tf
#     # base_model = tf.keras.applications.ResNetRS50(input_shape=(32, 32, 3), weights="imagenet")
#     # base_model.trainable = False
#     # model = tf.keras.Sequential([
#     #     base_model,
#     #     tf.keras.layers.Dense(10, activation="softmax")
#     # ])
#     # model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#     # model.fit(x=dataset['train']['X'], y=dataset['train']['Y'], batch_size=32, epochs=3)
#     # model.evaluate(x=dataset['test']['X'], y=dataset['test']['Y'])