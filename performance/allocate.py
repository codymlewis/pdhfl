"""
A simulator of the allocate function for finding the optimal model partition
that all clients in a device heterogeneous synchronous federated learning setting
should take.
"""

from enum import Enum
import json
import math

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import einops
from tqdm import trange


class Device(Enum):
    BIG = 0
    LITTLE = 1
    LAPTOP = 2


def utility(lamb, R):
    """
    Utility of model partition function

    Arguments:
    - lamb: Weighting of the partition itself
    - t: Function that states the importance of computation time w.r.t. the partition
    """
    @jax.jit
    def _apply(p):
        return jnp.sum(p)**lamb * R(p)**(1 - lamb)
    return _apply


def round_limiter(time, T):
    """
    Function that states the importance of computation time w.r.t. the partition

    Arguments:
    - T: Global time limit
    - i: Index of the client, to simulate differing computational capability
    """
    @jax.jit
    def _apply(p):
        return jsp.stats.norm.pdf(time(p), loc=T, scale=0.5)
    return _apply


@jax.jit
def pdhfl_time(p):
    return p[1] * 0.14 * (1 + 0.2 * p[0])


@jax.jit
def fjord_time(p):
    return 0.14 * (1 + 0.2 * p[0])


@jax.jit
def heterofl_time(p):
    return p[1] * 0.14 * (1 + 0.2 * p[1])


@jax.jit
def feddrop_time(p):
    return 0.14 * (1 + 0.2 * p[0])


def conv_time(p):
    return 0.04 + 0.08 * p[0]


def bigcom_time(base_time):
    @jax.jit
    def _apply(p):
        return base_time(p)
    return _apply


def littlecom_time(base_time):
    @jax.jit
    def _apply(p):
        return 2 * base_time(p)
    return _apply


def laptop_time(base_time):
    @jax.jit
    def _apply(p):
        return 16 * base_time(p)**2
    return _apply


class Client:
    "A client in the FL network, will attempt to find the optimal partition"

    def __init__(self, i, lamb, lrs, T, algorithm):
        self.T = T
        self.p = jnp.array([0.01, 0.01 if algorithm != "fjord" else 1.0])
        # self.p = jnp.array([np.random.uniform(), np.random.uniform()])
        self.time = {
            Device.BIG.value: bigcom_time,
            Device.LITTLE.value: littlecom_time,
            Device.LAPTOP.value: laptop_time
        }[i]({"pdhfl": pdhfl_time, "fjord": fjord_time, "heterofl": heterofl_time}[algorithm])
        self.u = utility(lamb, round_limiter(self.time, T))
        self.lrs = lrs
        self.t = 0
        self.algorithm = algorithm

    def step(self):
        "Take a step of gradient descent upon the parition utility function"
        util, grad = jax.value_and_grad(self.u)(self.p)
        new_p = jnp.clip(self.p + self.lrs.learning_rate * grad, 0.1, 1)
        if self.time(new_p) <= self.T:
            if self.algorithm != "heterofl":
                self.p = new_p
            else:
                self.p = jnp.array([new_p[1], new_p[1]])
        return util


class Server:
    "A server that co-ordinates the FL process"

    def __init__(self, nclients, lamb, lrs, T, algorithm="pdhfl"):
        self.clients = [Client(i % 3, lamb, lrs(), T, algorithm) for i in range(nclients)]

    def step(self):
        "Get all clients to perform a step of utility optimization and return the mean utility"
        utils = jnp.array([c.step() for c in self.clients])
        return einops.reduce(utils, 'c ->', 'mean')


class LearningRateSchedule:
    "Class for handling the learning rate, after a certain number of rounds it decays to a smaller rate for fine tuning"

    def __init__(self, lr0, decay_time):
        self.lr = lr0
        self.decay_time = decay_time
        self.time = 0

    @property
    def learning_rate(self):
        if self.decay_time == self.time:
            self.lr *= 0.1
        self.time += 1
        return self.lr


def round_down(ps):
    "round down p to a single decimal place"
    return [math.floor(p * 10) / 10 for p in ps.tolist()]


if __name__ == "__main__":
    allocations = {}
    # p = [p_w, p_d]
    T = 1/3
    for algorithm in ["pdhfl", "fjord", "heterofl"]:
        print(f"Optimizing allocations for {algorithm}")
        epochs = 800
        server = Server(
            nclients=3, lamb=0.1, lrs=lambda: LearningRateSchedule(0.1, int(epochs/2)), T=T, algorithm=algorithm
        )
        for _ in (pbar := trange(epochs)):
            utility_val = server.step()
            pbar.set_postfix_str(f"UTIL: {utility_val:.5f}")
        print(
            "Final allocation: {}".format(
                [f'p={round_down(c.p)}, t_i(p)={c.time(round_down(c.p))}' for c in server.clients]
            )
        )
        allocations[algorithm] = [[round_down(c.p)[i] for c in server.clients] for i in range(2)]

    print("Calculating allocation for FedDrop")
    fcn_times = jnp.array([
        bigcom_time(feddrop_time)(jnp.array([1.0, 1.0])),
        littlecom_time(feddrop_time)(jnp.array([1.0, 1.0])),
        laptop_time(feddrop_time)(jnp.array([1.0, 1.0])),
    ])
    conv_times = jnp.array([
        bigcom_time(conv_time)(jnp.array([1.0, 1.0])),
        littlecom_time(conv_time)(jnp.array([1.0, 1.0])),
        laptop_time(conv_time)(jnp.array([1.0, 1.0])),
    ])
    pw = round_down(jnp.minimum(1.0, jnp.sqrt((T - conv_times) / fcn_times)))
    print(f"Found allocation p={pw}")
    allocations['feddrop'] = [pw, [1.0, 1.0, 1.0]]

    with open("allocations.json", "w") as f:
        json.dump(allocations, f)
    print("Written allocations to allocations.json")
