import jax
import jax.numpy as jnp
import numpy as np

from . import model
from . import common


class Client:
    def __init__(self, model, data, batch_size, epochs, steps_per_epoch=None):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

    def step(self, parameters):
        client_parameters = self.model.sample_parameters()
        client_parameters['params'] = common.partition(client_parameters['params'], parameters)
        loss, grads = self.model.fit(
            client_parameters,
            self.data['train']['X'],
            self.data['train']['Y'],
            batch_size=self.batch_size,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            return_grads = True,
        )
        return loss, common.expand(grads, parameters), len(self.data['train']['Y'])

    def analytics(self, parameters):
        client_parameters = self.model.sample_parameters()
        client_parameters['params'] = common.partition(client_parameters['params'], parameters)
        return self.model.evaluate(
            client_parameters,
            self.data['test']['X'],
            self.data['test']['Y'],
            batch_size=self.batch_size
        )


class FedDrop(Client):
    def __init__(self, model, data, batch_size, epochs, p, steps_per_epoch=None, seed=42):
        super().__init__(model, data, batch_size, epochs, steps_per_epoch=steps_per_epoch)
        self.p = p
        self.rng = np.random.default_rng(seed)

    def step(self, parameters):
        parameters = feddrop(parameters, self.p, self.rng)
        return super().step(parameters)


def feddrop(params, pmin, rng):
    return {
        k: {
            dk: (rng.uniform(size=dp.shape[1]) < pmin) * dp if dk == "kernel" and "dense" in k.lower() else dp for dk, dp in d.items()
        }
        for k, d in params.items()
    }

    
class Local(Client):
    def step(self, parameters):
        loss, parameters = self.model.fit(
            parameters,
            self.data['train']['X'],
            self.data['train']['Y'],
            batch_size=self.batch_size,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            return_grads=False
        )
        return loss, parameters

    def analytics(self, parameters):
        return self.model.evaluate(
            parameters,
            self.data['test']['X'],
            self.data['test']['Y'],
            batch_size=self.batch_size,
        )