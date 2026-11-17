import keras

from bayesflow.utils.serialization import serializable


@serializable("bayesflow.links")
class Leaky(keras.Layer):
    r"""Leaky parity-odd power (l-POP) transform :math:`J_\lambda(x) = x(1 + |x|^{\lambda - 1})`.

    Expands large magnitudes super-linearly while remaining odd and smooth,
    improving numerical recovery of extreme log Bayes factors without affecting
    properness of the scoring rule.

    Parameters
    ----------
    power : float
        Exponent :math:`\lambda`. Default: 2.0.
    """

    def __init__(self, power: float = 2.0, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.power = power
        self.eps = eps

    def call(self, x):
        return x + x * keras.ops.power(keras.ops.abs(x) + self.eps, self.power - 1.0)

    def get_config(self):
        return super().get_config() | {"power": self.power, "eps": self.eps}
