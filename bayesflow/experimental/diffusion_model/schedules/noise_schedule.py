from abc import ABC, abstractmethod
from typing import Literal

from keras import ops

from bayesflow.types import Tensor
from bayesflow.utils.serialization import deserialize, serializable


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
class NoiseSchedule(ABC):
    r"""Noise schedule for diffusion models. We follow the notation from [1].

    The diffusion process is defined by a noise schedule, which determines how the noise level changes over time.
    We define the noise schedule as a function of the log signal-to-noise ratio (lambda), which can be
    interchangeably used with the diffusion time (t).

    The noise process is defined as: z = alpha(t) * x + sigma(t) * e, where e ~ N(0, I).
    The schedule is defined as: \lambda(t) = \log \sigma^2(t) - \log \alpha^2(t).

    We can also define a weighting function for each noise level for the loss function. Often the noise schedule is
    the same for the forward and reverse process, but this is not necessary and can be changed via the training flag.

    [1] Variational Diffusion Models 2.0: Understanding Diffusion Model Objectives as the ELBO with Simple Data
    Augmentation: Kingma et al. (2023)
    """

    def __init__(
        self,
        name: str,
        variance_type: Literal["preserving", "exploding"],
        weighting: Literal["sigmoid", "likelihood_weighting"] = None,
    ):
        """
        Initialize the noise schedule with given variance and weighting strategy.

        Parameters
        ----------
        name : str
            The name of the noise schedule.
        variance_type : Literal["preserving", "exploding"]
            If the variance of noise added to the data should be preserved over time, use "preserving".
            If the variance of noise added to the data should increase over time, use "exploding".
            Default is "preserving".
        weighting : Literal["sigmoid", "likelihood_weighting"], optional
            The type of weighting function to use for the noise schedule.
            Default is None, which means no weighting is applied.
        """
        self.name = name
        self._variance_type = variance_type
        self.log_snr_min = None  # should be set in the subclasses
        self.log_snr_max = None  # should be set in the subclasses
        self._weighting = weighting

    @abstractmethod
    def get_log_snr(self, t: float | Tensor, training: bool) -> Tensor:
        """Get the log signal-to-noise ratio (lambda) for a given diffusion time."""
        pass

    @abstractmethod
    def get_t_from_log_snr(self, log_snr_t: float | Tensor, training: bool) -> Tensor:
        """Get the diffusion time (t) from the log signal-to-noise ratio (lambda)."""
        pass

    @abstractmethod
    def derivative_log_snr(self, log_snr_t: float | Tensor, training: bool) -> Tensor:
        r"""Compute \beta(t) = d/dt log(1 + e^(-snr(t))). This is usually used for the reverse SDE."""
        pass

    def get_drift_diffusion(
        self, log_snr_t: Tensor, x: Tensor = None, training: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""Compute the drift and optionally the squared diffusion term for the reverse SDE.
        It can be derived from the derivative of the schedule:

        math::
            \beta(t) = d/dt \log(1 + e^{-snr(t)})

            f(z, t) = -0.5 * \beta(t) * z

            g(t)^2 = \beta(t)

        The corresponding differential equations are::

            SDE: d(z) = [ f(z, t) - g(t)^2 * score(z, lambda) ] dt + g(t) dW
            ODE: dz = [ f(z, t) - 0.5 * g(t)^2 * score(z, lambda) ] dt

        For a variance exploding schedule, one should set f(z, t) = 0.
        """
        beta = self.derivative_log_snr(log_snr_t=log_snr_t, training=training)
        if x is None:  # return g^2 only
            return beta
        if self._variance_type == "preserving":
            f = -0.5 * beta * x
        elif self._variance_type == "exploding":
            f = ops.zeros_like(beta)
        else:
            raise ValueError(f"Unknown variance type: {self._variance_type}")
        return f, beta

    def get_alpha_sigma(self, log_snr_t: Tensor) -> tuple[Tensor, Tensor]:
        """Get alpha and sigma for a given log signal-to-noise ratio (lambda).

        Default is a variance preserving schedule:

            alpha(t) = sqrt(sigmoid(log_snr_t))
            sigma(t) = sqrt(sigmoid(-log_snr_t))

        For a variance exploding schedule, one should set alpha^2 = 1 and sigma^2 = exp(-lambda)
        """
        if self._variance_type == "preserving":
            # variance preserving schedule
            alpha_t = ops.sqrt(ops.sigmoid(log_snr_t))
            sigma_t = ops.sqrt(ops.sigmoid(-log_snr_t))
        elif self._variance_type == "exploding":
            # variance exploding schedule
            alpha_t = ops.ones_like(log_snr_t)
            sigma_t = ops.sqrt(ops.exp(-log_snr_t))
        else:
            raise TypeError(f"Unknown variance type: {self._variance_type}")
        return alpha_t, sigma_t

    def get_weights_for_snr(self, log_snr_t: Tensor) -> Tensor:
        """
        Compute loss weights based on log signal-to-noise ratio (log-SNR).

        This method returns a tensor of weights used for loss re-weighting in diffusion models,
        depending on the selected strategy. If no weighting is specified, uniform weights (ones)
        are returned.

        Supported weighting strategies:
        - "sigmoid": Based on Kingma et al. (2023), uses a sigmoid of shifted log-SNR.
        - "likelihood_weighting": Based on Song et al. (2021), uses ratio of diffusion drift
          to squared noise scale.

        Parameters
        ----------
        log_snr_t : Tensor
            A tensor containing the log signal-to-noise ratio values.

        Returns
        -------
        Tensor
            A tensor of weights corresponding to each log-SNR value.

        Raises
        ------
        TypeError
            If the weighting strategy specified in `self._weighting` is unknown.
        """
        if self._weighting is None:
            return ops.ones_like(log_snr_t)
        elif self._weighting == "sigmoid":
            # sigmoid weighting based on Kingma et al. (2023)
            return ops.sigmoid(-log_snr_t + 2)
        elif self._weighting == "likelihood_weighting":
            # likelihood weighting based on Song et al. (2021)
            g_squared = self.get_drift_diffusion(log_snr_t)
            _, sigma_t = self.get_alpha_sigma(log_snr_t)
            return g_squared / ops.square(sigma_t)
        else:
            raise TypeError(f"Unknown weighting type: {self._weighting}")

    def get_config(self):
        return {"name": self.name, "variance_type": self._variance_type, "weighting": self._weighting}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def validate(self):
        """Validate the noise schedule."""

        if self.log_snr_min >= self.log_snr_max:
            raise ValueError("min_log_snr must be less than max_log_snr.")

        # Validate log SNR values and corresponding time mappings for both training and inference
        for training in (True, False):
            if not ops.isfinite(self.get_log_snr(0.0, training=training)):
                raise ValueError(f"log_snr(0.0) must be finite (training={training})")
            if not ops.isfinite(self.get_log_snr(1.0, training=training)):
                raise ValueError(f"log_snr(1.0) must be finite (training={training})")
            if not ops.isfinite(self.get_t_from_log_snr(self.log_snr_max, training=training)):
                raise ValueError(f"t(log_snr_max) must be finite (training={training})")
            if not ops.isfinite(self.get_t_from_log_snr(self.log_snr_min, training=training)):
                raise ValueError(f"t(log_snr_min) must be finite (training={training})")

        # Validate log SNR derivatives at the boundaries
        for boundary, name in [(self.log_snr_max, "log_snr_max (t=0)"), (self.log_snr_min, "log_snr_min (t=1)")]:
            if not ops.isfinite(self.derivative_log_snr(boundary, training=False)):
                raise ValueError(f"derivative_log_snr at {name} must be finite.")
