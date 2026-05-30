import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .categorical_scoring_rule import CategoricalScoringRule


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class CrossEntropyScore(CategoricalScoringRule):
    r""":math:`S(\hat p_{1\ldots C}, y) = -\sum_{c=1}^C y_c \log \hat p_c`

    Designed for model comparison / classification, where the network outputs
    raw logits and targets are one-hot encoded class indicators.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = {}

    def get_head_shapes_from_target_shape(self, target_shape: Shape):
        # keras.saving.load_model sometimes passes target_shape as a list, so we force a conversion
        target_shape = tuple(target_shape)
        return dict(logits=target_shape[1:])

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes categorical cross-entropy from logits or probs.

        Parameters
        ----------
        estimates : dict[str, Tensor]
            A dictionary containing tensors of estimated values. The ``"logits"`` key must be present,
            containing raw (unnormalized) class scores of shape ``(..., num_classes)``.
        targets : Tensor
            One-hot encoded target labels of shape ``(..., num_classes)``.
        weights : Tensor, optional
            Per-sample weights. If provided, computes a weighted mean score.

        Returns
        -------
        Tensor
            The (optionally weighted) mean categorical cross-entropy.
        """

        scores = keras.losses.categorical_crossentropy(
            targets,
            estimates["logits" if "logits" in estimates else "probs"],
            from_logits=("logits" in estimates),
        )
        score = weighted_mean(scores, weights)
        return score

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config
