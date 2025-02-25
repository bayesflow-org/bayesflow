import keras
from keras.saving import (
    deserialize_keras_object as deserialize,
    serialize_keras_object as serialize,
    register_keras_serializable as serializable,
)

from bayesflow.utils import keras_kwargs, find_network, serialize_value_or_type, deserialize_value_or_type
from bayesflow.types import Shape, Tensor
from bayesflow.scores import ScoringRule, ParametricDistributionRule
from bayesflow.utils.decorators import allow_batch_size


@serializable(package="networks.point_inference_network")
class PointInferenceNetwork(keras.Layer):
    """Implements point estimation for user specified scoring rules by a shared feed forward architecture
    with separate heads for each scoring rule.
    """

    def __init__(
        self,
        scores: dict[str, ScoringRule],
        subnet: str | type = "mlp",
        **kwargs,
    ):
        super().__init__(
            **keras_kwargs(kwargs)
        )  # TODO: need for bf.utils.keras_kwargs in regular InferenceNetwork class? seems to be a bug

        self.scores = scores

        self.subnet = find_network(subnet, **kwargs.get("subnet_kwargs", {}))

        self.config = {
            **kwargs,
        }
        self.config = serialize_value_or_type(self.config, "subnet", subnet)
        self.config["scores"] = serialize(self.scores)

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        if conditions_shape is None:  # unconditional estimation uses a fixed input vector
            input_shape = (1, 1)
        else:
            input_shape = conditions_shape

        # build the shared body network
        self.subnet.build(input_shape)
        body_output_shape = self.subnet.compute_output_shape(input_shape)

        # build head(s) for every scoring rule
        self.heads = dict()
        self.heads_flat = dict()  # see comment regarding heads_flat below

        for score_key, score in self.scores.items():
            score.set_head_shapes_from_target_shape(xz_shape)

            self.heads[score_key] = {}

            for head_key in score.head_shapes.keys():
                head = score.get_head(head_key)
                head.build(body_output_shape)
                # If head is not tracked explicitly, self.variables does not include them.
                # Testing with tests.utils.assert_layers_equal() would thus neglect heads
                head = self._tracker.track(head)  # explicitly track head

                self.heads[score_key][head_key] = head

                # Until keras issue [20598](https://github.com/keras-team/keras/issues/20598)
                # is resolved, a flat version of the heads dictionary is kept.
                # This allows to save head weights properly, see for reference
                # https://github.com/keras-team/keras/blob/v3.3.3/keras/src/saving/saving_lib.py#L481.
                # A nested heads dict is still prefered over this flat dict,
                # because it avoids string operation based filtering in `self._forward()`.
                flat_key = f"{score_key}___{head_key}"
                self.heads_flat[flat_key] = head

    def get_config(self):
        base_config = super().get_config()

        return base_config | self.config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["scores"] = deserialize(config["scores"])
        config = deserialize_value_or_type(config, "subnet")
        return cls(**config)

    def call(
        self,
        xz: Tensor = None,
        conditions: Tensor = None,
        training: bool = False,
        **kwargs,
    ) -> dict[str, Tensor]:
        if xz is None and not self.built:
            raise ValueError("Cannot build inference network without inference variables.")
        if conditions is None:  # unconditional estimation uses a fixed input vector
            conditions = keras.ops.convert_to_tensor([[1.0]], dtype="float32")
        return self._forward(conditions=conditions, training=training, **kwargs)

    def _forward(
        self,
        conditions: Tensor = None,
        training: bool = False,
        **kwargs,
    ) -> dict[str, Tensor]:
        output = self.subnet(conditions, training=training)

        output = {
            score_key: {head_key: head(output, training=training) for head_key, head in self.heads[score_key].items()}
            for score_key in self.heads.keys()
        }
        return output

    def compute_metrics(self, x: Tensor, conditions: Tensor = None, stage: str = "training") -> dict[str, Tensor]:
        output = self(x, conditions)

        metrics = {}
        # calculate negative score as mean over all scores
        neg_score = 0
        for score_key, score in self.scores.items():
            score_value = score.score(output[score_key], x)
            neg_score += score_value
            metrics |= {score_key: score_value}
        neg_score /= len(self.scores)

        if stage != "training" and any(self.metrics):
            # compute sample-based metrics
            samples = self.sample((keras.ops.shape(x)[0],), conditions=conditions)

            for metric in self.metrics:
                metrics[metric.name] = metric(samples, x)

        return metrics | {"loss": neg_score}

    # WIP: untested draft of sample method
    @allow_batch_size
    def sample(self, batch_shape: Shape, conditions: Tensor = None, **kwargs) -> dict[str, Tensor]:
        output = self.subnet(conditions)
        samples = {}

        for score_key, score in self.scores.items():
            if isinstance(score, ParametricDistributionRule):
                parameters = {head_key: head(output) for head_key, head in self.heads[score_key].items()}
                samples[score_key] = score.sample(batch_shape, **parameters)

        return samples

    # WIP: untested draft of log_prob method
    def log_prob(self, samples: Tensor, conditions: Tensor = None, **kwargs) -> dict[str, Tensor]:
        output = self.subnet(conditions)
        log_probs = {}

        for score_key, score in self.scores.items():
            if isinstance(score, ParametricDistributionRule):
                parameters = {head_key: head(output) for head_key, head in self.heads[score_key].items()}
                log_probs[score_key] = score.log_prob(x=samples, **parameters)

        return log_probs
