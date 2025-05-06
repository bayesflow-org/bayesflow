import keras

from bayesflow.utils import model_kwargs, find_network
from bayesflow.utils.serialization import deserialize, serializable, serialize
from bayesflow.types import Shape, Tensor
from bayesflow.scores import ScoringRule, ParametricDistributionScore
from bayesflow.utils.decorators import allow_batch_size


@serializable("bayesflow.networks")
class PointInferenceNetwork(keras.Layer):
    """Implements point estimation for user specified scoring rules by a shared feed forward architecture
    with separate heads for each scoring rule.
    """

    def __init__(
        self,
        scores: dict[str, ScoringRule],
        subnet: str | keras.Layer = "mlp",
        **kwargs,
    ):
        super().__init__(**model_kwargs(kwargs))

        self.scores = scores

        self.subnet = find_network(subnet, **kwargs.get("subnet_kwargs", {}))

        self.config = {
            "subnet": serialize(subnet),
            "scores": serialize(scores),
            **kwargs,
        }

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        """Builds all network components based on shapes of conditions and targets.

        For each score, corresponding estimation heads are constructed.
        There are two steps in this:

        #. Request a dictionary of names and output shapes of required heads from the score.
        #. Then for each required head, request corresponding head networks from the score.

        Since the score is in charge of constructing heads, this allows for convenient yet flexible building.
        """
        if conditions_shape is None:  # unconditional estimation uses a fixed input vector
            input_shape = (1, 1)
        else:
            input_shape = conditions_shape

        # Save input_shape and xz_shape for usage in get_build_config
        self._input_shape = input_shape
        self._xz_shape = xz_shape

        # build the shared body network
        self.subnet.build(input_shape)
        body_output_shape = self.subnet.compute_output_shape(input_shape)

        # build head(s) for every scoring rule
        self.heads = dict()
        self.heads_flat = dict()  # see comment regarding heads_flat below

        for score_key, score in self.scores.items():
            head_shapes = score.get_head_shapes_from_target_shape(xz_shape)

            self.heads[score_key] = {}

            for head_key, head_shape in head_shapes.items():
                head = score.get_head(head_key, head_shape)
                head.build(body_output_shape)
                # If head is not tracked explicitly, self.variables does not include them.
                # Testing with tests.utils.assert_layers_equal() would thus neglect heads
                head = self._tracker.track(head)  # explicitly track head

                self.heads[score_key][head_key] = head

                # Until keras issue [20598](https://github.com/keras-team/keras/issues/20598)
                # is resolved, a flat version of the heads dictionary is kept.
                # This allows to save head weights properly, see for reference
                # https://github.com/keras-team/keras/blob/v3.3.3/keras/src/saving/saving_lib.py#L481.
                # A nested heads dict is still preferred over this flat dict,
                # because it avoids string operation based filtering in `self._forward()`.
                flat_key = f"{score_key}___{head_key}"
                self.heads_flat[flat_key] = head

    def get_build_config(self):
        build_config = {
            "conditions_shape": self._input_shape,
            "xz_shape": self._xz_shape,
        }

        # Save names of head networks
        heads = {}
        for score_key in self.heads.keys():
            heads[score_key] = {}
            for head_key, head in self.heads[score_key].items():
                heads[score_key][head_key] = head.name

        build_config["heads"] = heads

        return build_config

    def build_from_config(self, config):
        self.build(xz_shape=config["xz_shape"], conditions_shape=config["conditions_shape"])

        for score_key in self.scores.keys():
            for head_key, head in self.heads[score_key].items():
                head.name = config["heads"][score_key][head_key]

    def get_config(self):
        base_config = super().get_config()

        return base_config | self.config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["scores"] = deserialize(config["scores"])
        config["subnet"] = deserialize(config["subnet"])
        return cls(**config)

    def call(
        self,
        xz: Tensor = None,
        conditions: Tensor = None,
        training: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, Tensor]]:
        if xz is None and not self.built:
            raise ValueError("Cannot build inference network without inference variables.")
        if conditions is None:  # unconditional estimation uses a fixed input vector
            conditions = keras.ops.convert_to_tensor(
                [[1.0]], dtype=keras.ops.dtype(xz) if xz is not None else "float32"
            )

        # pass conditions to the shared subnet
        output = self.subnet(conditions, training=training)

        # pass along to calculate individual head outputs
        output = {
            score_key: {head_key: head(output, training=training) for head_key, head in self.heads[score_key].items()}
            for score_key in self.heads.keys()
        }
        return output

    def compute_metrics(
        self, x: Tensor, conditions: Tensor = None, sample_weight: Tensor = None, stage: str = "training"
    ) -> dict[str, Tensor]:
        output = self(x, conditions)

        metrics = {}
        # calculate negative score as mean over all scores
        for score_key, score in self.scores.items():
            score_value = score.score(output[score_key], x, sample_weight)
            metrics[score_key] = score_value
        neg_score = keras.ops.mean(list(metrics.values()))

        if stage != "training" and any(self.metrics):
            # compute sample-based metrics
            samples = self.sample((keras.ops.shape(x)[0],), conditions=conditions)

            for metric in self.metrics:
                metrics[metric.name] = metric(samples, x)

        return metrics | {"loss": neg_score}

    @allow_batch_size
    def sample(self, batch_shape: Shape, conditions: Tensor = None) -> dict[str, Tensor]:
        """
        Parameters
        ----------
        batch_shape : tuple,
            Expected dimensions depend on `conditions`
            - conditional sampling: (batch_size, num_samples) if `conditions` is a tensor
              of shape (batch_size, num_samples)
            - unconditional sampling: (num_samples,) if `conditions` is None
        conditions : Tensor or None, default None
            Optional inference conditions. If `conditions` is not given, the method will return unconditional samples.

        Returns
        -------
        samples : dict[str, Tensor]
            Samples for every parametric scoring rule. Dict values have shape (batch_size, num_samples, num_variables)
            or (num_samples, num_variables) for conditional or unconditional sampling respectively.
        """
        if conditions is None:  # unconditional estimation uses a fixed input vector
            conditions = keras.ops.ones(batch_shape, dtype="float32").reshape(1, -1, 1)

        # conditions are duplicated along axis 1 num_sample times
        output = self.subnet(conditions[:, 0, :])
        samples = {}

        for score_key, score in self.scores.items():
            if isinstance(score, ParametricDistributionScore):
                parameters = {head_key: head(output) for head_key, head in self.heads[score_key].items()}
                samples[score_key] = score.sample(batch_shape, **parameters)

        return samples

    def log_prob(self, samples: Tensor, conditions: Tensor = None, **kwargs) -> dict[str, Tensor]:
        output = self.subnet(conditions)
        log_probs = {}

        for score_key, score in self.scores.items():
            if isinstance(score, ParametricDistributionScore):
                parameters = {head_key: head(output) for head_key, head in self.heads[score_key].items()}
                log_probs[score_key] = score.log_prob(x=samples, **parameters)

        return log_probs
