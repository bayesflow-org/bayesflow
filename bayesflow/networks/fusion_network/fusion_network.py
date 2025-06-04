from collections.abc import Mapping
from ..summary_network import SummaryNetwork
from bayesflow.utils.serialization import deserialize, serializable, serialize
from bayesflow.types import Tensor, Shape
import keras
from keras import ops


@serializable("bayesflow.networks")
class FusionNetwork(SummaryNetwork):
    def __init__(
        self,
        backbones: Mapping[str, keras.Layer],
        head: keras.Layer | None = None,
        **kwargs,
    ):
        """(SN) Wraps multiple summary networks (`backbones`) to learn summary statistics from multi-modal data.

        Networks and inputs are passed as dictionaries with corresponding keys, so that each input is processed
        by the correct summary network. This means the "summary_variables" entry to the approximator has to be
        a dictionary, which can be achieved using the :py:meth:`bayesflow.adapters.Adapter.group` method.

        This network implements _late_ fusion. The output of the individual summary networks is concatenated, and
        can be further processed by another neural network (`head`).

        Parameters
        ----------
        backbones : dict
            A dictionary with names of inputs as keys and corresponding summary networks as values.
        head : keras.Layer, optional
            A network to further process the concatenated outputs of the summary networks. By default,
            the concatenated outputs are returned without further processing.
        **kwargs
            Additional keyword arguments that are passed to the :py:class:`~bayesflow.networks.SummaryNetwork`
            base class.
        """
        super().__init__(**kwargs)
        self.backbones = backbones
        self.head = head
        self._ordered_keys = sorted(list(self.backbones.keys()))

    def build(self, inputs_shape: Mapping[str, Shape]):
        if self.built:
            return
        output_shapes = []
        for k, shape in inputs_shape.items():
            if not self.backbones[k].built:
                self.backbones[k].build(shape)
            output_shapes.append(self.backbones[k].compute_output_shape(shape))
        if self.head and not self.head.built:
            fusion_input_shape = (*output_shapes[0][:-1], sum(shape[-1] for shape in output_shapes))
            self.head.build(fusion_input_shape)
        self.built = True

    def compute_output_shape(self, inputs_shape: Mapping[str, Shape]):
        output_shapes = []
        for k, shape in inputs_shape.items():
            output_shapes.append(self.backbones[k].compute_output_shape(shape))
        output_shape = (*output_shapes[0][:-1], sum(shape[-1] for shape in output_shapes))
        if self.head:
            output_shape = self.head.compute_output_shape(output_shape)
        return output_shape

    def call(self, inputs: Mapping[str, Tensor], training=False):
        """
        Parameters
        ----------
        inputs : dict[str, Tensor]
            Each value in the dictionary is the input to the summary network with the corresponding key.
        training : bool, optional
            Whether the model is in training mode, affecting layers like dropout and
            batch normalization. Default is False.
        """
        outputs = [self.backbones[k](inputs[k], training=training) for k in self._ordered_keys]
        outputs = ops.concatenate(outputs, axis=-1)
        if self.head is None:
            return outputs
        return self.head(outputs, training=training)

    def compute_metrics(self, inputs: Mapping[str, Tensor], stage: str = "training", **kwargs) -> dict[str, Tensor]:
        """
        Parameters
        ----------
        inputs : dict[str, Tensor]
            Each value in the dictionary is the input to the summary network with the corresponding key.
        stage : bool, optional
            Whether the model is in training mode, affecting layers like dropout and
            batch normalization. Default is False.
        **kwargs
            Additional keyword arguments.
        """
        if not self.built:
            self.build(keras.tree.map_structure(keras.ops.shape, inputs))
        metrics = {"loss": [], "outputs": []}

        for k in self._ordered_keys:
            if isinstance(self.backbones[k], SummaryNetwork):
                metrics_k = self.backbones[k].compute_metrics(inputs[k], stage=stage, **kwargs)
                metrics["outputs"].append(metrics_k["outputs"])
                if "loss" in metrics_k:
                    metrics["loss"].append(metrics_k["loss"])
            else:
                metrics["outputs"].append(self.backbones[k](inputs[k], training=stage == "training"))
        if len(metrics["loss"]) == 0:
            del metrics["loss"]
        else:
            metrics["loss"] = ops.sum(metrics["loss"])
        metrics["outputs"] = ops.concatenate(metrics["outputs"], axis=-1)
        if self.head is not None:
            metrics["outputs"] = self.head(metrics["outputs"], training=stage == "training")

        return metrics

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            "backbones": self.backbones,
            "head": self.head,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        config = deserialize(config, custom_objects=custom_objects)
        return cls(**config)
