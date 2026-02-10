import keras

from bayesflow.types import Tensor
from bayesflow.utils.decorators import sanitize_input_shape
from bayesflow.utils.serialization import serializable

from .mab import MultiHeadAttentionBlock


@serializable("bayesflow.networks")
class SetAttentionBlock(MultiHeadAttentionBlock):
    """Implements the SAB block from [1] which represents learnable self-attention.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    # noinspection PyMethodOverriding
    @sanitize_input_shape
    def build(self, input_set_shape):
        self.call(keras.ops.zeros(input_set_shape))

    def call(self, input_set: Tensor, training: bool = False, **kwargs) -> Tensor:
        """Performs the forward pass through the self-attention layer.

        Parameters
        ----------
        input_set  : Tensor (e.g., np.ndarray, tf.Tensor, ...)
            Input of shape (batch_size, set_size, input_dim)
        training   : boolean, optional (default - True)
            Passed to the optional internal dropout and spectral normalization
            layers to distinguish between train and test time behavior.
        **kwargs   : dict, optional (default - {})
            Additional keyword arguments passed to the internal attention layer,
            such as ``attention_mask`` or ``return_attention_scores``

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, set_size, output_dim)
        """

        return super().call(input_set, input_set, training=training, **kwargs)

    # noinspection PyMethodOverriding
    @sanitize_input_shape
    def compute_output_shape(self, input_set_shape):
        return keras.ops.shape(self.call(keras.ops.zeros(input_set_shape)))
