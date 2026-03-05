import keras

from bayesflow.types import Tensor
from bayesflow.utils.decorators import sanitize_input_shape
from bayesflow.utils.serialization import serializable

from .multihead_attention import MultiHeadAttention


@serializable("bayesflow.networks")
class SetAttention(MultiHeadAttention):
    """Implements the SAB block from [1] which represents learnable self-attention.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    # noinspection PyMethodOverriding
    @sanitize_input_shape
    def build(self, input_set_shape):
        self.call(keras.ops.zeros(input_set_shape))

    def call(self, x: Tensor, training: bool = False, attention_mask: Tensor = None, **kwargs) -> Tensor:
        """Performs the forward pass through the self-attention layer. Note that this model
        should not use a caucal mask, as no ordering in the sequence is assumed.

        Parameters
        ----------
        x  : Tensor (e.g., np.ndarray, tf.Tensor, ...)
            Input of shape (batch_size, set_size, input_dim)
        training   : boolean, optional (default - True)
            Passed to the optional internal dropout and spectral normalization
            layers to distinguish between train and test time behavior.
        attention_mask: a boolean mask of shape `(batch_size, set_size, set_size)`, that prevents
            attention to certain positions. The boolean mask specifies which
            query elements can attend to which key elements, 1 indicates
            attention and 0 indicates no attention. Broadcasting can happen for
            the missing batch dimensions and the head dimension.
        **kwargs   : dict, optional (default - {})
            Additional keyword arguments passed to the internal attention layer,
            such as ``attention_mask`` or ``return_attention_scores``

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, set_size, output_dim)
        """

        return super().call(x, x, training=training, attention_mask=attention_mask, **kwargs)

    # noinspection PyMethodOverriding
    @sanitize_input_shape
    def compute_output_shape(self, input_shape):
        return keras.ops.shape(self.call(keras.ops.zeros(input_shape)))
