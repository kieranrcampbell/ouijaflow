import tensorflow as tf
ds = tf.contrib.distributions
import numpy as np


class LogitShiftBijector(ds.bijectors.Bijector):
    """

    """
    def __init__(self, a, b, event_ndims=0, validate_args=False, name="ExpShift"):
        # self.event_ndims = event_ndims
        self.a = tf.convert_to_tensor(a, name = "a")
        self.b = tf.convert_to_tensor(b, name = "b")
        super(LogitShiftBijector, self).__init__(event_ndims=event_ndims, validate_args=validate_args, name=name)

    def _forward(self, x):
        return (self.a + self.b * tf.exp(x)) / (1 + tf.exp(x))

    def _inverse(self, y):
        return tf.log(y - self.a) - tf.log(self.b - y)

    def _inverse_log_det_jacobian(self, y):
        event_dims = self._event_dims_tensor(y)
        ildj = tf.log(tf.abs(1 / (y - self.a) - 1 / (self.b - y)))

        return tf.reduce_sum(ildj, axis = event_dims)

    def _event_dims_tensor(self, sample):
        """Return a 1D `int32` tensor: `range(rank(sample))[-event_ndims:]`."""
        if self.event_ndims is None:
          raise ValueError("Jacobian cannot be computed with unknown event_ndims")
        static_event_ndims = tf.contrib.util.constant_value(tf.convert_to_tensor(self.event_ndims))
        static_rank = sample.get_shape().ndims
        if static_event_ndims is not None and static_rank is not None:
          return tf.convert_to_tensor(
              static_rank + np.arange(-static_event_ndims, 0).astype(np.int32))

        if static_event_ndims is not None:
          event_range = np.arange(-static_event_ndims, 0).astype(np.int32)
        else:
          event_range = tf.range(-self.event_ndims, 0, dtype=dtypes.int32)

        if static_rank is not None:
          return event_range + static_rank
        else:
          return event_range + tf.rank(sample)
