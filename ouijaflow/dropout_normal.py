from edward.models import RandomVariable, Normal
from tensorflow.contrib.distributions import Distribution
import tensorflow as tf

class DropoutNormal(RandomVariable, Distribution): 
    """Dropout-Normal random variable
    Models mixture of a point mass at 0 and an amplified component with a Gaussian Likelihood,
    where the probability of 0 is specified by an additional tensor p_dropout.

    """
    def __init__(self,
                p_dropout, loc, scale,
                validate_args = False,
                allow_nan_stats = False,
                name = "DropoutNormal",
                *args, **kwargs):
        """Initialise DropoutNormal random variable

        Parameters
        ------
        p_dropout: tf.Tensor
        loc: tf.Tensor
        scale: tf.Tensor

        """

        parameters = {'p_dropout': p_dropout, 'loc': loc, 'scale': scale}

        with tf.name_scope(name, values = [p_dropout, loc, scale]):
            with tf.control_dependencies([
                tf.assert_positive(scale)
            ] if validate_args else []):
                self._p_dropout = tf.identity(p_dropout, name = "p_dropout")
                self._loc = tf.identity(loc, name = "loc")
                self._scale = tf.identity(scale, name = "scale")
        
        super(DropoutNormal, self).__init__(
            dtype = self._loc.dtype,
            validate_args = False,
            allow_nan_stats = False,
            reparameterization_type = tf.contrib.distributions.FULLY_REPARAMETERIZED,
            parameters = parameters,
            graph_parents = [self._p_dropout, self._loc, self._scale],
            name = name,
            *args, **kwargs
        )
        self._kwargs = parameters

    def _batch_shape(self):
        return tf.broadcast_static_shape(
            self._loc.get_shape(),
            self._scale.get_shape())

    def _get_event_shape(self):
        # change to just tensorshape
        return tf.placeholder(tf.float32, shape = []).get_shape()

    def _log_prob(self, value):
        prob_dropout_logit = self._p_dropout # 1.763 - 1.156 * self._loc

        lp_amp = -prob_dropout_logit + tf.log_sigmoid(prob_dropout_logit)
        lp_drop =  tf.log_sigmoid(prob_dropout_logit)

        is_zero = tf.equal(value, tf.zeros(value.shape))
 
        # If value == 0
        log_p_truezero = lp_amp + Normal(loc = self._loc, scale = self._scale)._log_prob(value)
        log_p_zero = tf.reduce_logsumexp(tf.stack([lp_drop, log_p_truezero]), axis = 0) 

        # If value > 0
        log_p_nonzero = Normal(loc = self._loc, scale = self._scale)._log_prob(value) + lp_amp

        log_prob = tf.where(is_zero, log_p_zero, log_p_nonzero)
 
        return log_prob




    def _sample_n(self, n, seed = None):
        batch_shape = tf.convert_to_tensor(self._batch_shape(), dtype = "int32")
        n_tens = tf.convert_to_tensor([n], dtype = "int32")

        amplified = tf.random_normal(shape = tf.concat([n_tens, batch_shape], axis = 0),
                                    mean = self._loc,
                                    stddev = self._scale)
        dropout = tf.zeros(shape = amplified.shape)

        p_dropout_tiled = tf.stack([tf.nn.sigmoid(self._p_dropout) for nn in range(n)])

        rand_sample = tf.where(tf.random_uniform(shape = p_dropout_tiled.shape) > p_dropout_tiled,
                                dropout, amplified)
        
        return rand_sample


        
        


