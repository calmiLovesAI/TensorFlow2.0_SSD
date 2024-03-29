import tensorflow as tf


def clip_by_value(t, clip_value_min=None, clip_value_max=None):
    t = tf.convert_to_tensor(t)
    if clip_value_min is None and clip_value_max is None:
        raise ValueError("上下限不能同时是None")
    if clip_value_max is None:
        return tf.math.maximum(t, clip_value_min)
    if clip_value_min is None:
        return tf.math.minimum(t, clip_value_max)
    t_max = tf.math.maximum(t, clip_value_min)
    return tf.math.minimum(t_max, clip_value_max)


def log_sum_exp(x):
    x_max = tf.math.reduce_max(x)
    return tf.math.log(tf.math.reduce_sum(input_tensor=tf.math.exp(x - x_max), axis=1, keepdims=True)) + x_max
