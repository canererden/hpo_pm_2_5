# Metrics
def mae(y_true, y_pred):
    """
    Mean absolute error regression loss.
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples    Returns:
        [float]: mean absolute error
    """
    return K.mean(K.abs(y_pred - y_true), axis=-1)

def mse(y_true, y_pred):
    """
    Mean squared error regression loss.
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples    Returns:
        [float]: mean squared error
    """
    return K.mean(K.square(y_pred - y_true), axis=-1)

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples
    Returns:
        [float]: root mean squared error
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# Root Mean Square Logarithmic Error
def rmsle(y_true, y_pred):
    """
    Root Mean Squared Logarithm Error
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples
    Returns:
        [float]: root mean squared logarithm error
    """
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))

def r2(y_true, y_pred):
    """
    R^2 (coefficient of determination) regression score function.
    Best possible score is 1.0, lower values are worse.
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples
    Returns:
        [float]: R2
    """
    SS_res =  tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=-1)), axis=-1)
    return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))

def adj_r2(y_true, y_pred):
    """
    Adjusted R2 regression score function with default inputs.
    Best possible score is 1.0, lower values are worse.
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples
    Returns:
        [float]: adjusted R2
    """
    SS_res =  tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=-1)), axis=-1)
    return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon())) * (1 - (1 - r2(y_true, y_pred)) * (tf.cast(tf.size(y_true), tf.float32) - 1) / (tf.cast(tf.size(y_true), tf.float32) - tf.cast(tf.rank(y_true), tf.float32) - 1))

def mape(y_true, y_pred):
    """
    Mean absolute percentage error regression loss.
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples
    Returns:
        [float]: mean absolute percentage error
    """
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
    return 100. * K.mean(diff, axis=-1)

# Mean squared logarithmic Error
def msle(y_true, y_pred):
    """
    Mean squared logarithmic error regression loss.
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples
    Returns:
        [float]: mean squared logarithmic error
    """
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)

# Symmetric mean absolute percentage error
def smape(y_true, y_pred):
    """
    Symmetric mean absolute percentage error regression loss.
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples
    Returns:
        [float]: symmetric mean absolute percentage error
    """
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
    return 100. * K.mean(K.mean(diff, axis=-1))

# Normalized Root Mean Squared Error
def nrmse(y_true, y_pred):
    """
    Normalized Root Mean Squared Error
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples
    Returns:
        [float]: normalized root mean squared error
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) / K.mean(K.abs(y_true), axis=-1)