""" Ranger
Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


def centralized_gradient(x, use_gc=True, gc_conv_only=False):
    '''credit - https://github.com/Yonghongwei/Gradient-Centralization '''
    if use_gc:
        if gc_conv_only:
            if len(x.shape) > 3:
                mean = tf.reduce_mean(x, axis=tuple(range(1, len(x.shape))), keepdims=True)
                x = x - mean
        else:
            if len(x.shape) > 1:
                mean = tf.reduce_mean(x, axis=tuple(range(1, len(x.shape))), keepdims=True)
                x = x - mean
    return x


class Ranger(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=.95,
        beta2=0.999,
        epsilon=1e-5,
        weight_decay=0,
        alpha=0.5,
        k=6,
        N_sma_threshhold=5,
        use_gc=True,
        gc_conv_only=False,
        gc_loc=True,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="ranger2020",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = k
        self.N_sma_threshhold = N_sma_threshhold
        self.use_gc = use_gc
        self.gc_conv_only = gc_conv_only
        self.gc_loc = gc_loc
        print(
            f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}")
        if (self.use_gc and self.gc_conv_only == False):
            print(f"GC applied to both conv and fc layers")
        elif (self.use_gc and self.gc_conv_only == True):
            print(f"GC applied to conv layers only")

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.slow_buffer = []
        for var in var_list:
            var_fp32 = tf.Variable(tf.cast(var, 'float32'))
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var_fp32, name="exp_avg"
                )
            )
            self.exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var_fp32, name="exp_avg_sq"
                )
            )
            self.slow_buffer.append(tf.Variable(var))
            self._track_variable(self.slow_buffer[-1])

    def update_step(self, gradient, variable, learning_rate):
        if gradient.dtype != tf.float32:
            gradient = tf.cast(gradient, 'float32')
        if variable.dtype != tf.float32:
            variable_fp32 = tf.cast(variable, 'float32')
        else:
            variable_fp32 = tf.convert_to_tensor(variable)
        lr = tf.cast(learning_rate, variable_fp32.dtype)
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'Ranger optimizer does not support sparse gradients')
        
        # begin computations
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        
        # GC operation for Conv layers and FC layers
        # if len(gradient.shape) > self.gc_gradient_threshold:
        #     gradient = gradient - tf.reduce_mean(gradient, axis=tuple(range(1, len(gradient.shape))), keepdims=True)
        if self.gc_loc:
            gradient = centralized_gradient(gradient, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)

        step = tf.cast(self.iterations + 1, variable_fp32.dtype)

        # compute variance mov avg
        exp_avg_sq.assign(self.beta2 * exp_avg_sq + (1 - self.beta2) * tf.square(gradient))

        # compute mean moving avg
        exp_avg.assign(self.beta1 * exp_avg + (1 - self.beta1) * gradient)

        beta2_t = self.beta2 ** step
        N_sma_max = 2 / (1 - self.beta2) - 1
        N_sma = N_sma_max - 2 * step * beta2_t / (1 - beta2_t)
        
        step_size = tf.sqrt(
                (1 - beta2_t)
                * (N_sma - 4)
                / (N_sma_max - 4)
                * (N_sma - 2)
                / N_sma
                * N_sma_max
                / (N_sma_max - 2)
            ) / (1 - self.beta1 ** step)
        
        def true_fn():
            denom = denom = tf.sqrt(exp_avg_sq) + self.epsilon
            G_grad = exp_avg / denom
            return G_grad
        
        def false_fn():
            return exp_avg

        # if self.weight_decay != 0:
        #     variable_fp32.assign_sub(self.weight_decay * lr * variable_fp32)

        # apply lr
        G_grad = tf.cond(N_sma > self.N_sma_threshhold, true_fn, false_fn)

        if self.weight_decay != 0:
            G_grad += self.weight_decay * variable_fp32
        # GC operation
        if self.gc_loc == False:
            G_grad = centralized_gradient(G_grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)

        variable_fp32 += -step_size * lr * G_grad
        variable.assign(tf.cast(variable_fp32, variable.dtype))

        # integrated look ahead...
        # we do it at the param level instead of group level
        def true_fn():
            # get access to slow param tensor
            slow_p = self.slow_buffer[self._get_variable_index(variable)]
            # (fast weights - slow weights) * alpha
            slow_p.assign_add(self.alpha * (variable- slow_p))
            # copy interpolated weights to RAdam param tensor
            variable.assign(slow_p)
        
        def false_fn():
            pass
        
        tf.cond(step % self.k == 0, true_fn, false_fn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "k": self.k,
                "N_sma_threshhold": self.N_sma_threshhold,
                "use_gc": self.use_gc,
                "gc_conv_only": self.gc_conv_only,
                "gc_loc": self.gc_loc,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass