"""RAdam Optimizer.
Implementation lifted from: https://github.com/LiyuanLucasLiu/RAdam
Paper: `On the Variance of the Adaptive Learning Rate and Beyond` - https://arxiv.org/abs/1908.03265

Copyright 2025 NoteDance
"""
import tensorflow as tf
from optimizers import optimizer
import multiprocessing as mp


class RAdam(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="radam",
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

    def build(self, var_list):
        self.manager = mp.Manager()
        if self.built:
            self.exp_avg = self.manager.list(self.exp_avg)
            self.exp_avg_sq = self.manager.list(self.exp_avg_sq)
            self.buffer = self.manager.list(self.buffer)
            return
        super().build(var_list)
        self.exp_avg = self.manager.list()
        self.exp_avg_sq = self.manager.list()
        for var in var_list:
            var = tf.cast(var, 'float32')
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self.exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_sq"
                )
            )

    def update_step(self, gradient, variable, learning_rate):
        if variable.dtype != tf.float32:
            variable_fp32 = tf.cast(variable, 'float32')
        else:
            variable_fp32 = tf.convert_to_tensor(variable)
        lr = tf.cast(learning_rate, variable.dtype)
        
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        beta1, beta2 = self.beta1, self.beta2
        
        exp_avg_sq.assign(beta2 * exp_avg_sq + (1 - beta2) * tf.multiply(gradient, gradient))
        exp_avg.assign(beta1 * exp_avg + (1 - beta1) * gradient)

        step = tf.cast(self.iterations + 1, variable.dtype)
        beta2_t = beta2 ** step
        num_sma_max = 2 / (1 - beta2) - 1
        num_sma = num_sma_max - 2 * step * beta2_t / (1 - beta2_t)
        
        def true_fn():
            denom = tf.sqrt(exp_avg_sq) + self.epsilon
            
            step_size = lr * tf.sqrt(
                (1 - beta2_t) *
                (num_sma - 4) / (num_sma_max - 4) *
                (num_sma - 2) / num_sma *
                num_sma_max / (num_sma_max - 2)) / (1 - beta1 ** step)
            
            update = -step_size * exp_avg / denom
            return update
        
        def false_fn():
            step_size = lr / (1 - beta1 ** step)
            update = -step_size * exp_avg
            return update
        
        # more conservative since it's an approximated value
        update = tf.cond(num_sma >= 5, true_fn, false_fn)
        
        if self.weight_decay != 0:
            variable_fp32 += -self.weight_decay * lr * variable_fp32
        
        variable_fp32 += update
        
        variable.assign(tf.cast(variable_fp32, variable.dtype))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass