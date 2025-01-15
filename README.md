# AdaBelief

**Overview**:

The `AdaBelief` optimizer is a modification of the Adam optimizer designed to adapt the learning rate to the gradient’s variability. This approach makes it particularly effective for handling noisy gradients and improving generalization. It supports advanced features like rectification (inspired by RAdam), weight decay, gradient clipping, and the ability to degenerate into SGD when required.

**Features**:
- **Adaptive Learning Rate**: Learns from the gradient’s variability for more adaptive parameter updates.
- **Weight Decay**: Supports both decoupled and standard weight decay.
- **Rectified Updates**: Option to use rectification for variance control (RAdam-inspired).
- **AMSGrad**: Optional feature to use the maximum of second-moment estimates for normalization.
- **Gradient Clipping**: Supports multiple forms of gradient clipping (`clipnorm`, `clipvalue`, `global_clipnorm`).
- **Gradient Accumulation**: Enables gradient accumulation across steps for large-batch training.
- **EMA Support**: Supports exponential moving average for model weights.

---

**Parameters**:
- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta_1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta_2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-16)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay. Applies either decoupled or standard decay based on `decoupled_decay`.
- **`amsgrad`** *(bool, default=False)*: Whether to use the AMSGrad variant.
- **`decoupled_decay`** *(bool, default=True)*: Enables decoupled weight decay as described in AdamW.
- **`fixed_decay`** *(bool, default=False)*: Uses fixed weight decay instead of scaling it by the learning rate.
- **`rectify`** *(bool, default=True)*: Whether to apply rectified updates inspired by RAdam.
- **`degenerated_to_sgd`** *(bool, default=True)*: Degenerates into SGD in low-variance scenarios.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients.
- **`name`** *(str, default="adabelief")*: Name of the optimizer.

---

**Methods**:

**`__setstate__(state)`**
Restores optimizer state and ensures `amsgrad` is set to `False`.

**`reset()`**
Initializes or resets optimizer variables (`exp_avg`, `exp_avg_var`, etc.).

**`build(var_list)`**
Creates necessary optimizer state variables (`exp_avg`, `exp_avg_var`, etc.) for the given trainable variables.

**`update_step(gradient, variable, learning_rate)`**
Performs a single optimization step for the given variable and gradient.

- **`gradient`**: The gradient tensor.
- **`variable`**: The trainable variable to update.
- **`learning_rate`**: The learning rate for the current step.

**`get_config()`**
Returns the configuration of the optimizer as a dictionary.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.adabelief import AdaBelief

# Instantiate optimizer
optimizer = AdaBelief(
    learning_rate=1e-3,
    weight_decay=1e-2,
    rectify=True,
    decoupled_decay=True
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AdamP

**Overview**:

The `AdamP` optimizer is a modification of the Adam optimizer that aims to slow down the increase of weight norms in momentum-based optimizers. This is particularly useful for improving generalization and preventing overfitting. The optimizer uses a projection step to decouple sharp and flat components of the gradients, effectively reducing sensitivity to noise.

**Features**:
- **Projection Mechanism**: Decouples flat and sharp components of the gradients to mitigate overfitting.
- **Weight Decay**: Includes an adjusted weight decay term with a configurable ratio.
- **Momentum Updates**: Supports standard and Nesterov momentum.
- **Gradient Clipping**: Compatible with multiple gradient clipping strategies (`clipnorm`, `clipvalue`, `global_clipnorm`).
- **Gradient Accumulation**: Allows for gradient accumulation over multiple steps.
- **EMA Support**: Provides exponential moving average functionality for model weights.

---

**Parameters**:
- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta_1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta_2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: Weight decay coefficient.
- **`delta`** *(float, default=0.1)*: Threshold for decoupling sharp and flat gradient components.
- **`wd_ratio`** *(float, default=0.1)*: Ratio for scaling weight decay during projection.
- **`nesterov`** *(bool, default=False)*: Enables Nesterov momentum.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average for model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum value for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting model weights with EMA.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps over which gradients are accumulated.
- **`name`** *(str, default="adamp")*: Name of the optimizer.

---

**Methods**:

**`build(var_list)`**
Initializes optimizer state variables, such as first and second moment estimates (`exp_avg` and `exp_avg_sq`).

- **`var_list`** *(list of variables)*: List of trainable variables.

---

**`update_step(gradient, variable, learning_rate)`**
Performs a single optimization step for a given gradient and variable.

- **`gradient`** *(Tensor)*: The gradient tensor for the variable.
- **`variable`** *(Tensor)*: The trainable variable to update.
- **`learning_rate`** *(float)*: Learning rate for the current step.

---

**`get_config()`**
Returns the optimizer configuration as a dictionary, including all hyperparameters.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.adamp import AdamP

# Define the optimizer
optimizer = AdamP(
    learning_rate=1e-3,
    weight_decay=1e-2,
    delta=0.1,
    wd_ratio=0.1,
    nesterov=True
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# LaProp

**Overview**:

The `LaProp` optimizer is an adaptive gradient optimization algorithm that improves upon Adam by dynamically adjusting learning rates in proportion to the gradients. It includes optional features like centered second moments, AMSGrad stabilization, and weight decay, making it a versatile optimizer for deep learning tasks.

**Features**:
- **Learning Rate Adaptation**: Adjusts learning rates dynamically using gradient-based corrections.
- **Centered Updates**: Optionally centers second-moment estimates to improve convergence.
- **AMSGrad**: Stabilizes training by maintaining the maximum second-moment running average.
- **Weight Decay**: Regularizes weights during optimization.
- **Gradient Clipping**: Compatible with multiple gradient clipping techniques.
- **Gradient Accumulation**: Supports accumulation for distributed or large-batch training.

---

**Parameters**:
- **`learning_rate`** *(float, default=4e-4)*: Base step size for parameter updates.
- **`beta_1`** *(float, default=0.9)*: Coefficient for the moving average of the first moment (mean of gradients).
- **`beta_2`** *(float, default=0.999)*: Coefficient for the moving average of the second moment (variance of gradients).
- **`epsilon`** *(float, default=1e-15)*: Small constant for numerical stability.
- **`amsgrad`** *(bool, default=False)*: If `True`, uses the AMSGrad variant of the optimizer.
- **`centered`** *(bool, default=False)*: If `True`, centers the second-moment estimate for better stability.
- **`weight_decay`** *(float, default=0)*: Weight decay coefficient for L2 regularization.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) for model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum value for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting model weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
- **`steps_before_using_centered`** *(int, default=10)*: Minimum steps before enabling centered updates.
- **`name`** *(str, default="laprop")*: Name of the optimizer.

---

**Methods**:

**`build(var_list)`**
Initializes state variables for the optimizer, including:
- First and second moment estimates (`exp_avg`, `exp_avg_sq`).
- Learning rate-based adjustments for first and second moments.
- Optional variables for AMSGrad and centered updates.

- **`var_list`** *(list of variables)*: List of trainable variables to optimize.

---

**`update_step(gradient, variable, learning_rate)`**
Performs a single optimization step for a variable using its gradient.

---

**`get_config()`**
Returns the optimizer configuration as a dictionary, including all hyperparameters for easy reinitialization.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.laprop import LaProp

# Define the optimizer
optimizer = LaProp(
    learning_rate=4e-4,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-15,
    amsgrad=True,
    centered=True,
    weight_decay=1e-2,
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Lars

**Overview**:

The `Lars` optimizer is an implementation of **Layer-wise Adaptive Rate Scaling (LARS)**, a variant of stochastic gradient descent (SGD) designed for large-batch training. It combines weight decay and trust region-based learning rate adaptation, ensuring effective scaling for deep learning models with high-dimensional parameters. This implementation also includes optional **LARC (Layer-wise Adaptive Rate Clipping)**, momentum, and Nesterov updates.

**Features**:
- **Layer-wise Learning Rate Scaling**: Adapts learning rates based on parameter and gradient norms.
- **LARC Clipping**: Ensures trust ratio stays within a bounded range.
- **Momentum and Nesterov Updates**: Enhances optimization convergence speed.
- **Weight Decay**: Applies regularization to prevent overfitting.
- **Gradient Clipping**: Compatible with multiple gradient clipping techniques.
- **Gradient Accumulation**: Supports large-batch training via accumulation.

---

**Parameters**:
- **`learning_rate`** *(float, default=1.0)*: Base learning rate for parameter updates.
- **`momentum`** *(float, default=0)*: Momentum factor for the optimizer.
- **`dampening`** *(float, default=0)*: Dampening factor for momentum.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: Weight decay coefficient for L2 regularization.
- **`nesterov`** *(bool, default=False)*: Enables Nesterov momentum.
- **`trust_coeff`** *(float, default=0.001)*: Trust coefficient for scaling the learning rate based on LARS.
- **`trust_clip`** *(bool, default=False)*: If `True`, clips the trust ratio to a maximum value of 1.0.
- **`always_adapt`** *(bool, default=False)*: If `True`, forces the trust ratio to be computed regardless of weight decay.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) for model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum value for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting model weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
- **`name`** *(str, default="lars")*: Name of the optimizer.

---

**Methods**:

**`build(var_list)`**
Initializes the optimizer states for all variables. Each variable gets a state dictionary to store momentum buffers if `momentum` is enabled.

- **`var_list`** *(list of variables)*: List of trainable variables to optimize.

---

**`update_step(gradient, variable, learning_rate)`**
Performs a single optimization step for a variable using its gradient.

---

**`get_config()`**
Returns the optimizer configuration as a dictionary, including all hyperparameters for easy reinitialization.

---

**Example Usage**:
```python
import tensorflow as tf
from optimziers.lars import Lars

# Define the optimizer
optimizer = Lars(
    learning_rate=1.0,
    momentum=0.9,
    weight_decay=1e-4,
    trust_coeff=0.001,
    nesterov=True,
    trust_clip=True,
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# MADGRAD

**Overview**:

The `MADGRAD` optimizer is an advanced optimization algorithm designed for large-scale machine learning tasks. It is based on the paper [MADGRAD: Stochastic Optimization with Momentum Decay for Training Neural Networks](https://arxiv.org/abs/2101.11075) and provides benefits for sparse and dense gradient updates. This implementation is compatible with TensorFlow and includes support for advanced features like weight decay, momentum, and gradient accumulation.

**Features**:
- **Momentum Acceleration**: Enhances optimization convergence speed.
- **Second-Moment Accumulation**: Accumulates second-order gradient statistics for stable updates.
- **Decoupled Weight Decay**: Separates weight decay from gradient updates for improved performance.
- **Sparse Tensor Support**: Efficiently handles sparse gradients.
- **Gradient Clipping**: Compatible with multiple gradient clipping techniques.
- **Gradient Accumulation**: Supports large-batch training via accumulation.

---

**Parameters**:
- **`learning_rate`** *(float, default=1e-2)*: Base learning rate for parameter updates.
- **`epsilon`** *(float, default=1e-6)*: Small constant for numerical stability.
- **`momentum`** *(float, default=0.9)*: Momentum factor for the optimizer.
- **`weight_decay`** *(float, default=0)*: Weight decay coefficient for L2 regularization.
- **`decoupled_decay`** *(bool, default=False)*: If `True`, applies decoupled weight decay.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) for model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum value for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting model weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
- **`name`** *(str, default="madgrad")*: Name of the optimizer.

---

**Methods**:

**`build(var_list)`**
Initializes the optimizer states for all variables, creating:
1. `_grad_sum_sq`: Accumulator for second-order gradient statistics.
2. `_s`: Accumulator for gradients.
3. `_x0`: Initial variable values for momentum-based updates (if applicable).

- **`var_list`** *(list of variables)*: List of trainable variables to optimize.

---

**`update_step(gradient, variable, learning_rate)`**
Performs a single optimization step for a variable using its gradient.

---

**`get_config()`**
Returns the optimizer configuration as a dictionary, including all hyperparameters for easy reinitialization.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.madgrad import MADGRAD

# Define the optimizer
optimizer = MADGRAD(
    learning_rate=1e-2,
    momentum=0.9,
    weight_decay=1e-4,
    decoupled_decay=True,
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# MARS

**Overview**:

The **MARS** optimizer implements a novel optimization algorithm designed for training large-scale models effectively. It leverages variance reduction techniques, adaptive learning rates, and supports both AdamW-style and Lion-style updates for parameter optimization. MARS also incorporates specialized mechanisms to handle 1D and 2D gradients differently, ensuring efficiency and accuracy in various scenarios.

This implementation is based on the paper [MARS: Unleashing the Power of Variance Reduction for Training Large Models](https://arxiv.org/abs/2411.10438).

---

**Features**:
- **Variance Reduction**: Minimizes gradient variance to stabilize updates.
- **Customizable Update Rules**: Supports AdamW and Lion-style parameter updates.
- **Adaptive Gradient Clipping**: Provides enhanced control over gradient updates.
- **1D/2D Optimization**: Separate strategies for optimizing 1D and 2D parameters.
- **Weight Decay**: Integrated weight decay for L2 regularization.
- **Caution Mechanism**: Ensures robust updates by masking inconsistent gradient directions.
- **Gradient Accumulation Support**: Compatible with large-batch training.

---

**Parameters**:
- **`learning_rate`** *(float, default=3e-3)*: The learning rate for optimization.
- **`beta_1`** *(float, default=0.9)*: Coefficient for the first moment estimate.
- **`beta_2`** *(float, default=0.99)*: Coefficient for the second moment estimate.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay (L2 regularization).
- **`gamma`** *(float, default=0.025)*: Coefficient controlling the variance reduction term.
- **`mars_type`** *(str, default="adamw")*: Type of parameter update to use:
  - `"adamw"`: AdamW-style updates.
  - `"lion"`: Lion-style updates.
- **`optimize_1d`** *(bool, default=False)*: If `True`, applies MARS-specific updates to 1D parameters.
- **`lr_1d_factor`** *(float, default=1.0)*: Scaling factor for learning rate for 1D parameter updates.
- **`betas_1d`** *(tuple, optional)*: Separate `(beta1, beta2)` values for 1D parameters.
- **`caution`** *(bool, default=False)*: If `True`, applies a masking mechanism to stabilize updates.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) for model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum value for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting model weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
- **`name`** *(str, default="mars")*: Name of the optimizer.

---

**Methods**:

**`build(var_list)`**
Initializes the optimizer's state for the given variables, including:
- `_exp_avg`: First-moment estimates for each variable.
- `_exp_avg_sq`: Second-moment estimates for each variable.
- `_last_grad`: Stores the last gradient for variance reduction.
- `step`: Tracks the optimization step for each variable.

- **`var_list`** *(list of variables)*: List of trainable variables to optimize.

---

**`update_step(gradient, variable, learning_rate)`**
Performs a single optimization step for the given variable using its gradient.

---

**`get_config()`**
Returns the optimizer configuration as a dictionary for easy reinitialization.

---

**Example Usage**:

```python
import tensorflow as tf
from optimizers.mars import Mars

# Initialize the MARS optimizer
optimizer = Mars(
    learning_rate=3e-3,
    beta_1=0.9,
    beta_2=0.99,
    gamma=0.025,
    mars_type="adamw",
    optimize_1d=True,
    lr_1d_factor=0.8,
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# NAdam

**Overview**:

The **NAdam** optimizer is an implementation of the Nesterov-accelerated Adaptive Moment Estimation (Nadam) algorithm. Nadam extends the widely-used Adam optimizer by incorporating Nesterov momentum, providing faster convergence in some scenarios. This optimizer is particularly useful for tasks where momentum plays a critical role in overcoming saddle points and improving optimization dynamics.

The algorithm is described in:
- **"Incorporating Nesterov Momentum into Adam"** ([PDF link](http://cs229.stanford.edu/proj2015/054_report.pdf))
- **"On the Importance of Initialization and Momentum in Deep Learning"** ([PDF link](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf))

---

**Features**
- **Adaptive Learning Rates**: Adjusts learning rates based on gradients and their variance.
- **Nesterov Momentum**: Combines momentum-based updates with gradient adjustments, offering better convergence in some cases.
- **Bias Correction**: Corrects bias introduced during the initialization of moment estimates.
- **Weight Decay Support**: Includes L2 regularization for better generalization.
- **Warm Momentum Schedule**: Gradually adjusts momentum during training for smoother updates.

---

**Parameters**:
- **`learning_rate`** *(float, default=2e-3)*: Learning rate for the optimizer.
- **`beta_1`** *(float, default=0.9)*: Coefficient for the first moment estimate (momentum term).
- **`beta_2`** *(float, default=0.999)*: Coefficient for the second moment estimate (variance term).
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in divisions.
- **`weight_decay`** *(float, default=0)*: Weight decay coefficient for L2 regularization.
- **`schedule_decay`** *(float, default=4e-3)*: Decay factor for momentum scheduling.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) for model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum value for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
- **`name`** *(str, default="nadam")*: Name of the optimizer.

---

**Methods**

**`build(var_list)`**
Initializes optimizer state for the trainable variables.

- **`var_list`** *(list of variables)*: Trainable variables to optimize.
  
Initializations:
- `_exp_avg`: First moment (momentum) estimates.
- `_exp_avg_sq`: Second moment (variance) estimates.
- `_m_schedule`: Tracks the momentum schedule for warm restarts.
- `step`: Tracks the number of optimization steps for each variable.

---

**`update_step(gradient, variable, learning_rate)`**
Performs a single optimization step for the given variable using the provided gradient.

---

**`get_config()`**
Returns the optimizer's configuration as a dictionary for serialization and reinitialization.

---

**Example Usage**:

```python
import tensorflow as tf
from optimizers.nadam import NAdam

# Initialize the NAdam optimizer
optimizer = NAdam(
    learning_rate=2e-3,
    beta_1=0.9,
    beta_2=0.999,
    schedule_decay=4e-3,
    weight_decay=1e-4,
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# NvNovoGrad

**Overview**:

The **NvNovoGrad** optimizer is an implementation of NovoGrad, an optimization algorithm designed for deep learning that uses layer-wise adaptive moments for efficient and robust training. NovoGrad is particularly effective for large-scale and resource-constrained deep learning tasks, as it combines the benefits of Adam and L2 regularization while being computationally efficient.

The algorithm is described in:
- **"Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks"** ([arXiv link](https://arxiv.org/abs/1905.11286))

This implementation is inspired by NVIDIA's original implementation in PyTorch, used in speech recognition models like **Jasper**:
- [Jasper Example](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper)

---

**Features**:
- **Layer-wise Adaptive Learning Rates**: Adjusts learning rates for each layer, making it suitable for large-scale tasks.
- **Efficient Memory Usage**: Optimized for memory efficiency, making it ideal for resource-constrained environments.
- **Support for AMSGrad**: Includes an optional variant of NovoGrad with AMSGrad for improved convergence.
- **Gradient Averaging**: Option to average gradients for smoother updates.
- **Weight Decay**: Includes support for L2 regularization.

---

**Parameters**:
- **`learning_rate`** *(float, default=1e-3)*: Learning rate for the optimizer.
- **`beta_1`** *(float, default=0.95)*: Exponential decay rate for the first moment estimate (momentum term).
- **`beta_2`** *(float, default=0.98)*: Exponential decay rate for the second moment estimate (variance term).
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in divisions.
- **`weight_decay`** *(float, default=0)*: Weight decay coefficient for L2 regularization.
- **`grad_averaging`** *(bool, default=False)*: Enables gradient averaging for smoother updates.
- **`amsgrad`** *(bool, default=False)*: Enables the AMSGrad variant for convergence improvements.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) for model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum value for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
- **`name`** *(str, default="nvnovograd")*: Name of the optimizer.

---

**Methods**:

**`build(var_list)`**
Initializes optimizer state for the trainable variables.

- **`var_list`** *(list of variables)*: Trainable variables to optimize.
  
Initializations:
- `_exp_avg`: Stores first-moment estimates for each variable.
- `_exp_avg_sq`: Stores second-moment estimates (squared gradients).
- `_max_exp_avg_sq`: (Optional) Stores maximum second-moment estimates for AMSGrad.
- `step`: Tracks the number of optimization steps for each variable.

---

**`update_step(gradient, variable, learning_rate)`**
Performs a single optimization step for the given variable using the provided gradient.

---

**`get_config()`**
Returns the optimizer's configuration as a dictionary for serialization and reinitialization.

---

**Example Usage**:

```python
import tensorflow as tf
from optimizers.nvnovograd import NvNovoGrad

# Initialize the NvNovoGrad optimizer
optimizer = NvNovoGrad(
    learning_rate=1e-3,
    beta_1=0.95,
    beta_2=0.98,
    weight_decay=1e-4,
    grad_averaging=True,
    amsgrad=True,
)

# Compile a model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# RAdam

**Overview**:

The **RAdam (Rectified Adam)** optimizer is a variant of the Adam optimizer that incorporates a mechanism to rectify the variance of adaptive learning rates. This rectification improves stability and prevents early training instabilities, especially in the initial training phase. RAdam maintains the benefits of Adam while being more robust for a wide range of applications.

The algorithm is described in the paper:
- **"On the Variance of the Adaptive Learning Rate and Beyond"** ([arXiv link](https://arxiv.org/abs/1908.03265))

This implementation is inspired by the original PyTorch implementation:
- [RAdam GitHub Repository](https://github.com/LiyuanLucasLiu/RAdam)

---

**Features**:
- **Variance Rectification**: Dynamically adjusts learning rates to mitigate training instabilities during the early stages.
- **Automatic Transition**: Automatically transitions between SGD-like behavior and Adam-like behavior based on available statistics.
- **Weight Decay Support**: Includes support for L2 regularization.
- **Efficient Buffering**: Caches intermediate calculations for computational efficiency.

---

**Parameters**:
- **`learning_rate`** *(float, default=1e-3)*: Base learning rate for the optimizer.
- **`beta_1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimate (momentum term).
- **`beta_2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimate (variance term).
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in divisions.
- **`weight_decay`** *(float, default=0)*: Weight decay coefficient for L2 regularization.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) for model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum value for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
- **`name`** *(str, default="radam")*: Name of the optimizer.

---

**Methods**:

**`build(var_list)`**
Initializes the optimizer state for the trainable variables.

- **`var_list`** *(list of variables)*: List of trainable variables to optimize.

Initializations:
- `_exp_avg`: First-moment estimates for each variable.
- `_exp_avg_sq`: Second-moment estimates for each variable.
- `step`: Tracks the optimization steps for each variable.
- `buffer`: Stores intermediate calculations for variance rectification.

---

**`update_step(gradient, variable, learning_rate)`**
Performs a single optimization step for the given variable using the provided gradient.

---

**`get_config()`**
Returns a dictionary containing the optimizer's configuration, suitable for serialization or reinitialization.

---

**Example Usage**:

```python
import tensorflow as tf
from optimizers.radam import RAdam

# Initialize the RAdam optimizer
optimizer = RAdam(
    learning_rate=1e-3,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    weight_decay=1e-4,
)

# Compile a model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# SGDP

**Overview**:

The **SGDP (Stochastic Gradient Descent with Projection and Weight Decay)** optimizer is a variant of SGD that incorporates **decoupled weight decay regularization** and **gradient projection**. These features help control weight norm growth during training, improving convergence and performance. 

This algorithm is described in the paper:
- **"Slowing Down the Weight Norm Increase in Momentum-based Optimizers"** ([arXiv link](https://arxiv.org/abs/2006.08217)).

The implementation is inspired by the official repository:
- [AdamP GitHub Repository](https://github.com/clovaai/AdamP)

---

**Features**:
- **Projection Mechanism**: Projects gradients onto a subspace to reduce ineffective gradient directions and improve optimization efficiency.
- **Decoupled Weight Decay**: Effectively regularizes the model parameters without interfering with gradient updates.
- **Momentum and Nesterov Support**: Enhances optimization with momentum or Nesterov acceleration.
- **Flexibility**: Offers a range of hyperparameters for customization.

---

**Parameters**:
- **`learning_rate`** *(float, default=1e-3)*: Learning rate for the optimizer.
- **`momentum`** *(float, default=0)*: Momentum factor for SGD.
- **`dampening`** *(float, default=0)*: Dampening factor to control momentum updates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: L2 regularization coefficient (weight decay).
- **`delta`** *(float, default=0.1)*: Threshold for the cosine similarity in the projection mechanism.
- **`wd_ratio`** *(float, default=0.1)*: Weight decay ratio for decoupling.
- **`nesterov`** *(bool, default=False)*: If `True`, enables Nesterov momentum.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) for model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
- **`name`** *(str, default="sgdp")*: Name of the optimizer.

---

**Methods**:

**`build(var_list)`**
Initializes the optimizer state for the trainable variables.

- **`var_list`** *(list of variables)*: List of trainable variables.

Initialization:
- `_momentum`: Stores momentum terms for each variable.

---

**`update_step(gradient, variable, learning_rate)`**
Performs a single optimization step for the given variable using the provided gradient.

---

**`get_config()`**
Returns a dictionary containing the optimizer's configuration, suitable for serialization or reinitialization.

---

**Example Usage**:

```python
import tensorflow as tf
from optimizers.sgdp import SGDP

# Initialize the SGDP optimizer
optimizer = SGDP(
    learning_rate=1e-3,
    momentum=0.9,
    dampening=0.1,
    weight_decay=1e-4,
    delta=0.1,
    wd_ratio=0.1,
    nesterov=True,
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Adan

**Overview**:

The **Adan (Adaptive Nesterov Momentum)** optimizer is a next-generation optimization algorithm designed to accelerate training and improve convergence in deep learning models. It combines **adaptive gradient estimation** and **multi-step momentum** for enhanced performance.

This algorithm is introduced in the paper:
- **"Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"** ([arXiv link](https://arxiv.org/abs/2208.06677)).

The implementation is inspired by the official repository:
- [Adan GitHub Repository](https://github.com/sail-sg/Adan)

---

**Features**:
- **Adaptive Nesterov Momentum**: Improves gradient estimation using adaptive updates.
- **Gradient Difference Momentum**: Stabilizes updates by tracking gradient differences.
- **Decoupled Weight Decay**: Supports weight decay without interfering with gradient updates.
- **Multi-Tensor Operations**: Optimized for efficient parallel computations.
- **Bias Correction**: Ensures unbiased gradient estimates.

---

**Parameters**:
- **`learning_rate`** *(float, default=1e-3)*: Learning rate for the optimizer.
- **`beta_1`** *(float, default=0.98)*: Exponential decay rate for the first moment estimates.
- **`beta_2`** *(float, default=0.92)*: Exponential decay rate for gradient difference momentum.
- **`beta_3`** *(float, default=0.99)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Strength of weight decay regularization.
- **`no_prox`** *(bool, default=False)*: If `True`, disables proximal updates during weight decay.
- **`foreach`** *(bool, default=True)*: Enables multi-tensor operations for optimization.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) for model parameters.
- **`ema_momentum`** *(float, default=0.99)*: EMA momentum for parameter averaging.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting model parameters with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values in mixed-precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
- **`name`** *(str, default="adan")*: Name of the optimizer.

---

**Methods**:

**`build(var_list)`**
Initializes the optimizer state for the trainable variables.

- **`var_list`** *(list of variables)*: List of trainable variables.

Initialization:
- `exp_avg`: First moment estimates for each variable.
- `exp_avg_sq`: Second moment estimates for each variable.
- `exp_avg_diff`: Gradient difference momentum for each variable.
- `neg_pre_grad`: Stores the previous negative gradient for updates.

---

**`update_step(gradient, variable, learning_rate)`**
Performs a single optimization step for the given variable using the provided gradient.

---

**`get_config()`**
Returns a dictionary containing the optimizer's configuration, suitable for serialization or reinitialization.

---

**Example Usage**:

```python
import tensorflow as tf
from optimizers.adan import Adan

# Initialize the Adan optimizer
optimizer = Adan(
    learning_rate=1e-3,
    beta_1=0.98,
    beta_2=0.92,
    beta_3=0.99,
    weight_decay=0.01,
    use_ema=True,
    ema_momentum=0.999
)

# Compile a model
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Lamb

**Overview**:

The **Lamb (Layer-wise Adaptive Moments)** optimizer is an advanced optimization algorithm designed for large-scale machine learning tasks. It extends the Adam optimizer by incorporating a trust ratio to adjust learning rates across layers, making it particularly effective for training deep neural networks with large batch sizes. This implementation supports features like gradient clipping, bias correction, decoupled weight decay, and compatibility with PyTorch XLA for TPU acceleration.

This implementation is inspired by:
- **HabanaAI Model References** ([Source](https://github.com/HabanaAI/Model-References/blob/2b435114fe8e31f159b1d3063b8280ae37af7423/PyTorch/nlp/bert/pretraining/lamb.py)).
- **NVIDIA Deep Learning Examples** ([Source](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py)).
- **CybertronAI PyTorch Lamb** ([Source](https://github.com/cybertronai/pytorch-lamb)).

This version is tailored for environments without NVIDIA GPUs or APEX, offering similar behavior to FusedLamb and support for TensorFlow and TPU platforms.

---

**Features**:

- **Layer-wise Adaptive Moments**: Adjusts learning rates across layers based on the trust ratio, enabling better generalization and convergence.
- **Gradient Clipping**: Supports clipping gradients by norm to enhance stability during training.
- **Decoupled Weight Decay**: Option to decouple weight decay from gradient updates.
- **Bias Correction**: Handles bias in first and second moment estimates for improved optimization.
- **Compatibility**: Tested with TensorFlow and PyTorch XLA for TPU support.
- **Flexibility**: Offers customization through various hyperparameters.

---

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: Learning rate for the optimizer.
- **`bias_correction`** *(bool, default=True)*: If `True`, applies bias correction to the moment estimates.
- **`beta_1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta_2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-6)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.01)*: L2 regularization coefficient for weight decay.
- **`grad_averaging`** *(bool, default=True)*: If `True`, averages gradients for more robust updates.
- **`max_grad_norm`** *(float, default=1.0)*: Maximum norm for gradient clipping. Disabled if `None`.
- **`trust_clip`** *(bool, default=False)*: If `True`, clips the trust ratio to 1.0.
- **`always_adapt`** *(bool, default=False)*: Always adapt the learning rate using the trust ratio, even if weight decay is zero.
- **`caution`** *(bool, default=False)*: If `True`, applies cautious updates as per the "Cautious Optimizers" paper.
- **`decoupled_decay`** *(bool, default=False)*: If `True`, applies decoupled weight decay.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) for model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
- **`name`** *(str, default="lamb")*: Name of the optimizer.

---

**Methods**:

**`build(var_list)`**
Initializes the optimizer state for the trainable variables.

- **`var_list`** *(list of variables)*: List of trainable variables.

Initialization:
- `_exp_avg`: First moment estimates for each variable.
- `_exp_avg_sq`: Second moment estimates for each variable.
- `step`: Tracks the optimization steps.

---

**`update_step(grads, trainable_variables, learning_rate)`**
Performs a single optimization step for the given variables using the provided gradients.

---

**`get_config()`**
Returns a dictionary containing the optimizer's configuration for serialization or reinitialization.

---

**Example Usage**:

```python
import tensorflow as tf
from optimizers.lamb import Lamb

# Initialize the Lamb optimizer
optimizer = Lamb(
    learning_rate=1e-3,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-6,
    weight_decay=0.01,
    max_grad_norm=1.0,
    decoupled_decay=True,
    trust_clip=True,
)

# Compile a model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Adahessian

**Overview**:

The **Adahessian Optimizer** is an advanced second-order optimization algorithm that leverages the Hessian trace (approximated using Hutchinson's method) to adaptively scale learning rates for each parameter. Adahessian extends first-order optimization techniques like Adam by incorporating curvature information from the loss surface, which enables better adaptation to the optimization landscape, especially for highly non-convex problems.

---

**Features**:

- **Second-order optimization**: Uses an approximation of the Hessian diagonal to scale parameter updates.
- **Hutchinson’s approximation**: Efficiently computes the Hessian trace with minimal computational overhead.
- **Parameter-wise learning rates**: Adapts the step size for each parameter based on curvature information.
- **Compatible with distributed training**: Uses a deterministic generator to ensure reproducible results across devices.
- **Hessian smoothing for convolutions**: Optional averaging for convolutional kernels.

---

**Parameters**:

- **`learning_rate`** *(float)*: Initial learning rate (default: `0.1`).
- **`beta_1`** *(float)*: Exponential decay rate for the first moment estimates (default: `0.9`).
- **`beta_2`** *(float)*: Exponential decay rate for the Hessian diagonal squared estimates (default: `0.999`).
- **`epsilon`** *(float)*: Small value to prevent division by zero (default: `1e-8`).
- **`weight_decay`** *(float)*: L2 regularization factor for weights (default: `0.0`).
- **`hessian_power`** *(float)*: Scaling factor for the Hessian diagonal (default: `1.0`).
- **`update_each`** *(int)*: Frequency (in steps) for Hessian trace updates (default: `1`).
- **`n_samples`** *(int)*: Number of samples for Hutchinson’s approximation (default: `1`).
- **`avg_conv_kernel`** *(bool)*: Whether to average Hessian diagonals over convolutional kernel dimensions (default: `False`).
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) for model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
- **`name`** *(str, default="adahessian")*: Name of the optimizer.

---

**Methods**:

1. **`apply_gradients(grads_and_vars, tape)`**
   - Computes parameter updates based on gradients and Hessian traces.
   - Arguments:
     - `grads_and_vars`: A list of (gradient, variable) pairs.
     - `tape`: A TensorFlow `GradientTape` used to compute Hessians.
   - Returns: Current optimizer step count.

2. **`zero_hessian(trainable_variables)`**
   - Resets the accumulated Hessian traces to zero for each parameter.

3. **`set_hessian(grads, trainable_variables)`**
   - Computes the Hessian diagonal traces using Hutchinson's approximation.

4. **`update_step(grads, trainable_variables, learning_rate)`**
   - Executes the optimization step using the accumulated Hessian traces and gradient information.

5. **`get_config()`**
   - Returns the configuration of the optimizer, including all hyperparameters.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.adahessian import Adahessian

# Define model and loss
model = tf.keras.Sequential([...])
loss_fn = tf.keras.losses.MeanSquaredError()

# Initialize optimizer
optimizer = Adahessian(
    learning_rate=0.01, 
    beta_1=0.9, 
    beta_2=0.999, 
    weight_decay=0.01
)

# Training step
@tf.function
def train_step(x, y, model, optimizer):
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables), tape)

# Training loop
for epoch in range(epochs):
    for x_batch, y_batch in dataset:
        train_step(x_batch, y_batch, model, optimizer)
```

# Adopt

**Overview**:  

The **ADOPT (Adaptive Optimization with Trust)** optimizer is a novel variant of Adam designed to achieve optimal convergence rates with any value of \(\beta_2\). It introduces enhancements such as adaptive gradient scaling and cautious updates, making it suitable for diverse optimization scenarios, including tasks requiring stability and robustness in gradient updates.  

This TensorFlow implementation is adapted from the PyTorch version available in the [timm library](https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/adopt.py). The optimizer builds on concepts from Adam while adding innovative features for enhanced convergence and generalization.

---  

**Features**:

- **Adaptive Gradient Scaling**: Scales gradients adaptively for stable updates, especially in the presence of large or small values.  
- **Cautious Updates**: Optionally applies a "cautious optimizer" mechanism for safer gradient adjustments (see: [Cautious Optimizers](https://arxiv.org/abs/2411.16085)).  
- **Weight Decay Options**: Supports both standard and decoupled weight decay for improved regularization.  
- **Gradient Clipping**: Enables gradient clipping by norm or exponent for numerical stability.  
- **Complex Tensor Compatibility**: Handles complex-valued tensors seamlessly.  
- **Customization**: Extensive configurability for diverse training requirements.  

---

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: Learning rate for the optimizer.  
- **`beta_1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.  
- **`beta_2`** *(float, default=0.9999)*: Exponential decay rate for the second moment estimates.  
- **`epsilon`** *(float, default=1e-6)*: Small constant for numerical stability.  
- **`weight_decay`** *(float, default=0.0)*: Weight decay factor for L2 regularization.  
- **`clip_exp`** *(float, default=0.333)*: Exponent for gradient clipping.  
- **`decoupled`** *(bool, default=False)*: Whether to decouple weight decay from gradient updates.  
- **`caution`** *(bool, default=False)*: Enables cautious updates to prevent overshooting during optimization.  
- **`foreach`** *(bool, default=False)*: If `True`, processes variables in parallel for efficiency.  
- **`maximize`** *(bool, default=False)*: Maximizes the objective function instead of minimizing.  
- **`capturable`** *(bool, default=False)*: Enables capturable state for graph execution.  
- **`differentiable`** *(bool, default=False)*: Ensures the optimizer remains differentiable for higher-order optimization tasks.  
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) for model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
- **`name`** *(str, default="adopt")*: Name of the optimizer.

---

**Methods**:  

**`build(var_list)`**  
Initializes the optimizer state for the trainable variables.  

- **`var_list`** *(list of variables)*: List of variables to be optimized.  
Initialization:  
- `exp_avg`: Stores the first moment estimates.  
- `exp_avg_sq`: Stores the second moment estimates.  
- `step`: Tracks the optimization steps for each variable.  

---  

**`update_step(grads, trainable_variables, learning_rate)`**  
Performs a single optimization step for the given variables using their gradients.  

---  

**`get_config()`**  
Returns a dictionary containing the optimizer's configuration for saving or reinitialization.  

---  

**Example Usage**:  

```python
import tensorflow as tf
from optimizers.adopt import Adopt

# Initialize the ADOPT optimizer
optimizer = Adopt(
    learning_rate=1e-3,
    beta_1=0.9,
    beta_2=0.9999,
    epsilon=1e-6,
    weight_decay=0.01,
    clip_exp=0.333,
    decoupled=True,
    caution=True,
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# NAdamW

**Overview**:

The **NAdamW** optimizer is a novel optimization algorithm that extends the traditional AdamW optimizer by incorporating Nesterov momentum, improving both convergence speed and generalization performance. It is well-suited for training deep neural networks across various machine learning tasks and offers advanced features such as cautious updates and multi-tensor optimization paths.

This implementation is inspired by the algorithm in the MLCommons algorithmic efficiency repository and includes a multi-tensor path for better performance on large-scale models.

---

**Features**:

- **Nesterov Momentum**: Integrates Nesterov momentum into the AdamW framework for faster convergence.
- **Multi-Tensor Optimization**: Offers a path for optimizing multiple tensors simultaneously, improving computational efficiency.
- **Cautious Updates**: Implements cautious optimizations for improved training stability as per the "Cautious Optimizers" paper.
- **Decoupled Weight Decay**: Decouples weight decay from gradient updates for better control.
- **Gradient Clipping**: Supports gradient clipping by norm, value, or globally, ensuring numerical stability.
- **Customizability**: Extensive hyperparameter support for flexible training setups.

---

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: Learning rate for the optimizer.
- **`beta_1`** *(float, default=0.9)*: Coefficient for computing running averages of gradient moments.
- **`beta_2`** *(float, default=0.999)*: Coefficient for computing running averages of squared gradients.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=1e-2)*: Weight decay coefficient for L2 regularization.
- **`caution`** *(bool, default=False)*: If `True`, applies cautious updates for enhanced training stability.
- **`maximize`** *(bool, default=False)*: If `True`, maximizes the objective function instead of minimizing it.
- **`foreach`** *(bool, optional)*: Enables multi-tensor optimization paths if `True`.
- **`capturable`** *(bool, default=False)*: If `True`, supports CUDA graph capturing for enhanced performance.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) of model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before an optimization step.
- **`name`** *(str, default="nadamw")*: Name of the optimizer.

---

**Methods**:

**`build(var_list)`**
Initializes the optimizer state for the trainable variables.

- **`var_list`** *(list of variables)*: List of trainable variables.

Initialization:
- **`exp_avg`**: First moment estimates for each variable.
- **`exp_avg_sq`**: Second moment estimates for each variable.
- **`step`**: Tracks optimization steps for bias correction.

---

**`update_step(grads, trainable_variables, learning_rate)`**
Performs a single optimization step using the given gradients.

---

**`get_config()`**
Returns a dictionary containing the optimizer's configuration for serialization or reinitialization.

---

**Functional API**
**`nadamw`** performs the core NAdamW computation, supporting single-tensor and multi-tensor paths.

- **Parameters**:
  - **`params`**: List of model parameters to be updated.
  - **`grads`**: Corresponding gradients for each parameter.
  - **`exp_avgs`**: List of first moment estimates.
  - **`exp_avg_sqs`**: List of second moment estimates.
  - **`state_steps`**: Optimization steps for bias correction.
  - Additional hyperparameters for controlling the optimization process.

---

**Example Usage**:

```python
import tensorflow as tf
from nadamw import NAdamW

# Initialize the optimizer
optimizer = NAdamW(
    learning_rate=1e-3,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    weight_decay=1e-2,
    caution=True,
    maximize=False
)

# Compile a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

# AdafactorBigVision

**Overview**:

The **AdafactorBigVision** optimizer is an adaptation of the Adafactor optimization algorithm, tailored specifically for scaling large vision transformer models. This implementation is inspired by the algorithm described in the paper ["Scaling Vision Transformers"](https://arxiv.org/abs/2106.04560) and the [Big Vision](https://github.com/google-research/big_vision) codebase. 

Adafactor is designed to reduce memory usage during training by using a factored approximation for the second-moment estimates. It achieves efficiency and scalability while maintaining effective optimization. The **Big Vision variant** introduces several enhancements, including factored second moments for high-dimensional variables, support for momentum, gradient clipping, and cautious updates for improved robustness.

---

**Features**:

- **Factored Second Moments**: Efficient approximation of second-moment estimates for variables with large dimensions.
- **Gradient Clipping**: Optional clipping of updates to ensure stability during training.
- **Momentum Updates**: Optional momentum-based updates for better convergence properties.
- **Cautious Optimizer Support**: Implements the "Cautious Optimizers" approach to mitigate risks during optimization.
- **Scalability**: Optimized for high-dimensional parameter spaces such as those in vision transformers.

---

**Parameters**:

- **`learning_rate`** *(float, default=1.0)*: The base learning rate for updates.
- **`epsilon`** *(float, optional)*: A small constant added for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 regularization (weight decay).
- **`min_dim_size_to_factor`** *(int, default=16)*: Minimum size of dimensions to apply factored second moments.
- **`decay_rate`** *(float, default=0.8)*: Exponential decay rate for second-moment estimation.
- **`decay_offset`** *(int, default=0)*: Offset for the decay schedule.
- **`beta2_cap`** *(float, default=0.999)*: Maximum cap for the second-moment decay factor.
- **`momentum`** *(float, optional)*: Momentum factor for updates. If `None`, momentum is not applied.
- **`momentum_dtype`** *(dtype, default=tf.bfloat16)*: Data type for momentum storage.
- **`clipping_threshold`** *(float, optional)*: Threshold for gradient clipping. Disabled if `None`.
- **`unscaled_wd`** *(bool, default=False)*: If `True`, applies unscaled weight decay independent of the learning rate.
- **`caution`** *(bool, default=False)*: Enables cautious updates as per the "Cautious Optimizers" paper.
- **`foreach`** *(bool, default=False)*: Enables multi-tensor optimization for enhanced performance (not yet implemented).
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) of model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before an optimization step.
- **`name`** *(str, default="adafactor_bv")*: Name of the optimizer.

---

**Methods**:

**`build(var_list)`**  
Initializes the optimizer state for the trainable variables.

- **`var_list`** *(list of variables)*: List of trainable model parameters.

Initialization:
- `exp_avg`: Momentum storage (if enabled).
- `exp_avg_sq_r` and `exp_avg_sq_c`: Row and column factored second moments (if applicable).
- `exp_avg_sq`: Non-factored second moments (if applicable).
- `step`: Tracks optimization steps.

---

**`update_step(grads, trainable_variables, learning_rate)`**  
Performs an optimization step using gradients and updates the model parameters.

---

**`get_config()`**  
Returns a dictionary containing the optimizer's configuration, enabling serialization or reconstruction of the optimizer.

---

**Example Usage**:

```python
import tensorflow as tf
from optimizers.adafactor_bv import AdafactorBigVision

# Initialize the AdafactorBigVision optimizer
optimizer = AdafactorBigVision(
    learning_rate=1.0,
    weight_decay=0.01,
    decay_rate=0.8,
    min_dim_size_to_factor=16,
    clipping_threshold=1.0,
    momentum=0.9,
    caution=True,
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# SGDW

**Overview**:

The **SGDW (Stochastic Gradient Descent with Weight Decay)** optimizer is a widely used optimization algorithm for deep learning tasks. It extends the traditional SGD by incorporating weight decay for improved regularization and momentum for accelerated convergence. The optimizer also supports advanced features such as Nesterov momentum, cautious updates, and compatibility with TensorFlow's distributed training mechanisms.

This implementation draws inspiration from existing research and optimizers to provide a flexible and robust solution suitable for modern deep learning workflows.

---

**Features**:

- **Weight Decay**: Decouples weight decay from gradient updates for better regularization.
- **Momentum**: Accelerates convergence by incorporating past gradients.
- **Nesterov Momentum**: Applies Nesterov momentum for improved optimization.
- **Cautious Updates**: Implements updates as per the "Cautious Optimizers" paper to enhance stability.
- **Gradient Clipping**: Supports clipping gradients by norm, value, or global norm.
- **Exponential Moving Average (EMA)**: Maintains a moving average of model weights for smoother updates.
- **Distributed Training**: Compatible with TensorFlow’s distributed training strategies.
- **Customizability**: Offers numerous hyperparameters for fine-tuning.

---

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The learning rate for the optimizer.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 regularization.
- **`momentum`** *(float, default=0.0)*: Momentum factor for optimization.
- **`dampening`** *(float, default=0.0)*: Dampening for momentum.
- **`nesterov`** *(bool, default=False)*: If `True`, enables Nesterov momentum.
- **`caution`** *(bool, default=False)*: If `True`, applies cautious updates.
- **`maximize`** *(bool, default=False)*: If `True`, maximizes the objective instead of minimizing it.
- **`foreach`** *(bool, default=None)*: If `True`, enables multi-tensor updates for improved performance.
- **`differentiable`** *(bool, default=False)*: If `True`, computes gradients in a differentiable manner.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) for model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
- **`name`** *(str, default="sgdw")*: Name of the optimizer.

---

**Methods**:

**`build(var_list)`**
Initializes the optimizer state for the given trainable variables.

- **`var_list`** *(list of variables)*: List of trainable variables.

Initialization:
- **`momentum_buffer`**: Stores momentum values for each variable.
- **`momentum_buffer_list`**: A list-based version of the momentum buffers.

---

**`update_step(grads, trainable_variables, learning_rate)`**
Performs a single optimization step for the provided gradients and variables.

- **`grads`** *(list of gradients)*: Gradients for the trainable variables.
- **`trainable_variables`** *(list of variables)*: Variables to update.
- **`learning_rate`** *(float)*: Learning rate for the current step.

---

**`get_config()`**
Returns a dictionary containing the optimizer’s configuration for serialization or reinitialization.

---

**Functional API**:

**`sgdw(params, grads, momentum_buffer_list, has_sparse_grad, foreach, *, weight_decay, momentum, lr, dampening, nesterov, caution, maximize)`**
A functional implementation of SGDW for advanced use cases.

- **`params`** *(list of variables)*: Trainable variables.
- **`grads`** *(list of gradients)*: Gradients for the variables.
- **`momentum_buffer_list`** *(list)*: List of momentum buffers.
- **`has_sparse_grad`** *(bool)*: Indicates if sparse gradients are present.
- **`foreach`** *(bool)*: Enables multi-tensor updates if `True`.
- Additional parameters include: `weight_decay`, `momentum`, `lr`, `dampening`, `nesterov`, `caution`, and `maximize`.

---

**Example Usage**:

```python
import tensorflow as tf
from optimizers.sgdw import SGDW

# Initialize the SGDW optimizer
optimizer = SGDW(
    learning_rate=1e-3,
    weight_decay=0.01,
    momentum=0.9,
    nesterov=True,
    caution=True,
)

# Compile a model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Lookahead

**Overview**:

The **Lookahead Optimizer** is a wrapper that enhances the performance of an existing optimizer by implementing the "k steps forward, 1 step back" strategy proposed in the paper [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610). This method improves both stability and convergence by periodically aligning the fast-moving weights (inner optimizer) with a slower-moving set of weights (lookahead weights).

This implementation is designed for TensorFlow and integrates seamlessly with the Keras API and the optimizers in this repository, enabling the use of Lookahead with any existing optimizer.

---

**Features**:

- **Optimizer Agnostic**: Can wrap any standard optimizer to improve its stability and convergence.
- **Periodic Synchronization**: Updates the slower weights after a fixed number of steps (`k`), incorporating progress made by the faster optimizer.
- **Improved Generalization**: Helps models generalize better by smoothing the optimization trajectory.
- **Easy Integration**: Fully compatible with TensorFlow and the Keras API.

---

**Parameters**:

- **`base_optimizer`**: The underlying optimizer to be wrapped by Lookahead.
- **`alpha`** *(float, default=0.5)*: The slow update rate for interpolating between the fast and slow weights. Must be in the range `[0, 1]`.
- **`k`** *(int, default=6)*: Number of steps to take with the fast optimizer before synchronizing with the slow weights.
- **`name`** *(str, default="lookahead")*: Name of the optimizer.

---

**Methods**:

**`build(var_list)`**  
Initializes the optimizer state for the trainable variables.

- **`var_list`** *(list of variables)*: List of trainable variables.  
This method initializes the slow weights for each trainable variable.

---

**`update_step(grads, trainable_variables, learning_rate)`**  
Performs an optimization step using the base optimizer and synchronizes with the slow weights every `k` steps.

- **`grads`** *(list of tensors)*: Gradients of the loss with respect to the trainable variables.
- **`trainable_variables`** *(list of variables)*: Variables to be updated.
- **`learning_rate`** *(float)*: Learning rate for the base optimizer.

---

**`apply_gradients(grads_and_vars, tape=None)`**  
Applies gradients to the trainable variables using the base optimizer.

- **`grads_and_vars`** *(list of tuples)*: Pairs of gradients and variables.
- **`tape`** *(tf.GradientTape, optional)*: Gradient tape for recording operations.

---

**`sync_lookahead()`**  
Synchronizes the fast weights with the slow weights.

---

**`get_config()`**  
Returns a dictionary containing the optimizer's configuration for serialization or reinitialization.

---

**Example Usage**:

```python
import tensorflow as tf
from optimizers.lookahead import Lookahead
from optimizers.adamp import AdamP

# Initialize a base optimizer
base_optimizer = AdamP(learning_rate=1e-3)

# Wrap the base optimizer with Lookahead
optimizer = Lookahead(base_optimizer=base_optimizer, alpha=0.5, k=5)

# Compile a model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```
