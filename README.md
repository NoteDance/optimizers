# AdaBelief

## Overview
The `AdaBelief` optimizer is a modification of the Adam optimizer designed to adapt the learning rate to the gradient’s variability. This approach makes it particularly effective for handling noisy gradients and improving generalization. It supports advanced features like rectification (inspired by RAdam), weight decay, gradient clipping, and the ability to degenerate into SGD when required.

## Features
- **Adaptive Learning Rate**: Learns from the gradient’s variability for more adaptive parameter updates.
- **Weight Decay**: Supports both decoupled and standard weight decay.
- **Rectified Updates**: Option to use rectification for variance control (RAdam-inspired).
- **AMSGrad**: Optional feature to use the maximum of second-moment estimates for normalization.
- **Gradient Clipping**: Supports multiple forms of gradient clipping (`clipnorm`, `clipvalue`, `global_clipnorm`).
- **Gradient Accumulation**: Enables gradient accumulation across steps for large-batch training.
- **EMA Support**: Supports exponential moving average for model weights.

---

## Class Definition
```python
class AdaBelief(optimizer.Optimizer):
```

### Parameters
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

## Methods

### `__setstate__(state)`
Restores optimizer state and ensures `amsgrad` is set to `False`.

### `reset()`
Initializes or resets optimizer variables (`exp_avg`, `exp_avg_var`, etc.).

### `build(var_list)`
Creates necessary optimizer state variables (`exp_avg`, `exp_avg_var`, etc.) for the given trainable variables.

### `update_step(gradient, variable, learning_rate)`
Performs a single optimization step for the given variable and gradient.

- **`gradient`**: The gradient tensor.
- **`variable`**: The trainable variable to update.
- **`learning_rate`**: The learning rate for the current step.

### `get_config()`
Returns the configuration of the optimizer as a dictionary.

---

## Example Usage
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

## Overview

The `AdamP` optimizer is a modification of the Adam optimizer that aims to slow down the increase of weight norms in momentum-based optimizers. This is particularly useful for improving generalization and preventing overfitting. The optimizer uses a projection step to decouple sharp and flat components of the gradients, effectively reducing sensitivity to noise.

### Features
- **Projection Mechanism**: Decouples flat and sharp components of the gradients to mitigate overfitting.
- **Weight Decay**: Includes an adjusted weight decay term with a configurable ratio.
- **Momentum Updates**: Supports standard and Nesterov momentum.
- **Gradient Clipping**: Compatible with multiple gradient clipping strategies (`clipnorm`, `clipvalue`, `global_clipnorm`).
- **Gradient Accumulation**: Allows for gradient accumulation over multiple steps.
- **EMA Support**: Provides exponential moving average functionality for model weights.

---

## Class Definition
```python
class AdamP(optimizer.Optimizer):
```

---

## Parameters
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

## Methods

### `build(var_list)`
Initializes optimizer state variables, such as first and second moment estimates (`exp_avg` and `exp_avg_sq`).

- **`var_list`** *(list of variables)*: List of trainable variables.

---

### `update_step(gradient, variable, learning_rate)`
Performs a single optimization step for a given gradient and variable.

- **`gradient`** *(Tensor)*: The gradient tensor for the variable.
- **`variable`** *(Tensor)*: The trainable variable to update.
- **`learning_rate`** *(float)*: Learning rate for the current step.

---

### `get_config()`
Returns the optimizer configuration as a dictionary, including all hyperparameters.

---

## Example Usage
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

## Overview

The `LaProp` optimizer is an adaptive gradient optimization algorithm that improves upon Adam by dynamically adjusting learning rates in proportion to the gradients. It includes optional features like centered second moments, AMSGrad stabilization, and weight decay, making it a versatile optimizer for deep learning tasks.

### Features
- **Learning Rate Adaptation**: Adjusts learning rates dynamically using gradient-based corrections.
- **Centered Updates**: Optionally centers second-moment estimates to improve convergence.
- **AMSGrad**: Stabilizes training by maintaining the maximum second-moment running average.
- **Weight Decay**: Regularizes weights during optimization.
- **Gradient Clipping**: Compatible with multiple gradient clipping techniques.
- **Gradient Accumulation**: Supports accumulation for distributed or large-batch training.

---

## Class Definition
```python
class LaProp(optimizer.Optimizer):
```

---

## Parameters
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

## Methods

### `build(var_list)`
Initializes state variables for the optimizer, including:
- First and second moment estimates (`exp_avg`, `exp_avg_sq`).
- Learning rate-based adjustments for first and second moments.
- Optional variables for AMSGrad and centered updates.

- **`var_list`** *(list of variables)*: List of trainable variables to optimize.

---

### `update_step(gradient, variable, learning_rate)`
Performs a single optimization step for a variable using its gradient.

---

### `get_config()`
Returns the optimizer configuration as a dictionary, including all hyperparameters for easy reinitialization.

---

## Example Usage
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

## Overview

The `Lars` optimizer is an implementation of **Layer-wise Adaptive Rate Scaling (LARS)**, a variant of stochastic gradient descent (SGD) designed for large-batch training. It combines weight decay and trust region-based learning rate adaptation, ensuring effective scaling for deep learning models with high-dimensional parameters. This implementation also includes optional **LARC (Layer-wise Adaptive Rate Clipping)**, momentum, and Nesterov updates.

### Features
- **Layer-wise Learning Rate Scaling**: Adapts learning rates based on parameter and gradient norms.
- **LARC Clipping**: Ensures trust ratio stays within a bounded range.
- **Momentum and Nesterov Updates**: Enhances optimization convergence speed.
- **Weight Decay**: Applies regularization to prevent overfitting.
- **Gradient Clipping**: Compatible with multiple gradient clipping techniques.
- **Gradient Accumulation**: Supports large-batch training via accumulation.

---

## Class Definition
```python
class Lars(optimizer.Optimizer):
```

---

## Parameters
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

## Methods

### `build(var_list)`
Initializes the optimizer states for all variables. Each variable gets a state dictionary to store momentum buffers if `momentum` is enabled.

- **`var_list`** *(list of variables)*: List of trainable variables to optimize.

---

### `update_step(gradient, variable, learning_rate)`
Performs a single optimization step for a variable using its gradient.

---

### `get_config()`
Returns the optimizer configuration as a dictionary, including all hyperparameters for easy reinitialization.

---

## Example Usage
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

## Overview

The `MADGRAD` optimizer is an advanced optimization algorithm designed for large-scale machine learning tasks. It is based on the paper [MADGRAD: Stochastic Optimization with Momentum Decay for Training Neural Networks](https://arxiv.org/abs/2101.11075) and provides benefits for sparse and dense gradient updates. This implementation is compatible with TensorFlow and includes support for advanced features like weight decay, momentum, and gradient accumulation.

### Features
- **Momentum Acceleration**: Enhances optimization convergence speed.
- **Second-Moment Accumulation**: Accumulates second-order gradient statistics for stable updates.
- **Decoupled Weight Decay**: Separates weight decay from gradient updates for improved performance.
- **Sparse Tensor Support**: Efficiently handles sparse gradients.
- **Gradient Clipping**: Compatible with multiple gradient clipping techniques.
- **Gradient Accumulation**: Supports large-batch training via accumulation.

---

## Class Definition
```python
class MADGRAD(optimizer.Optimizer):
```

---

## Parameters
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

## Methods

### `build(var_list)`
Initializes the optimizer states for all variables, creating:
1. `_grad_sum_sq`: Accumulator for second-order gradient statistics.
2. `_s`: Accumulator for gradients.
3. `_x0`: Initial variable values for momentum-based updates (if applicable).

- **`var_list`** *(list of variables)*: List of trainable variables to optimize.

---

### `update_step(gradient, variable, learning_rate)`
Performs a single optimization step for a variable using its gradient.

---

### `get_config()`
Returns the optimizer configuration as a dictionary, including all hyperparameters for easy reinitialization.

---

## Example Usage
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

## Overview

The **MARS** optimizer implements a novel optimization algorithm designed for training large-scale models effectively. It leverages variance reduction techniques, adaptive learning rates, and supports both AdamW-style and Lion-style updates for parameter optimization. MARS also incorporates specialized mechanisms to handle 1D and 2D gradients differently, ensuring efficiency and accuracy in various scenarios.

This implementation is based on the paper [MARS: Unleashing the Power of Variance Reduction for Training Large Models](https://arxiv.org/abs/2411.10438).

---

## Features
- **Variance Reduction**: Minimizes gradient variance to stabilize updates.
- **Customizable Update Rules**: Supports AdamW and Lion-style parameter updates.
- **Adaptive Gradient Clipping**: Provides enhanced control over gradient updates.
- **1D/2D Optimization**: Separate strategies for optimizing 1D and 2D parameters.
- **Weight Decay**: Integrated weight decay for L2 regularization.
- **Caution Mechanism**: Ensures robust updates by masking inconsistent gradient directions.
- **Gradient Accumulation Support**: Compatible with large-batch training.

---

## Class Definition

```python
class Mars(optimizer.Optimizer)
```

---

## Parameters

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

## Methods

### `build(var_list)`
Initializes the optimizer's state for the given variables, including:
- `_exp_avg`: First-moment estimates for each variable.
- `_exp_avg_sq`: Second-moment estimates for each variable.
- `_last_grad`: Stores the last gradient for variance reduction.
- `step`: Tracks the optimization step for each variable.

- **`var_list`** *(list of variables)*: List of trainable variables to optimize.

---

### `update_step(gradient, variable, learning_rate)`
Performs a single optimization step for the given variable using its gradient.

---

### `get_config()`
Returns the optimizer configuration as a dictionary for easy reinitialization.

---

## Example Usage

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

## Overview

The **NAdam** optimizer is an implementation of the Nesterov-accelerated Adaptive Moment Estimation (Nadam) algorithm. Nadam extends the widely-used Adam optimizer by incorporating Nesterov momentum, providing faster convergence in some scenarios. This optimizer is particularly useful for tasks where momentum plays a critical role in overcoming saddle points and improving optimization dynamics.

The algorithm is described in:
- **"Incorporating Nesterov Momentum into Adam"** ([PDF link](http://cs229.stanford.edu/proj2015/054_report.pdf))
- **"On the Importance of Initialization and Momentum in Deep Learning"** ([PDF link](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf))

---

## Features

- **Adaptive Learning Rates**: Adjusts learning rates based on gradients and their variance.
- **Nesterov Momentum**: Combines momentum-based updates with gradient adjustments, offering better convergence in some cases.
- **Bias Correction**: Corrects bias introduced during the initialization of moment estimates.
- **Weight Decay Support**: Includes L2 regularization for better generalization.
- **Warm Momentum Schedule**: Gradually adjusts momentum during training for smoother updates.

---

## Class Definition

```python
class NAdam(optimizer.Optimizer)
```

---

## Parameters

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

## Methods

### `build(var_list)`
Initializes optimizer state for the trainable variables.

- **`var_list`** *(list of variables)*: Trainable variables to optimize.
  
Initializations:
- `_exp_avg`: First moment (momentum) estimates.
- `_exp_avg_sq`: Second moment (variance) estimates.
- `_m_schedule`: Tracks the momentum schedule for warm restarts.
- `step`: Tracks the number of optimization steps for each variable.

---

### `update_step(gradient, variable, learning_rate)`
Performs a single optimization step for the given variable using the provided gradient.

---

### `get_config()`
Returns the optimizer's configuration as a dictionary for serialization and reinitialization.

---

## Example Usage

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

## Overview

The **NvNovoGrad** optimizer is an implementation of NovoGrad, an optimization algorithm designed for deep learning that uses layer-wise adaptive moments for efficient and robust training. NovoGrad is particularly effective for large-scale and resource-constrained deep learning tasks, as it combines the benefits of Adam and L2 regularization while being computationally efficient.

The algorithm is described in:
- **"Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks"** ([arXiv link](https://arxiv.org/abs/1905.11286))

This implementation is inspired by NVIDIA's original implementation in PyTorch, used in speech recognition models like **Jasper**:
- [Jasper Example](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper)

---

## Features

- **Layer-wise Adaptive Learning Rates**: Adjusts learning rates for each layer, making it suitable for large-scale tasks.
- **Efficient Memory Usage**: Optimized for memory efficiency, making it ideal for resource-constrained environments.
- **Support for AMSGrad**: Includes an optional variant of NovoGrad with AMSGrad for improved convergence.
- **Gradient Averaging**: Option to average gradients for smoother updates.
- **Weight Decay**: Includes support for L2 regularization.

---

## Class Definition

```python
class NvNovoGrad(optimizer.Optimizer)
```

---

## Parameters

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

## Methods

### `build(var_list)`
Initializes optimizer state for the trainable variables.

- **`var_list`** *(list of variables)*: Trainable variables to optimize.
  
Initializations:
- `_exp_avg`: Stores first-moment estimates for each variable.
- `_exp_avg_sq`: Stores second-moment estimates (squared gradients).
- `_max_exp_avg_sq`: (Optional) Stores maximum second-moment estimates for AMSGrad.
- `step`: Tracks the number of optimization steps for each variable.

---

### `update_step(gradient, variable, learning_rate)`
Performs a single optimization step for the given variable using the provided gradient.

---

### `get_config()`
Returns the optimizer's configuration as a dictionary for serialization and reinitialization.

---

## Example Usage

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

## Overview

The **RAdam (Rectified Adam)** optimizer is a variant of the Adam optimizer that incorporates a mechanism to rectify the variance of adaptive learning rates. This rectification improves stability and prevents early training instabilities, especially in the initial training phase. RAdam maintains the benefits of Adam while being more robust for a wide range of applications.

The algorithm is described in the paper:
- **"On the Variance of the Adaptive Learning Rate and Beyond"** ([arXiv link](https://arxiv.org/abs/1908.03265))

This implementation is inspired by the original PyTorch implementation:
- [RAdam GitHub Repository](https://github.com/LiyuanLucasLiu/RAdam)

---

## Features

- **Variance Rectification**: Dynamically adjusts learning rates to mitigate training instabilities during the early stages.
- **Automatic Transition**: Automatically transitions between SGD-like behavior and Adam-like behavior based on available statistics.
- **Weight Decay Support**: Includes support for L2 regularization.
- **Efficient Buffering**: Caches intermediate calculations for computational efficiency.

---

## Class Definition

```python
class RAdam(optimizer.Optimizer)
```

---

## Parameters

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

## Methods

### `build(var_list)`
Initializes the optimizer state for the trainable variables.

- **`var_list`** *(list of variables)*: List of trainable variables to optimize.

#### State Initialization:
- `_exp_avg`: First-moment estimates for each variable.
- `_exp_avg_sq`: Second-moment estimates for each variable.
- `step`: Tracks the optimization steps for each variable.
- `buffer`: Stores intermediate calculations for variance rectification.

---

### `update_step(gradient, variable, learning_rate)`
Performs a single optimization step for the given variable using the provided gradient.

---

### `get_config()`
Returns a dictionary containing the optimizer's configuration, suitable for serialization or reinitialization.

---

## Example Usage

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

## Overview

The **SGDP (Stochastic Gradient Descent with Projection and Weight Decay)** optimizer is a variant of SGD that incorporates **decoupled weight decay regularization** and **gradient projection**. These features help control weight norm growth during training, improving convergence and performance. 

This algorithm is described in the paper:
- **"Slowing Down the Weight Norm Increase in Momentum-based Optimizers"** ([arXiv link](https://arxiv.org/abs/2006.08217)).

The implementation is inspired by the official repository:
- [AdamP GitHub Repository](https://github.com/clovaai/AdamP)

---

## Features

- **Projection Mechanism**: Projects gradients onto a subspace to reduce ineffective gradient directions and improve optimization efficiency.
- **Decoupled Weight Decay**: Effectively regularizes the model parameters without interfering with gradient updates.
- **Momentum and Nesterov Support**: Enhances optimization with momentum or Nesterov acceleration.
- **Flexibility**: Offers a range of hyperparameters for customization.

---

## Class Definition

```python
class SGDP(optimizer.Optimizer)
```

---

## Parameters

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

## Methods

### `build(var_list)`
Initializes the optimizer state for the trainable variables.

- **`var_list`** *(list of variables)*: List of trainable variables.

#### State Initialization:
- `_momentum`: Stores momentum terms for each variable.

---

### `update_step(gradient, variable, learning_rate)`
Performs a single optimization step for the given variable using the provided gradient.

---

### `get_config()`
Returns a dictionary containing the optimizer's configuration, suitable for serialization or reinitialization.

---

## Example Usage

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

## Overview

The **Adan (Adaptive Nesterov Momentum)** optimizer is a next-generation optimization algorithm designed to accelerate training and improve convergence in deep learning models. It combines **adaptive gradient estimation** and **multi-step momentum** for enhanced performance.

This algorithm is introduced in the paper:
- **"Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"** ([arXiv link](https://arxiv.org/abs/2208.06677)).

The implementation is inspired by the official repository:
- [Adan GitHub Repository](https://github.com/sail-sg/Adan)

---

## Features

- **Adaptive Nesterov Momentum**: Improves gradient estimation using adaptive updates.
- **Gradient Difference Momentum**: Stabilizes updates by tracking gradient differences.
- **Decoupled Weight Decay**: Supports weight decay without interfering with gradient updates.
- **Multi-Tensor Operations**: Optimized for efficient parallel computations.
- **Bias Correction**: Ensures unbiased gradient estimates.

---

## Class Definition

```python
class Adan(optimizer.Optimizer)
```

---

## Parameters

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

## Methods

### `build(var_list)`
Initializes the optimizer state for the trainable variables.

- **`var_list`** *(list of variables)*: List of trainable variables.

#### State Initialization:
- `exp_avg`: First moment estimates for each variable.
- `exp_avg_sq`: Second moment estimates for each variable.
- `exp_avg_diff`: Gradient difference momentum for each variable.
- `neg_pre_grad`: Stores the previous negative gradient for updates.

---

### `update_step(gradient, variable, learning_rate)`
Performs a single optimization step for the given variable using the provided gradient.

---

### `get_config()`
Returns a dictionary containing the optimizer's configuration, suitable for serialization or reinitialization.

---

## Example Usage

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
