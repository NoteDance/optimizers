# AdaBelief

**Overview**:

The `AdaBelief` optimizer is a modification of the Adam optimizer designed to adapt the learning rate to the gradient’s variability. This approach makes it particularly effective for handling noisy gradients and improving generalization. It supports advanced features like rectification (inspired by RAdam), weight decay, gradient clipping, and the ability to degenerate into SGD when required.

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

**Parameters**:

- **`base_optimizer`**: The underlying optimizer to be wrapped by Lookahead.
- **`alpha`** *(float, default=0.5)*: The slow update rate for interpolating between the fast and slow weights. Must be in the range `[0, 1]`.
- **`k`** *(int, default=6)*: Number of steps to take with the fast optimizer before synchronizing with the slow weights.
- **`name`** *(str, default="lookahead")*: Name of the optimizer.

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

# AdaBound

**Overview**:

The `AdaBound` optimizer is an adaptive gradient method that dynamically bounds the learning rate for each parameter, stabilizing training and ensuring better generalization. It combines the benefits of Adam's adaptive learning rate with the bounded nature of SGD. The learning rate smoothly transitions from an adaptive method to a final learning rate, providing a balance between fast convergence and generalization.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta_1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta_2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay, used to regularize weights.
- **`final_lr`** *(float, default=0.1)*: The final learning rate towards which the optimizer converges.
- **`gamma`** *(float, default=1e-3)*: Controls the convergence speed of the learning rate towards `final_lr`.
- **`amsbound`** *(bool, default=False)*: Whether to use the AMSBound variant, which enforces upper bounds on the second moment estimates.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients.
- **`name`** *(str, default="adabound")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.adabound import AdaBound

# Instantiate optimizer
optimizer = AdaBound(
    learning_rate=1e-3,
    final_lr=0.1,
    weight_decay=1e-4,
    gamma=1e-3,
    amsbound=True
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AdaBoundW

**Overview**:

The `AdaBoundW` optimizer is an adaptive gradient descent method that builds upon Adam with dynamic learning rate bounds, ensuring convergence to a stable final learning rate. It incorporates weight decay regularization and supports the AMSBound variant for improved optimization.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The initial learning rate for parameter updates.
- **`beta_1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta_2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in division.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay regularization.
- **`final_lr`** *(float, default=0.1)*: The final (stable) learning rate to which the optimizer gradually bounds.
- **`gamma`** *(float, default=1e-3)*: Controls the speed at which the bounds tighten toward the final learning rate.
- **`amsbound`** *(bool, default=False)*: If `True`, uses the AMSBound variant, which applies bounds to the second moment estimate for stability.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm to avoid exploding gradients.
- **`clipvalue`** *(float, optional)*: Clips gradients by their absolute value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients globally by their norm.
- **`use_ema`** *(bool, default=False)*: If `True`, applies Exponential Moving Average (EMA) to model weights during training.
- **`ema_momentum`** *(float, default=0.99)*: Momentum value for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting model weights with their EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scale factor for loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before applying updates.
- **`name`** *(str, default="adaboundw")*: Name of the optimizer instance.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.adaboundw import AdaBoundW

# Instantiate optimizer
optimizer = AdaBoundW(
    learning_rate=1e-3,
    weight_decay=1e-4,
    final_lr=0.01,
    gamma=1e-3,
    amsbound=True
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AdaMod

**Overview**:

The `AdaMod` optimizer is an extension of the Adam optimizer that incorporates an additional third moment term to modulate learning rates. It is designed to stabilize training by applying momental bounds on learning rates, effectively adapting them to recent updates. This helps to maintain consistent updates and prevents excessive changes in parameter values.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta_1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta_2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`beta_3`** *(float, default=0.999)*: Exponential decay rate for the moving average of learning rates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay. Applies regularization to the model weights.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for the Exponential Moving Average.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients before updating parameters.
- **`name`** *(str, default="adamod")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.adamod import AdaMod

# Instantiate optimizer
optimizer = AdaMod(
    learning_rate=1e-3,
    beta_1=0.9,
    beta_2=0.999,
    beta_3=0.999,
    weight_decay=1e-2
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AccSGD

**Overview**:

The `AccSGD` optimizer is a provably accelerated stochastic optimization algorithm designed to improve the convergence rate of deep learning models. It builds upon SGD by introducing momentum-based acceleration techniques as detailed in the [AccSGD paper](https://arxiv.org/pdf/1704.08227.pdf). This method is particularly suitable for large-scale deep learning tasks where faster convergence is desired.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The initial learning rate for parameter updates.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay regularization.
- **`kappa`** *(float, default=1000.0)*: A parameter that controls the large learning rate scaling factor.
- **`xi`** *(float, default=10.0)*: Parameter that influences the balance between acceleration and stability.
- **`smallConst`** *(float, default=0.7)*: A constant that determines the step size and contributes to stabilization during updates.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm to prevent exploding gradients.
- **`clipvalue`** *(float, optional)*: Clips gradients by their absolute value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients globally by their norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum value for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting model weights with their EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scale factor for loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before applying updates.
- **`name`** *(str, default="accsgd")*: Name of the optimizer instance.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.accsgd import AccSGD

# Instantiate optimizer
optimizer = AccSGD(
    learning_rate=1e-3,
    weight_decay=1e-4,
    kappa=500.0,
    xi=5.0,
    smallConst=0.8
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AggMo

**Overview**:

The `AggMo` optimizer (Aggregated Momentum) is a momentum-based optimization method that aggregates multiple momentum terms with varying decay rates (betas). This approach helps to smooth out the optimization process and accelerates convergence by leveraging multiple momentum terms simultaneously. It is particularly useful for training deep learning models where stability and speed are critical.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`betas`** *(tuple of floats, default=(0.0, 0.9, 0.99))*: A tuple of momentum coefficients. Each value represents a different momentum decay rate to aggregate.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay regularization.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm to prevent exploding gradients.
- **`clipvalue`** *(float, optional)*: Clips gradients by their absolute value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients globally by their norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum value for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting model weights with their EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scale factor for loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before applying updates.
- **`name`** *(str, default="aggmo")*: Name of the optimizer instance.

**Alternate Constructor**:

The `AggMo` optimizer also provides a convenient class method, `from_exp_form`, to generate betas using an exponential decay formula:

- **`lr`** *(float, default=1e-3)*: Learning rate for the optimizer.
- **`a`** *(float, default=0.1)*: Base value for exponential decay.
- **`k`** *(int, default=3)*: Number of momentum coefficients.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay.

---

**Example Usage**:

```python
import tensorflow as tf
from optimizers.aggmo import AggMo

# Instantiate optimizer with predefined betas
optimizer = AggMo(
    learning_rate=1e-3,
    betas=(0.0, 0.5, 0.9),
    weight_decay=1e-4
)

# Or create using the exponential form
optimizer = AggMo.from_exp_form(
    lr=1e-3,
    a=0.1,
    k=3,
    weight_decay=1e-4
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Ranger

**Overview**:

The `Ranger` optimizer combines the benefits of RAdam (Rectified Adam) and LookAhead optimizers to improve training stability, convergence, and generalization. It incorporates gradient centralization (GC) for both convolutional and fully connected layers and uses an integrated LookAhead mechanism to smooth updates.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.95)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-5)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay. Applies standard weight decay during updates.
- **`alpha`** *(float, default=0.5)*: LookAhead interpolation factor, determining how much to interpolate between fast and slow weights.
- **`k`** *(int, default=6)*: Number of optimizer steps before LookAhead updates.
- **`N_sma_threshhold`** *(int, default=5)*: Threshold for the simple moving average (SMA) used in RAdam to enable variance rectification.
- **`use_gc`** *(bool, default=True)*: Whether to apply gradient centralization (GC).
- **`gc_conv_only`** *(bool, default=False)*: Whether to apply GC only to convolutional layers (`True`) or to all layers (`False`).
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients.
- **`name`** *(str, default="ranger")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.ranger import Ranger

# Instantiate optimizer
optimizer = Ranger(
    learning_rate=1e-3,
    weight_decay=1e-2,
    alpha=0.5,
    k=6,
    use_gc=True
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# DiffGrad

**Overview**:  

The `DiffGrad` optimizer is a variant of the Adam optimizer designed to adapt the learning rate based on the gradient differences between consecutive updates. This mechanism introduces a differential coefficient, which scales the first moment estimate based on the similarity of the current and previous gradients. It has been proposed in the paper [*diffGrad: An Optimization Method for Convolutional Neural Networks*](https://arxiv.org/abs/1909.11015). DiffGrad aims to improve convergence by adjusting the optimization dynamics based on gradient changes.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The initial step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates (moving average of gradients).
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates (moving average of squared gradients).
- **`epsilon`** *(float, default=1e-8)*: A small constant for numerical stability, especially during division.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay, used for regularization. If set to a non-zero value, adds a decay term to the gradients.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm to prevent exploding gradients.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value to control individual gradient magnitudes.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm, aggregating all gradient norms before clipping.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average (EMA) to model weights for stabilization.
- **`ema_momentum`** *(float, default=0.99)*: The momentum for updating the EMA of model weights.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights with model weights.
- **`loss_scale_factor`** *(float, optional)*: A factor for scaling the loss, useful in mixed precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before applying updates.
- **`name`** *(str, default="diffgrad")*: The name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.diffgrad import DiffGrad

# Instantiate optimizer
optimizer = DiffGrad(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    weight_decay=1e-4
)

# Compile a model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Ranger2020

**Overview**:

The `Ranger` optimizer combines the techniques of RAdam (Rectified Adam) and LookAhead to achieve faster convergence and better generalization. It also optionally incorporates Gradient Centralization (GC), which re-centers the gradient to improve optimization stability.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.95)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-5)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay.
- **`alpha`** *(float, default=0.5)*: Interpolation factor for the LookAhead mechanism.
- **`k`** *(int, default=6)*: Number of update steps before LookAhead interpolates weights.
- **`N_sma_threshhold`** *(int, default=5)*: Threshold for the simple moving average (SMA) in RAdam to apply rectified updates.
- **`use_gc`** *(bool, default=True)*: Whether to apply Gradient Centralization (GC).
- **`gc_conv_only`** *(bool, default=False)*: If `True`, GC is only applied to convolutional layers.
- **`gc_loc`** *(bool, default=True)*: If `True`, GC is applied during the gradient computation step; otherwise, it is applied after.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients.
- **`name`** *(str, default="ranger2020")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.ranger2020 import Ranger

# Instantiate optimizer
optimizer = Ranger(
    learning_rate=1e-3,
    alpha=0.5,
    k=6,
    use_gc=True,
    gc_conv_only=False
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```
