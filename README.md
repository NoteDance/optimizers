# AdaBelief

**Overview**:

The `AdaBelief` optimizer is a modification of the Adam optimizer designed to adapt the learning rate to the gradient’s variability. This approach makes it particularly effective for handling noisy gradients and improving generalization. It supports advanced features like rectification (inspired by RAdam), weight decay, gradient clipping, and the ability to degenerate into SGD when required.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
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
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
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
- **`beta1`** *(float, default=0.9)*: Coefficient for the moving average of the first moment (mean of gradients).
- **`beta2`** *(float, default=0.999)*: Coefficient for the moving average of the second moment (variance of gradients).
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
    beta1=0.9,
    beta2=0.999,
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
- **`beta1`** *(float, default=0.9)*: Coefficient for the first moment estimate.
- **`beta2`** *(float, default=0.99)*: Coefficient for the second moment estimate.
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
    beta1=0.9,
    beta2=0.99,
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
- **`beta1`** *(float, default=0.9)*: Coefficient for the first moment estimate (momentum term).
- **`beta2`** *(float, default=0.999)*: Coefficient for the second moment estimate (variance term).
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
    beta1=0.9,
    beta2=0.999,
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
- **`beta1`** *(float, default=0.95)*: Exponential decay rate for the first moment estimate (momentum term).
- **`beta2`** *(float, default=0.98)*: Exponential decay rate for the second moment estimate (variance term).
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
    beta1=0.95,
    beta2=0.98,
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
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimate (momentum term).
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimate (variance term).
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
    beta1=0.9,
    beta2=0.999,
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
- **`beta1`** *(float, default=0.98)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.92)*: Exponential decay rate for gradient difference momentum.
- **`beta3`** *(float, default=0.99)*: Exponential decay rate for the second moment estimates.
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
    beta1=0.98,
    beta2=0.92,
    beta3=0.99,
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
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
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
    beta1=0.9,
    beta2=0.999,
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
- **`beta1`** *(float)*: Exponential decay rate for the first moment estimates (default: `0.9`).
- **`beta2`** *(float)*: Exponential decay rate for the Hessian diagonal squared estimates (default: `0.999`).
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
    beta1=0.9, 
    beta2=0.999, 
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

The **ADOPT (Adaptive Optimization with Trust)** optimizer is a novel variant of Adam designed to achieve optimal convergence rates with any value of \(\beta2\). It introduces enhancements such as adaptive gradient scaling and cautious updates, making it suitable for diverse optimization scenarios, including tasks requiring stability and robustness in gradient updates.  

This TensorFlow implementation is adapted from the PyTorch version available in the [timm library](https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/adopt.py). The optimizer builds on concepts from Adam while adding innovative features for enhanced convergence and generalization.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: Learning rate for the optimizer.  
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.  
- **`beta2`** *(float, default=0.9999)*: Exponential decay rate for the second moment estimates.  
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
    beta1=0.9,
    beta2=0.9999,
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
- **`beta1`** *(float, default=0.9)*: Coefficient for computing running averages of gradient moments.
- **`beta2`** *(float, default=0.999)*: Coefficient for computing running averages of squared gradients.
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
    beta1=0.9,
    beta2=0.999,
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
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
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
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
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
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`beta3`** *(float, default=0.999)*: Exponential decay rate for the moving average of learning rates.
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
    beta1=0.9,
    beta2=0.999,
    beta3=0.999,
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

# RangerVA

**Overview**:

The `RangerVA` optimizer is a hybrid optimizer that combines the techniques of Rectified Adam (RAdam), Lookahead optimization, and gradient transformations, making it suitable for modern deep learning tasks. It also includes support for custom gradient transformations and adaptive learning rate calibration, making it highly flexible and efficient for various scenarios.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.95)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-5)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay.
- **`alpha`** *(float, default=0.5)*: Lookahead interpolation factor controlling the update between fast and slow weights.
- **`k`** *(int, default=6)*: Number of steps before Lookahead updates are applied.
- **`n_sma_threshhold`** *(int, default=5)*: Threshold for the number of simple moving averages in RAdam's variance rectification mechanism.
- **`amsgrad`** *(bool, default=True)*: Whether to use the AMSGrad variant.
- **`transformer`** *(str, default='softplus')*: Specifies the transformation function applied to the adaptive learning rate (e.g., `'softplus'` for smooth adaptation).
- **`smooth`** *(float, default=50)*: Smoothing factor for the Softplus transformation function.
- **`grad_transformer`** *(str, default='square')*: Specifies the transformation applied to gradients (e.g., `'square'` or `'abs'`).
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients.
- **`name`** *(str, default="rangerva")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.rangerva import RangerVA

# Instantiate optimizer
optimizer = RangerVA(
    learning_rate=1e-3,
    alpha=0.5,
    k=6,
    weight_decay=1e-2,
    transformer='softplus',
    smooth=50,
    grad_transformer='square'
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# PID

**Overview**:

The `PID` optimizer is inspired by Proportional-Integral-Derivative (PID) control theory and implements stochastic gradient descent (SGD) with momentum and optional Nesterov momentum. It introduces integral (`I`) and derivative (`D`) components to enhance learning dynamics and provide better control of gradient updates.

**Parameters**:

- **`learning_rate`** *(float)*: The step size for parameter updates.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay, used for regularization.
- **`momentum`** *(float, default=0)*: Momentum factor to accelerate gradient updates.
- **`dampening`** *(float, default=0)*: Dampening for momentum to reduce its influence on updates.
- **`nesterov`** *(bool, default=False)*: Enables Nesterov momentum for accelerated convergence.
- **`I`** *(float, default=5.0)*: Integral factor for accumulating gradients over time to improve stability.
- **`D`** *(float, default=10.0)*: Derivative factor for incorporating the rate of gradient change.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients before applying updates.
- **`name`** *(str, default="pid")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.pid import PID

# Instantiate optimizer
optimizer = PID(
    learning_rate=1e-3,
    weight_decay=1e-4,
    momentum=0.9,
    I=5.0,
    D=10.0,
    nesterov=True
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# QHAdam

**Overview**:

The `QHAdam` optimizer is an extension of the Adam optimizer designed to improve adaptability in non-convex optimization scenarios. It incorporates additional parameters, `nu1` and `nu2`, to allow fine-grained control over the combination of gradient and root mean square (RMS) terms. This optimizer is particularly effective in handling challenging optimization landscapes.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. Regularizes model parameters.
- **`Nus2`** *(tuple of floats, default=(1.0, 1.0))*: Controls the mixing of the first moment (`nu1`) and second moment (`nu2`) estimates.
- **`decouple_weight_decay`** *(bool, default=False)*: If `True`, applies decoupled weight decay as described in AdamW.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients before applying updates.
- **`name`** *(str, default="qhadam")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.qhadam import QHAdam

# Instantiate optimizer
optimizer = QHAdam(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    Nus2=(0.7, 1.0),
    weight_decay=1e-4,
    decouple_weight_decay=True
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# RangerQH

**Overview**:

The `RangerQH` optimizer combines the QHAdam optimization algorithm with the Lookahead mechanism. QHAdam, introduced by Ma and Yarats (2019), balances the contributions of gradients and gradient variances with tunable parameters `nu1` and `nu2`. The Lookahead mechanism, developed by Hinton and Zhang, enhances convergence by interpolating between "fast" and "slow" weights over multiple steps. This combination provides a powerful optimization approach for deep learning tasks, offering improved convergence and generalization.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. Supports both standard and decoupled decay.
- **`nus`** *(tuple, default=(0.7, 1.0))*: The `nu1` and `nu2` parameters controlling the blending of gradient and variance components.
- **`k`** *(int, default=6)*: Number of optimization steps before updating "slow" weights in the Lookahead mechanism.
- **`alpha`** *(float, default=0.5)*: Interpolation factor for Lookahead updates.
- **`decouple_weight_decay`** *(bool, default=False)*: Enables decoupled weight decay as described in AdamW.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients.
- **`name`** *(str, default="rangerqh")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.ranger_qh import RangerQH

# Instantiate optimizer
optimizer = RangerQH(
    learning_rate=1e-3,
    nus=(0.8, 1.0),
    k=5,
    alpha=0.6,
    weight_decay=1e-2,
    decouple_weight_decay=True
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# QHM

**Overview**:

The `QHM` optimizer implements the quasi-hyperbolic momentum (QHM) optimization algorithm. QHM blends the benefits of momentum-based updates with those of standard stochastic gradient descent (SGD) by introducing a hyperparameter `nu` that balances the contribution of the momentum term with the immediate gradient. This approach offers more flexible control over the optimization dynamics, which can lead to improved convergence and generalization in deep learning models.

---

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. Depending on the value of `weight_decay_type`, weight decay is applied either by adding a decay term to the gradient (`"grad"`) or by directly scaling the model parameters (`"direct"`).
- **`momentum`** *(float, default=0.0)*: Momentum factor used to accumulate past gradients.
- **`nu`** *(float, default=0.7)*: A hyperparameter that controls the balance between the momentum term and the immediate gradient update.
- **`weight_decay_type`** *(str, default="grad")*: Specifies how weight decay is applied. Use `"grad"` to add weight decay to the gradient or `"direct"` to directly scale the parameters.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum factor for the EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for updating the EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients before applying updates.
- **`name`** *(str, default="qhm")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.qhm import QHM

# Instantiate optimizer
optimizer = QHM(
    learning_rate=1e-3,
    weight_decay=1e-2,
    momentum=0.9,
    nu=0.7,
    weight_decay_type="grad"
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Shampoo

**Overview**:

The `Shampoo` optimizer implements the Shampoo optimization algorithm, which leverages second-order information by maintaining and updating preconditioners for each tensor dimension. By computing and applying matrix powers (inverses) of these preconditioners, Shampoo effectively preconditions gradients to accelerate convergence, especially in large-scale deep learning tasks. This approach was introduced in [*Shampoo: Preconditioned Stochastic Tensor Optimization*](https://arxiv.org/abs/1802.09568).

---

**Parameters**:

- **`learning_rate`** *(float, default=1e-1)*: The step size for parameter updates.
- **`epsilon`** *(float, default=1e-4)*: A small constant added for numerical stability when computing the preconditioners.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay regularization.
- **`momentum`** *(float, default=0.0)*: Momentum factor used to smooth gradient updates by accumulating past gradients.
- **`update_freq`** *(int, default=1)*: Frequency (in steps) at which to update the inverse preconditioners.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum factor for the EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients before applying updates.
- **`name`** *(str, default="shampoo")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.shampoo import Shampoo

# Instantiate optimizer
optimizer = Shampoo(
    learning_rate=1e-1,
    epsilon=1e-4,
    weight_decay=1e-4,
    momentum=0.9,
    update_freq=1
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# SWATS

**Overview**:

The `SWATS` optimizer implements the SWATS algorithm, which dynamically switches from an adaptive method (Adam) to SGD during training. By initially leveraging the fast convergence of Adam and later transitioning to SGD, SWATS aims to improve generalization performance. It maintains first and second moment estimates (with optional AMSGrad and Nesterov momentum) and monitors a non-orthogonal scaling criterion to determine the appropriate moment to switch phases.

---

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-3)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay regularization.
- **`amsgrad`** *(bool, default=False)*: Whether to use the AMSGrad variant.
- **`nesterov`** *(bool, default=False)*: Whether to apply Nesterov momentum.
- **`phase`** *(str, default="ADAM")*: Indicates the current phase of optimization. The optimizer starts in the "ADAM" phase and switches to "SGD" once the scaling criterion is met.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients before applying updates.
- **`name`** *(str, default="swats")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.swats import SWATS

# Instantiate optimizer
optimizer = SWATS(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-3,
    weight_decay=1e-4,
    amsgrad=False,
    nesterov=False,
    phase="ADAM"
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Yogi

**Overview**:

The `Yogi` optimizer is an adaptive gradient method that modifies the Adam update rule to control the growth of the second moment estimate. By using an update based on the sign of the difference between the current second moment and the square of the gradient, Yogi aims to adjust the effective learning rate in a more controlled fashion, thereby addressing some of Adam’s convergence issues—especially in nonconvex settings. This approach makes Yogi particularly effective for tasks where careful regulation of adaptivity leads to improved generalization and stable training.

---

**Parameters**:

- **`learning_rate`** *(float, default=1e-2)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment (mean) estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment (variance) estimates.
- **`epsilon`** *(float, default=1e-3)*: Small constant for numerical stability to prevent division by zero.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay regularization; if non-zero, adds a decay term to the gradient.
- **`initial_accumulator`** *(float, default=1e-6)*: Initial value used to fill the first and second moment accumulators.
- **`clipnorm`** *(float, optional)*: Maximum norm for clipping gradients.
- **`clipvalue`** *(float, optional)*: Maximum absolute value for clipping gradients.
- **`global_clipnorm`** *(float, optional)*: Maximum norm for global gradient clipping.
- **`use_ema`** *(bool, default=False)*: Whether to use an Exponential Moving Average (EMA) of the model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum used in the EMA computation.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency at which the EMA weights are overwritten.
- **`loss_scale_factor`** *(float, optional)*: Factor to scale the loss during gradient computation (useful in mixed precision training).
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before applying an update.
- **`name`** *(str, default="yogi")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.yogi import Yogi

# Instantiate the optimizer
optimizer = Yogi(
    learning_rate=1e-2,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-3,
    weight_decay=0,
    initial_accumulator=1e-6
)

# Compile a Keras model
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Adai

**Overview**:

The `Adai` optimizer implements the Adaptive Inertia Estimation (Adai) algorithm, proposed in the ICML 2022 Oral paper [Adaptive Inertia: Disentangling the Effects of Adaptive Learning Rate and Momentum](https://arxiv.org/abs/2006.15815). Adai disentangles the contributions of adaptive learning rate and momentum by dynamically estimating an inertia term for each parameter. This controlled adjustment of momentum improves convergence and generalization, especially in settings with noisy gradients.

---

**Parameters**:

- **`learning_rate`** *(float)*: The step size for parameter updates.
- **`beta0`** *(float, default=0.1)*: Scaling factor that modulates the adaptive momentum coefficient.
- **`beta2`** *(float, default=0.99)*: Exponential decay rate for the second moment (variance) estimates.
- **`epsilon`** *(float, default=1e-3)*: Small constant added for numerical stability to prevent division by zero.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay regularization. When nonzero, weight decay is applied either additively to the gradients or directly to the parameters depending on the `decoupled` flag.
- **`decoupled`** *(bool, default=False)*: Determines the application of weight decay. If `True`, weight decay is applied in a decoupled manner (i.e., directly scaling the parameters); otherwise, it is added to the gradients.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to the model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum factor for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor to scale the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before applying an update.
- **`name`** *(str, default="adai")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.adai import Adai

# Instantiate the optimizer
optimizer = Adai(
    learning_rate=1e-3,
    beta0=0.1,
    beta2=0.99,
    epsilon=1e-3,
    weight_decay=1e-4,
    decoupled=True
)

# Compile a Keras model
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AdaiV2

**Overview**:

The `AdaiV2` optimizer is a generalized variant of the Adai algorithm based on the work presented in [Adaptive Inertia: Disentangling the Effects of Adaptive Learning Rate and Momentum](https://arxiv.org/abs/2006.15815). AdaiV2 extends the original Adai approach by incorporating a dampening parameter to further modulate the adaptive momentum update. By dynamically estimating inertia via adaptive per-parameter momentum (through a generalized clipping mechanism) and coupling it with controlled weight decay (either coupled or decoupled), AdaiV2 aims to improve convergence and generalization—especially in settings with noisy gradients.

---

**Parameters**:

- **`learning_rate`** *(float)*: The step size used for updating parameters.
- **`beta0`** *(float, default=0.1)*: A scaling parameter that influences the computation of the adaptive momentum coefficient.
- **`beta2`** *(float, default=0.99)*: Exponential decay rate for the second moment (variance) estimates.
- **`epsilon`** *(float, default=1e-3)*: A small constant for numerical stability to prevent division by zero.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay regularization. When non-zero, weight decay is applied either additively to the gradients (if `decoupled` is False) or directly to the parameters (if `decoupled` is True).
- **`dampening`** *(float, default=1.0)*: Controls the nonlinearity in the adaptive momentum update. This parameter adjusts the effective inertia by modulating the computed momentum coefficient.
- **`decoupled`** *(bool, default=False)*: Determines how weight decay is applied. If `True`, weight decay is decoupled (applied directly to the parameters); otherwise, it is coupled by adding the decay term to the gradients.
- **`clipnorm`** *(float, optional)*: Maximum norm for clipping gradients.
- **`clipvalue`** *(float, optional)*: Maximum absolute value for clipping gradients.
- **`global_clipnorm`** *(float, optional)*: Maximum norm for clipping gradients globally.
- **`use_ema`** *(bool, default=False)*: Whether to maintain an Exponential Moving Average (EMA) of the model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum factor used for the EMA of the model weights.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency at which the EMA weights are overwritten.
- **`loss_scale_factor`** *(float, optional)*: A factor to scale the loss during gradient computation (useful in mixed precision training).
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before performing an update.
- **`name`** *(str, default="adaiv2")*: The name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.adaiv2 import AdaiV2

# Instantiate the AdaiV2 optimizer
optimizer = AdaiV2(
    learning_rate=1e-3,
    beta0=0.1,
    beta2=0.99,
    epsilon=1e-3,
    weight_decay=1e-4,
    dampening=1.0,
    decoupled=True
)

# Compile a model using the optimizer
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Kron

**Overview**:

The `Kron` optimizer implements the PSGD Kron algorithm, which uses Kronecker-based preconditioning to accelerate stochastic gradient descent. By maintaining a set of per-parameter preconditioners (built via Kronecker products) and updating them probabilistically during training, Kron adapts the effective gradient direction and scaling. This method is particularly useful for large models where efficient preconditioning can significantly improve convergence while managing memory consumption.

---

**Parameters**:

- **`learning_rate`** *(float, default=0.0003)*: The base step size for updating model parameters.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay regularization. When non-zero, weight decay is applied either additively to the gradients or directly to the parameters based on the `decoupled` flag.
- **`b1`** *(float, default=0.9)*: Exponential decay rate used in updating the momentum buffer.
- **`preconditioner_update_probability`** *(callable or float, optional)*: The probability schedule controlling how frequently the preconditioner is updated. If not provided, a default schedule (flat start for 500 steps then exponential annealing) is used.
- **`max_size_triangular`** *(int, default=8192)*: Maximum size for using a full (triangular) preconditioner; dimensions larger than this use a diagonal approximation.
- **`min_ndim_triangular`** *(int, default=2)*: Minimum number of dimensions required for a tensor to receive a triangular (non-diagonal) preconditioner.
- **`memory_save_mode`** *(str, optional)*: Option to control memory usage for preconditioners. Options include `None`, `"smart_one_diag"`, `"one_diag"`, and `"all_diag"`.
- **`momentum_into_precond_update`** *(bool, default=True)*: Determines whether the momentum buffer (updated with decay `b1`) is used when updating the preconditioner.
- **`precond_lr`** *(float, default=0.1)*: Learning rate specifically used for preconditioner updates.
- **`precond_init_scale`** *(float, default=1.0)*: Initial scaling factor for the preconditioners.
- **`mu_dtype`** *(dtype, optional)*: Data type for the momentum buffer; if specified, momentum values are cast to this type.
- **`precond_dtype`** *(dtype, default=tf.float32)*: Data type for the preconditioners and related computations.
- **`clipnorm`** *(float, optional)*: If set, gradients are clipped to this maximum norm.
- **`clipvalue`** *(float, optional)*: If set, gradients are clipped element-wise to this maximum absolute value.
- **`global_clipnorm`** *(float, optional)*: If set, the global norm of all gradients is clipped to this value.
- **`use_ema`** *(bool, default=False)*: Whether to use an Exponential Moving Average (EMA) of the model weights during training.
- **`ema_momentum`** *(float, default=0.99)*: Momentum factor for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: A scaling factor for the loss during gradient computation, useful for mixed precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: The number of steps over which gradients are accumulated before updating parameters.
- **`name`** *(str, default="kron")*: The name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.kron import Kron

# Instantiate the Kron optimizer with default preconditioner schedule.
optimizer = Kron(
    learning_rate=0.0003,
    weight_decay=1e-4,
    b1=0.9,
    # preconditioner_update_probability can be omitted to use the default schedule
    max_size_triangular=8192,
    min_ndim_triangular=2,
    memory_save_mode="smart_one_diag",
    momentum_into_precond_update=True,
    precond_lr=0.1,
    precond_init_scale=1.0,
    mu_dtype=tf.float32,
    precond_dtype=tf.float32,
)

# Compile a Keras model using the Kron optimizer.
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Adalite

**Overview**:

The `Adalite` optimizer is an adaptive optimization algorithm that extends ideas from adaptive methods (such as Adam) by integrating trust ratio scaling and specialized handling for both low- and high-dimensional parameters. It computes per-parameter moving averages and normalizes gradient updates based on the ratio between the parameter norm and the gradient norm. Additionally, Adalite supports both coupled and decoupled weight decay, along with gradient clipping and other stabilization techniques, making it well-suited for training deep neural networks with diverse parameter scales.

---

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment (mean) estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment (variance) estimates.
- **`weight_decay`** *(float, default=1e-2)*: Coefficient for weight decay regularization.
- **`weight_decouple`** *(bool, default=False)*: If set to True, weight decay is applied in a decoupled manner (directly scaling parameters) rather than added to the gradients.
- **`fixed_decay`** *(bool, default=False)*: If True, uses a fixed weight decay rather than scaling it by the learning rate.
- **`g_norm_min`** *(float, default=1e-10)*: Minimum threshold for the gradient norm to prevent division by very small values.
- **`ratio_min`** *(float, default=1e-4)*: Minimum allowed trust ratio used for scaling the gradient.
- **`tau`** *(float, default=1.0)*: Temperature parameter used in computing importance weights when handling high-dimensional parameters.
- **`eps1`** *(float, default=1e-6)*: Small constant for numerical stability in normalization during update computation.
- **`eps2`** *(float, default=1e-10)*: Additional small constant to ensure stability in aggregate computations.
- **`clipnorm`** *(float, optional)*: If provided, gradients are clipped to this maximum norm.
- **`clipvalue`** *(float, optional)*: If provided, gradients are clipped element-wise to this maximum value.
- **`global_clipnorm`** *(float, optional)*: If set, clips the global norm of all gradients to this value.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to the model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum factor used for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting the EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation (useful in mixed precision training).
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before updating parameters.
- **`name`** *(str, default="adalite")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.adalite import Adalite

# Instantiate optimizer
optimizer = Adalite(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    weight_decay=1e-2,
    weight_decouple=False,
    fixed_decay=False,
    g_norm_min=1e-10,
    ratio_min=1e-4,
    tau=1.0,
    eps1=1e-6,
    eps2=1e-10
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

# AdaNorm

**Overview**:

The `AdaNorm` optimizer is an extension of the Adam optimizer that introduces an adaptive normalization of gradients. By maintaining an exponential moving average of the gradient norms, AdaNorm scales updates based on the relative magnitude of current and past gradients. This approach helps stabilize training and can improve convergence in scenarios where gradient scales vary over time.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size used for updating parameters.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the moving average of gradients.
- **`beta2`** *(float, default=0.99)*: Exponential decay rate for the moving average of squared gradients.
- **`epsilon`** *(float, default=1e-8)*: A small constant to prevent division by zero and ensure numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. When enabled, this can be applied decoupled from the gradient update.
- **`r`** *(float, default=0.95)*: Smoothing factor for the exponential moving average of the gradient norms.
- **`weight_decouple`** *(bool, default=True)*: Determines if weight decay should be decoupled from the gradient-based update.
- **`fixed_decay`** *(bool, default=False)*: When set to True, applies a fixed weight decay rather than scaling it with the learning rate.
- **`ams_bound`** *(bool, default=False)*: If True, uses the AMS-bound variant to ensure more stable convergence.
- **`adam_debias`** *(bool, default=False)*: Chooses whether to apply Adam-style debiasing to the learning rate.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm to mitigate exploding gradients.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients based on the global norm across all variables.
- **`use_ema`** *(bool, default=False)*: If True, applies an Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum coefficient for the EMA of model weights.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) at which the EMA weights are updated.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation, useful in mixed precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps over which gradients are accumulated before performing an update.
- **`name`** *(str, default="adanorm")*: Identifier name for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.adanorm import AdaNorm

# Instantiate the AdaNorm optimizer
optimizer = AdaNorm(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.99,
    epsilon=1e-8,
    weight_decay=1e-2,
    r=0.95,
    weight_decouple=True,
    ams_bound=False,
    adam_debias=False
)

# Compile a TensorFlow/Keras model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AdaPNM

**Overview**:

The `AdaPNM` optimizer is a variant of adaptive gradient methods that integrates a predictive negative momentum mechanism. It leverages two momentum accumulators—a positive and a negative moving average—to dynamically adjust the update direction, aiming to counteract the overshooting typical in momentum methods. Additionally, it optionally incorporates gradient normalization (akin to AdaNorm) and supports features like decoupled weight decay and AMS-bound adjustments, making it well-suited for tasks with noisy gradients and complex optimization landscapes.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size used for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the primary (positive) momentum estimate.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment (variance) estimate.
- **`beta3`** *(float, default=1.0)*: Factor controlling the influence of the negative momentum component.
- **`epsilon`** *(float, default=1e-8)*: A small constant to maintain numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay regularization.
- **`weight_decouple`** *(bool, default=True)*: If True, applies weight decay independently of the gradient-based update.
- **`fixed_decay`** *(bool, default=False)*: Uses a fixed weight decay value rather than scaling it by the learning rate.
- **`ams_bound`** *(bool, default=True)*: Whether to use the AMS-bound variant to cap the second moment, improving stability.
- **`r`** *(float, default=0.95)*: Smoothing factor for the exponential moving average of gradient norms (active when `adanorm` is enabled).
- **`adanorm`** *(bool, default=False)*: When enabled, applies adaptive gradient normalization to adjust the gradient scale.
- **`adam_debias`** *(bool, default=False)*: Determines whether to apply Adam-style bias correction to the learning rate.
- **`clipnorm`** *(float, optional)*: Maximum norm for clipping gradients.
- **`clipvalue`** *(float, optional)*: Maximum value for clipping gradients.
- **`global_clipnorm`** *(float, optional)*: Clips gradients based on a global norm across all model parameters.
- **`use_ema`** *(bool, default=False)*: Enables the use of an Exponential Moving Average (EMA) on model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum coefficient for computing the EMA of model weights.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) at which the EMA weights are updated.
- **`loss_scale_factor`** *(float, optional)*: Factor to scale the loss during gradient computation, useful in mixed-precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before updating the model.
- **`name`** *(str, default="adapnm")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.adapnm import AdaPNM

# Instantiate the AdaPNM optimizer
optimizer = AdaPNM(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    beta3=1.0,
    weight_decay=1e-2,
    adanorm=True,      # Enable adaptive gradient normalization if needed
    ams_bound=True     # Use AMS-bound variant for additional stability
)

# Compile a TensorFlow/Keras model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AdaShift

**Overview**:

The `AdaShift` optimizer is an adaptive optimization algorithm that leverages a shifting window of historical gradients to compute more robust update statistics. By maintaining a fixed-size deque of past gradients (controlled by `keep_num`), it shifts and aggregates gradient information, which can lead to more stable estimates of the first and second moments. An optional cautious mode further refines updates by masking inconsistent directions between the gradient and the update, potentially reducing harmful parameter oscillations. This design is particularly effective in settings with noisy or non-stationary gradients.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the moving average of gradients.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-10)*: Small constant for numerical stability.
- **`keep_num`** *(int, default=10)*: Number of past gradients to maintain in the shifting window.
- **`reduce_func`** *(function, default=`tf.reduce_max`)*: Function used to aggregate squared gradients from the shifted window.
- **`cautious`** *(bool, default=False)*: When enabled, applies a masking strategy to filter the update, ensuring that only consistent gradient directions contribute.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all model parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to the model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for updating the EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation, useful for mixed precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before updating the model.
- **`name`** *(str, default="adashift")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.adashift import AdaShift

# Instantiate the AdaShift optimizer
optimizer = AdaShift(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-10,
    keep_num=10,
    reduce_func=tf.reduce_max,
    cautious=True  # Enable cautious update mode for additional stability
)

# Compile a TensorFlow/Keras model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AdaSmooth

**Overview**:

The `AdaSmooth` optimizer is an adaptive optimization algorithm that adjusts the update steps by leveraging the smoothness of parameter transitions. By comparing the current parameters to their previous states, it computes a smoothing coefficient that modulates the effective learning rate. This approach helps reduce oscillations during training and can promote more stable convergence, particularly in settings with rapidly changing gradients.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.5)*: Coefficient for accumulating the difference between current and previous parameters, influencing the smoothing effect.
- **`beta2`** *(float, default=0.99)*: Coefficient for updating the exponential moving average of squared gradients.
- **`epsilon`** *(float, default=1e-6)*: Small constant added for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay regularization.
- **`weight_decouple`** *(bool, default=False)*: If True, decouples weight decay from the gradient update.
- **`fixed_decay`** *(bool, default=False)*: If True, applies a fixed weight decay rather than scaling it with the learning rate.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for computing the EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for updating the EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation, useful in mixed-precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps over which gradients are accumulated before updating the model.
- **`name`** *(str, default="adasmooth")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.adasmooth import AdaSmooth

# Instantiate the AdaSmooth optimizer
optimizer = AdaSmooth(
    learning_rate=1e-3,
    beta1=0.5,
    beta2=0.99,
    epsilon=1e-6,
    weight_decay=1e-2,
    weight_decouple=False,
    fixed_decay=False
)

# Compile a TensorFlow/Keras model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AdEMAMix

**Overview**:

The `AdEMAMix` optimizer is an advanced adaptive optimization algorithm that extends traditional Adam methods by incorporating a slow-moving average of gradients. By mixing the fast (standard) exponential moving average with a slower one—weighted by an adaptively scheduled mixing coefficient—the optimizer is designed to capture both short-term fluctuations and long-term trends in the gradient. This dynamic balance can lead to more robust convergence and improved generalization, particularly in challenging training scenarios.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment (fast moving average) estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`beta3`** *(float, default=0.9999)*: Base decay rate for the slow moving average component.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: Coefficient for weight decay. When non-zero, weight decay is applied directly to the parameters.
- **`alpha`** *(float, default=5.0)*: Mixing coefficient that scales the contribution of the slow moving average.
- **`T_alpha_beta3`** *(float, optional)*: If provided, defines a scheduling horizon over which both the effective mixing coefficient (`alpha_t`) and the slow decay rate (`beta3_t`) are adaptively annealed.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients based on the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to the model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum coefficient for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) at which EMA weights are updated.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation, useful in mixed precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before updating the model.
- **`name`** *(str, default="ademamix")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.ademamix import AdEMAMix

# Instantiate the AdEMAMix optimizer
optimizer = AdEMAMix(
    learning_rate=1e-3,
    weight_decay=1e-2,
    beta1=0.9,
    beta2=0.999,
    beta3=0.9999,
    alpha=5.0,
    T_alpha_beta3=10000  # Optional scheduling parameter for adaptive mixing
)

# Compile a TensorFlow/Keras model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
``` 

# Aida

**Overview**:

The `Aida` optimizer is an advanced adaptive optimization algorithm that builds on Adam by integrating gradient projection and optional rectification mechanisms. It refines momentum estimation through iterative projection steps—controlled by parameters `k` and `xi`—which help align the gradient and momentum directions. Additionally, Aida offers flexibility with decoupled weight decay, adaptive gradient normalization (akin to AdaNorm), and a dynamic switch to SGD-like updates when gradient variance is low. These features make it effective in handling noisy gradients and improving convergence stability across a variety of tasks.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment (momentum) estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay regularization. When non-zero, weight decay can be applied in a decoupled manner if `weight_decouple` is True.
- **`k`** *(int, default=2)*: Number of iterations for the gradient projection process used to refine momentum alignment.
- **`xi`** *(float, default=1e-20)*: Small constant added during projection to avoid division by zero.
- **`weight_decouple`** *(bool, default=False)*: If True, applies weight decay decoupled from the gradient update.
- **`fixed_decay`** *(bool, default=False)*: When enabled, uses a fixed weight decay instead of scaling it by the learning rate.
- **`rectify`** *(bool, default=False)*: Enables rectified updates inspired by RAdam, adjusting the update based on the variance of gradients.
- **`n_sma_threshold`** *(int, default=5)*: Threshold for the number of stochastic moving average steps used to decide whether to apply rectification or switch to an SGD-like update.
- **`degenerated_to_sgd`** *(bool, default=True)*: If True, the optimizer degenerates into SGD when gradient variance is too low.
- **`ams_bound`** *(bool, default=False)*: If True, employs the AMS-bound variant to maintain a maximum of second moment estimates for added stability.
- **`r`** *(float, default=0.95)*: Smoothing factor for the exponential moving average of gradient norms when `adanorm` is enabled.
- **`adanorm`** *(bool, default=False)*: If True, applies adaptive gradient normalization to scale the gradients based on their historical norms.
- **`adam_debias`** *(bool, default=False)*: Determines whether to apply Adam-style bias correction to the updates.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all model parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to the model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for updating the EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation, useful in mixed precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps over which gradients are accumulated before an update.
- **`name`** *(str, default="aida")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.aida import Aida

# Instantiate the Aida optimizer
optimizer = Aida(
    learning_rate=1e-3,
    weight_decay=1e-2,
    rectify=True,
    weight_decouple=True,
    k=2,
    xi=1e-20,
    n_sma_threshold=5,
    degenerated_to_sgd=True,
    ams_bound=False,
    adanorm=True,
    adam_debias=False
)

# Compile a TensorFlow/Keras model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AliG

**Overview**:

The `AliG` optimizer is an adaptive optimization algorithm that dynamically scales its update step by leveraging the ratio of the current loss to the global gradient norm. In addition to this adaptive step sizing, AliG optionally integrates momentum and projection operations to enforce constraints (such as l2-norm projection) on the parameters. This design helps stabilize training by balancing the update magnitude across parameters and can be particularly useful when the scale of the loss and gradients varies significantly.

**Parameters**:

- **`max_lr`** *(float, optional)*: Maximum allowable learning rate; the computed step size is capped at this value if provided.
- **`projection_fn`** *(function, optional)*: A projection function that is applied after each update to enforce constraints (e.g., l2-norm projection).
- **`momentum`** *(float, default=0.0)*: Momentum factor for smoothing updates. When set above 0, a momentum buffer is maintained for each parameter.
- **`adjusted_momentum`** *(bool, default=False)*: If True, uses an adjusted momentum update scheme that scales the momentum buffer differently.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum coefficient for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps over which gradients are accumulated before updating the model.
- **`name`** *(str, default="alig")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.alig import AliG, l2_projection

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

def my_projection():
    l2_projection(model.trainable_variables, max_norm=1e2)

optimizer = AliG(
    max_lr=1e-2,
    projection_fn=my_projection,
    momentum=0.9,
    adjusted_momentum=True
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

for epoch in range(10):
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = loss_fn(y_batch, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables), loss)
    print(f"Epoch {epoch + 1} completed.")
```

# Amos

**Overview**:

The `Amos` optimizer is an advanced optimization algorithm that extends adaptive gradient methods by incorporating additional regularization and decay mechanisms. By maintaining a running average of squared gradients with a high decay rate and combining it with explicit decay factors (`c_coef` and `d_coef`), Amos dynamically scales updates based on both gradient statistics and the inherent scale of the model parameters. Optionally, a momentum term can be applied to further smooth updates. This design makes Amos particularly effective for training models with complex loss landscapes and noisy gradients.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta`** *(float, default=0.999)*: Exponential decay rate for the running average of squared gradients.
- **`epsilon`** *(float, default=1e-18)*: Small constant for numerical stability to avoid division by zero.
- **`momentum`** *(float, default=0.0)*: Momentum factor for smoothing the parameter updates.
- **`extra_l2`** *(float, default=0.0)*: Additional L2 regularization coefficient applied in the update computation.
- **`c_coef`** *(float, default=0.25)*: Coefficient used to compute a decay factor based on the squared gradient statistics.
- **`d_coef`** *(float, default=0.25)*: Coefficient used to further adjust the update magnitude via an additional decay factor.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients before updating the parameters.
- **`name`** *(str, default="amos")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.amos import Amos

# Instantiate the Amos optimizer
optimizer = Amos(
    learning_rate=1e-3,
    beta=0.999,
    epsilon=1e-18,
    momentum=0.9,
    extra_l2=1e-4,
    c_coef=0.25,
    d_coef=0.25
)

# Compile a model with the Amos optimizer
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AvaGrad

**Overview**:

The `AvaGrad` optimizer is an adaptive optimization algorithm that adjusts parameter updates based on both the variance of the gradients and a dynamically computed scaling factor. By maintaining running averages of gradients and squared gradients with decay rates (`beta1` and `beta2`), AvaGrad computes a global scaling factor (γ) that modulates the learning rate. Additionally, it supports decoupled weight decay and optional Adam-style bias correction. This design makes AvaGrad particularly suitable for settings where the gradient scales vary significantly, promoting more stable convergence in training deep neural networks.

**Parameters**:

- **`learning_rate`** *(float, default=1e-1)*: The base learning rate for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-1)*: A small constant added for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. When non-zero, weight decay is applied either in a decoupled manner or directly to the gradients.
- **`weight_decouple`** *(bool, default=True)*: Determines whether to decouple weight decay from the gradient update.
- **`fixed_decay`** *(bool, default=False)*: If True, applies a fixed weight decay rather than scaling it by the learning rate.
- **`adam_debias`** *(bool, default=False)*: If True, applies Adam-style bias correction to the learning rate.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before updating parameters.
- **`name`** *(str, default="avagrad")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.avagrad import AvaGrad

# Instantiate the AvaGrad optimizer
optimizer = AvaGrad(
    learning_rate=1e-1,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-1,
    weight_decay=1e-2,
    weight_decouple=True,
    fixed_decay=False,
    adam_debias=False
)

# Compile a model with the AvaGrad optimizer
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# CAME

**Overview**:

The `CAME` optimizer is an advanced adaptive optimization algorithm that leverages factored second-moment estimation to better capture the structure of multi-dimensional parameters. It maintains separate exponential moving averages for the rows and columns of the gradient (when applicable), which allows it to approximate the squared gradients more efficiently for tensors with two or more dimensions. In addition, CAME employs dynamic update clipping based on the root mean square (RMS) of the parameters, decoupled weight decay, and an optional AMS-bound mechanism. These features combine to improve stability and generalization, especially when training large-scale deep neural networks with complex parameter structures.

**Parameters**:

- **`learning_rate`** *(float, default=2e-4)*: The step size used for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment (mean) estimate.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the factored second moment estimate.
- **`beta3`** *(float, default=0.9999)*: Exponential decay rate for the residual squared gradient estimate.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. When non-zero, weight decay is applied either decoupled from the gradient update or directly, depending on `weight_decouple`.
- **`weight_decouple`** *(bool, default=True)*: Determines whether to decouple weight decay from the gradient update.
- **`fixed_decay`** *(bool, default=False)*: If True, applies a fixed weight decay rather than scaling it by the learning rate.
- **`clip_threshold`** *(float, default=1.0)*: Threshold for clipping the update based on its RMS value.
- **`ams_bound`** *(bool, default=False)*: Whether to use the AMS-bound variant, which maintains a maximum of the squared gradient estimates for added stability.
- **`eps1`** *(float, default=1e-30)*: Small constant for numerical stability in the squared gradient computation.
- **`eps2`** *(float, default=1e-16)*: Small constant for numerical stability in the residual computation.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients before updating parameters.
- **`name`** *(str, default="came")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.came import CAME

# Instantiate the CAME optimizer
optimizer = CAME(
    learning_rate=2e-4,
    beta1=0.9,
    beta2=0.999,
    beta3=0.9999,
    weight_decay=1e-2,
    weight_decouple=True,
    fixed_decay=False,
    clip_threshold=1.0,
    ams_bound=False,
    eps1=1e-30,
    eps2=1e-16
)

# Compile a model with the CAME optimizer
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# DAdaptAdam

**Overview**:

The `DAdaptAdam` optimizer is an adaptive optimization algorithm that dynamically adjusts its update scaling based on observed gradient statistics. It extends the Adam framework by introducing a separate scaling accumulator and an adaptive scaling factor (d₀) that evolves during training. This dynamic adaptation enables DAdaptAdam to automatically calibrate the effective learning rate based on the structure and variability of gradients. Additional features such as optional bias correction, decoupled weight decay, and fixed decay further enhance its robustness in diverse training scenarios.

**Parameters**:

- **`learning_rate`** *(float, default=1.0)*: Base learning rate for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment (mean) estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment (variance) estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. Applied either in a decoupled fashion or directly, depending on `weight_decouple`.
- **`d0`** *(float, default=1e-6)*: Initial scaling factor that adapts the update magnitude based on gradient statistics.
- **`growth_rate`** *(float, default=`inf`)*: Upper bound on the allowed growth of the scaling factor.
- **`weight_decouple`** *(bool, default=True)*: Whether to decouple weight decay from the gradient update.
- **`fixed_decay`** *(bool, default=False)*: Uses a fixed weight decay value instead of scaling it by the learning rate.
- **`bias_correction`** *(bool, default=False)*: If enabled, applies bias correction when computing the adaptive scaling factor.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum coefficient for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before an update.
- **`name`** *(str, default="dadaptadam")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.dadaptadam import DAdaptAdam

# Instantiate the DAdaptAdam optimizer
optimizer = DAdaptAdam(
    learning_rate=1.0,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.0,
    d0=1e-6,
    growth_rate=float('inf'),
    weight_decouple=True,
    fixed_decay=False,
    bias_correction=False
)

# Compile a model with the DAdaptAdam optimizer
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# DAdaptSGD

**Overview**:

The `DAdaptSGD` optimizer is an adaptive variant of stochastic gradient descent that automatically calibrates its effective learning rate based on the observed gradient statistics. By maintaining an accumulated statistic of gradients and leveraging a dynamic scaling factor (d₀), it adjusts updates to better match the curvature of the loss landscape. Additionally, the optimizer supports momentum and decoupled weight decay, making it a robust choice for training deep neural networks with varying gradient scales.

**Parameters**:

- **`learning_rate`** *(float, default=1.0)*: The base learning rate for parameter updates.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay regularization. When non-zero, weight decay is applied either decoupled from the gradient update or directly, based on `weight_decouple`.
- **`momentum`** *(float, default=0.9)*: Momentum factor for smoothing updates.
- **`d0`** *(float, default=1e-6)*: Initial scaling factor that adapts the effective learning rate based on accumulated gradient information.
- **`growth_rate`** *(float, default=`inf`)*: Maximum factor by which the scaling factor is allowed to grow during training.
- **`weight_decouple`** *(bool, default=True)*: Determines whether weight decay is applied decoupled from the gradient update.
- **`fixed_decay`** *(bool, default=False)*: If enabled, applies a fixed weight decay rather than scaling it by the learning rate.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients before performing an update.
- **`name`** *(str, default="dadaptsgd")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.dadaptsgd import DAdaptSGD

# Instantiate the DAdaptSGD optimizer
optimizer = DAdaptSGD(
    learning_rate=1.0,
    weight_decay=1e-2,
    momentum=0.9,
    d0=1e-6,
    growth_rate=float('inf'),
    weight_decouple=True,
    fixed_decay=False
)

# Compile a model with the DAdaptSGD optimizer
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# DAdaptLion

**Overview**:

The `DAdaptLion` optimizer is an adaptive variant of the Lion optimizer that dynamically adjusts its scaling factor based on accumulated gradient statistics. It uses a sign-based update rule combined with an exponential moving average and a secondary accumulator to compute a dynamic scaling parameter (d₀). This mechanism allows the optimizer to automatically calibrate its effective learning rate in response to the observed gradient structure while optionally applying decoupled weight decay.

**Parameters**:

- **`learning_rate`** *(float, default=1.0)*: The base learning rate for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimate used in the sign update.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for accumulating gradient statistics.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. When non-zero, weight decay is applied either decoupled from the update or directly, depending on `weight_decouple`.
- **`d0`** *(float, default=1e-6)*: Initial adaptive scaling factor that adjusts the effective step size based on gradient statistics.
- **`weight_decouple`** *(bool, default=True)*: If True, applies weight decay decoupled from the gradient update.
- **`fixed_decay`** *(bool, default=False)*: Uses a fixed weight decay value instead of scaling it by the learning rate.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum coefficient for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before performing an update.
- **`name`** *(str, default="dadaptlion")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.dadaptlion import DAdaptLion

# Instantiate the DAdaptLion optimizer
optimizer = DAdaptLion(
    learning_rate=1.0,
    beta1=0.9,
    beta2=0.999,
    weight_decay=1e-2,
    d0=1e-6,
    weight_decouple=True,
    fixed_decay=False
)

# Compile a TensorFlow/Keras model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# DAdaptAdan

**Overview**:

The `DAdaptAdan` optimizer is an adaptive optimization algorithm that extends the Adam family by dynamically adjusting its effective learning rate based on observed gradient differences and higher-order statistics. By maintaining separate exponential moving averages for the gradients, their squared values, and their differences, it computes an adaptive scaling factor (d₀) that automatically calibrates the update magnitude during training. This approach aims to improve convergence and robustness, especially in scenarios with varying gradient dynamics. Additionally, the optimizer supports decoupled weight decay and flexible decay scaling.

**Parameters**:

- **`learning_rate`** *(float, default=1.0)*: The base learning rate for parameter updates.
- **`beta1`** *(float, default=0.98)*: Exponential decay rate for the first moment (mean) estimates.
- **`beta2`** *(float, default=0.92)*: Exponential decay rate for the moving average of gradient differences.
- **`beta3`** *(float, default=0.99)*: Exponential decay rate for the second moment (variance) estimates of the gradient differences.
- **`epsilon`** *(float, default=1e-8)*: Small constant added for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. When non-zero, weight decay is applied either in a decoupled manner or directly to the gradients.
- **`d0`** *(float, default=1e-6)*: Initial adaptive scaling factor that governs the effective update magnitude.
- **`growth_rate`** *(float, default=`inf`)*: Upper bound for the allowed growth of the scaling factor during training.
- **`weight_decouple`** *(bool, default=True)*: If True, decouples weight decay from the gradient update.
- **`fixed_decay`** *(bool, default=False)*: Uses fixed weight decay instead of scaling it by the learning rate.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before performing an update.
- **`name`** *(str, default="dadaptadan")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.dadaptadan import DAdaptAdan

# Instantiate the DAdaptAdan optimizer
optimizer = DAdaptAdan(
    learning_rate=1.0,
    beta1=0.98,
    beta2=0.92,
    beta3=0.99,
    epsilon=1e-8,
    weight_decay=1e-2,
    d0=1e-6,
    growth_rate=float('inf'),
    weight_decouple=True,
    fixed_decay=False
)

# Compile a model using the DAdaptAdan optimizer
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# DAdaptAdaGrad

**Overview**:

The `DAdaptAdaGrad` optimizer is an adaptive optimization algorithm that builds upon the AdaGrad method by dynamically adjusting its effective update scaling. It maintains per-parameter accumulators for squared gradients (stored in `alpha_k`), an auxiliary accumulator `sk` for gradient updates, and the initial parameter values `x0`. These accumulators are used to compute a dynamic scaling factor (d₀) that is adjusted during training based on the difference between the weighted squared norm of the accumulated updates and the accumulated squared gradients. The optimizer also supports momentum, decoupled weight decay, and bias correction, and is capable of handling sparse gradients via specialized masking functions.

**Parameters**:

- **`learning_rate`** *(float, default=1.0)*: The base step size for parameter updates.
- **`epsilon`** *(float, default=0.0)*: A small constant for numerical stability, added to denominators to avoid division by zero.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay regularization. When non-zero, weight decay is applied either in a decoupled manner or directly, depending on `weight_decouple`.
- **`momentum`** *(float, default=0.0)*: Momentum factor for smoothing the updates. When set above 0, a momentum update is applied to the parameters.
- **`d0`** *(float, default=1e-6)*: Initial adaptive scaling factor that controls the magnitude of the updates.
- **`growth_rate`** *(float, default=`inf`)*: The maximum factor by which the adaptive scaling factor is allowed to grow.
- **`weight_decouple`** *(bool, default=True)*: Determines whether weight decay is decoupled from the gradient update.
- **`fixed_decay`** *(bool, default=False)*: Uses a fixed weight decay value rather than scaling it by the learning rate.
- **`bias_correction`** *(bool, default=False)*: If enabled, applies bias correction during the computation of the adaptive scaling factor.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before performing an update.
- **`name`** *(str, default="dadaptadagrad")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.dadaptadagrad import DAdaptAdaGrad

# Instantiate the DAdaptAdaGrad optimizer
optimizer = DAdaptAdaGrad(
    learning_rate=1.0,
    epsilon=0.0,
    weight_decay=1e-2,
    momentum=0.9,
    d0=1e-6,
    growth_rate=float('inf'),
    weight_decouple=True,
    fixed_decay=False,
    bias_correction=False
)

# Compile a model using the DAdaptAdaGrad optimizer
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# EXAdam

**Overview**:

The `EXAdam` optimizer extends the Adam framework by integrating a logarithmically scaled step size along with adaptive adjustments derived from the exponential moving averages of the gradients and their squares. In particular, it computes additional scaling factors (d1 and d2) based on bias-corrected moment estimates to modulate the effective learning rate. This design makes EXAdam especially effective in environments with noisy or non-stationary gradients, promoting more stable convergence and improved generalization.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The base step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. Applied either decoupled from the gradient update or directly based on `weight_decouple`.
- **`weight_decouple`** *(bool, default=True)*: Whether to decouple weight decay from the gradient update.
- **`fixed_decay`** *(bool, default=False)*: Uses fixed weight decay instead of scaling it by the learning rate.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum coefficient for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients before performing an update.
- **`name`** *(str, default="exadam")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.exadam import EXAdam

# Instantiate the EXAdam optimizer
optimizer = EXAdam(
    learning_rate=1e-3,
    weight_decay=1e-2,
    weight_decouple=True,
    fixed_decay=False
)

# Compile a model using the EXAdam optimizer
model.compile(optimizer=optimizer, 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# FAdam

**Overview**:

The `FAdam` optimizer is a variant of the Adam optimizer that incorporates Fisher Information Matrix (FIM) based scaling into the update rule. By maintaining both a momentum term and a FIM estimate for each parameter, FAdam adjusts the gradient updates using an exponentiation factor (p) and a clipping mechanism to enhance stability. This approach aims to improve convergence, especially in settings with noisy gradients, while also supporting decoupled weight decay.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment (momentum) estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the FIM (second moment) estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.1)*: Coefficient for weight decay. Applied either decoupled from the gradient update or directly based on `weight_decouple`.
- **`clip`** *(float, default=1.0)*: Clipping threshold applied to normalized gradients to ensure stability.
- **`p`** *(float, default=0.5)*: Exponent factor applied to the FIM estimate, modulating its effect on the update.
- **`momentum_dtype`** *(tf.dtype, default=tf.float32)*: Data type used for the momentum accumulator.
- **`fim_dtype`** *(tf.dtype, default=tf.float32)*: Data type used for the FIM accumulator.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before an update.
- **`name`** *(str, default="fadam")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.fadam import FAdam

# Instantiate the FAdam optimizer
optimizer = FAdam(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.1,
    clip=1.0,
    p=0.5,
    momentum_dtype=tf.float32,
    fim_dtype=tf.float32
)

# Compile a model with the FAdam optimizer
model.compile(optimizer=optimizer, 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```
