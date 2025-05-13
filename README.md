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

# FOCUS

**Overview**:

The `FOCUS` optimizer is an adaptive gradient method that combines momentum-based updates with parameter tracking to enhance training stability. It maintains exponential moving averages of both the gradients and the parameters to form a predictive term (p̄) representing a smoothed estimate of the model’s parameters. The update is computed based on the sign differences between the current parameters and this prediction, with an additional term derived from the gradient’s moving average. The hyperparameter `gamma` scales the influence of these components. This design helps improve convergence, particularly in scenarios with noisy or shifting gradient landscapes.

**Parameters**:

- **`learning_rate`** *(float, default=1e-2)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment (gradient) estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for tracking the parameters (p̄).
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. Applied directly to the parameters if non-zero.
- **`gamma`** *(float, default=0.1)*: Scaling factor that modulates the combined influence of the momentum and parameter tracking signals.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum coefficient for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before performing an update.
- **`name`** *(str, default="focus")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.focus import FOCUS

# Instantiate the FOCUS optimizer
optimizer = FOCUS(
    learning_rate=1e-2,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.0,
    gamma=0.1
)

# Compile a model using the FOCUS optimizer
model.compile(optimizer=optimizer, 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Fromage

**Overview**:

The `Fromage` optimizer is a novel optimization method that departs from traditional gradient descent by incorporating geometric scaling based on parameter and gradient norms. Instead of relying solely on the raw gradient, Fromage rescales the update direction by the ratio of the parameter norm to the gradient norm and then normalizes the updated parameters to control the effective step length. An optional bound (`p_bound`) can be specified to limit the growth of parameter norms, helping to maintain stability during training. This approach is particularly useful when preserving the scale of the parameters is critical for convergence and generalization.

**Parameters**:

- **`learning_rate`** *(float, default=1e-2)*: The base step size for parameter updates.
- **`p_bound`** *(float, optional)*: A factor used to bound the norm of parameters after the update. If provided, parameters are clipped so that their norm does not exceed the initial norm multiplied by this value.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before performing an update.
- **`name`** *(str, default="fromage")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.fromage import Fromage

# Instantiate the Fromage optimizer with an optional p_bound parameter
optimizer = Fromage(
    learning_rate=1e-2,
    p_bound=0.1  # Optional: limits the parameter norm to 0.1 times the initial norm
)

# Compile a model using the Fromage optimizer
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# GaLore

**Overview**:

The `GaLore` optimizer extends the Adam framework by incorporating low-rank projection techniques into the gradient update process. For parameters with two or more dimensions, it leverages a dedicated projector (via `GaLoreProjector`) to project the gradient into a lower-dimensional subspace before applying the adaptive update. This extra projection step is intended to better capture and exploit the low-rank structure of weight matrices, potentially leading to more robust convergence. Along with standard exponential moving averages for gradients and their squares, GaLore supports decoupled weight decay and bias correction via learning rate scaling.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The base step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-6)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. Applied in a decoupled manner if enabled.
- **`rank`** *(int, optional)*: The target rank for low-rank projection. When provided and the parameter is multi-dimensional, the projector is activated.
- **`update_proj_gap`** *(int, optional)*: Frequency gap for updating the projection; governs how often the low-rank projection is recalculated.
- **`scale`** *(float, optional)*: Scaling factor applied within the projector.
- **`projection_type`** *(str, optional)*: Specifies the type of projection to perform (e.g., symmetric or asymmetric).
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum coefficient for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients.
- **`name`** *(str, default="galore")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.galore import GaLore

# Instantiate the GaLore optimizer with low-rank projection enabled
optimizer = GaLore(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-6,
    weight_decay=1e-4,
    rank=10,  # Enable low-rank projection for parameters with rank ≥ 2
    update_proj_gap=50,
    scale=0.5,
    projection_type="symmetric"
)

# Compile a model using the GaLore optimizer
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Grams

**Overview**:

The `Grams` optimizer is a novel adaptive optimization algorithm that modifies the traditional Adam update by leveraging gradient normalization based on the absolute gradient sign. Instead of directly using the raw gradient, Grams computes an adaptive update by scaling the moving average of gradients with bias correction and then re-normalizing it using the square root of the second moment. The final update is further adjusted by taking the absolute value and reintroducing the original gradient's sign, thereby aiming to stabilize updates in the presence of noisy gradients. This approach can improve convergence and generalization, especially in scenarios where gradient variability is a concern.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-6)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. Applies either decoupled or standard decay based on `weight_decouple`.
- **`weight_decouple`** *(bool, default=True)*: Determines whether weight decay is applied decoupled from the gradient update.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients.
- **`name`** *(str, default="grams")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.grams import Grams

# Instantiate the Grams optimizer
optimizer = Grams(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-6,
    weight_decay=0.0,
    weight_decouple=True
)

# Compile a model with the Grams optimizer
model.compile(optimizer=optimizer, 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Gravity

**Overview**:

The `Gravity` optimizer is a novel optimization algorithm inspired by gravitational dynamics. Instead of directly applying raw gradients, Gravity first computes a modified gradient by scaling the update direction based on the relative magnitude of the gradient. Specifically, it calculates a scaling factor from the inverse of the maximum absolute value of the gradient and then uses this factor to modulate the gradient through a nonlinear transformation. A velocity vector is maintained for each parameter—initialized with random noise scaled by the ratio of `alpha` to the learning rate—and updated as a weighted average using a dynamic coefficient derived from the current update step and the hyperparameter `beta`. The final parameter update is then performed using this velocity, which helps stabilize training and improves convergence, especially in the presence of large gradient variations.

**Parameters**:

- **`learning_rate`** *(float, default=1e-2)*: The base step size for parameter updates.
- **`alpha`** *(float, default=0.01)*: Scaling factor for initializing the velocity vector; it sets the standard deviation of the initial noise relative to the learning rate.
- **`beta`** *(float, default=0.9)*: Smoothing coefficient that influences the update weighting; a dynamic value `beta_t` is computed at each step to balance the contribution of the previous velocity and the new modified gradient.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm to mitigate exploding gradients.
- **`clipvalue`** *(float, optional)*: Clips gradients by a specified value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients based on the global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to the model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for the EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before updating parameters.
- **`name`** *(str, default="gravity")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.gravity import Gravity

# Instantiate the Gravity optimizer
optimizer = Gravity(
    learning_rate=1e-2,
    alpha=0.01,
    beta=0.9
)

# Compile a model using the Gravity optimizer
model.compile(optimizer=optimizer, 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# GrokFastAdamW

**Overview**:

The `GrokFastAdamW` optimizer extends the AdamW algorithm by incorporating a "grokfast" mechanism and optional gradient filtering to accelerate convergence and improve stability. When enabled, the grokfast mechanism maintains an exponential moving average of gradients and, after a specified number of steps, augments the current gradient with a scaled version of this average. In addition, the optimizer supports configurable gradient filtering methods—either a moving-average (MA) filter or an exponential moving average (EMA) filter—through which gradients are preprocessed before the standard AdamW update. It also provides options for decoupled weight decay and learning rate normalization, making it particularly effective in training scenarios with noisy or rapidly changing gradients.

**Parameters**:

- **`learning_rate`** *(float, default=1e-4)*: The base step size for parameter updates. If both `grokfast` and `normalize_lr` are enabled, the learning rate is normalized by dividing by (1 + grokfast_lamb).
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment (mean) estimates.
- **`beta2`** *(float, default=0.99)*: Exponential decay rate for the second moment (variance) estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. Applied decoupled from the gradient update if `weight_decouple` is True.
- **`grokfast`** *(bool, default=True)*: Whether to enable the grokfast mechanism that augments the gradient using its exponential moving average.
- **`grokfast_alpha`** *(float, default=0.98)*: Smoothing coefficient for the grokfast exponential moving average.
- **`grokfast_lamb`** *(float, default=2.0)*: Scaling factor for the additional gradient correction introduced by the grokfast mechanism.
- **`grokfast_after_step`** *(int, default=0)*: The number of steps after which the grokfast mechanism becomes active.
- **`weight_decouple`** *(bool, default=True)*: If True, applies weight decay in a decoupled fashion from the gradient update.
- **`fixed_decay`** *(bool, default=False)*: Uses fixed weight decay instead of scaling it by the learning rate.
- **`normalize_lr`** *(bool, default=True)*: Whether to normalize the learning rate when grokfast is enabled.
- **`filter`** *(str, default="ma")*: Specifies the gradient filtering method to apply before the update. Supported values are `"ma"` for moving average filtering and `"eam"` for EMA filtering.
- **`filter_params`** *(dict, optional)*: A dictionary containing parameters for the gradient filtering functions. For example, if using `"ma"`, it should include keys such as `window_size`, `lamb`, `filter_type` (e.g., "mean" or "sum"), and `warmup`; if using `"eam"`, it should contain keys like `alpha` and `lamb`.
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Options to clip gradients by norm, value, or global norm respectively.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before updating parameters.
- **`name`** *(str, default="grokfastadamw")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.grokfastadamw import GrokFastAdamW

# Define filter parameters for the moving-average filter
filter_params = {
    "window_size": 100,
    "lamb": 5.0,
    "filter_type": "mean",
    "warmup": True
}

# Instantiate the GrokFastAdamW optimizer with gradient filtering enabled
optimizer = GrokFastAdamW(
    learning_rate=1e-4,
    beta1=0.9,
    beta2=0.99,
    epsilon=1e-8,
    weight_decay=1e-2,
    grokfast=True,
    grokfast_alpha=0.98,
    grokfast_lamb=2.0,
    grokfast_after_step=100,
    weight_decouple=True,
    fixed_decay=False,
    normalize_lr=True,
    filter='ma',            # Choose 'ma' for moving-average filter or 'eam' for EMA filter
    filter_params=filter_params
)

# Build a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model with GrokFastAdamW
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Assume train_dataset and val_dataset are predefined tf.data.Datasets
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Kate

**Overview**:

The `Kate` optimizer introduces a novel approach to adaptive gradient updating by incorporating an additional term that modulates the momentum update. Instead of relying solely on conventional moment estimates, Kate computes a modified momentum `m` and a scaling factor `b` based on the squared gradients and an extra hyperparameter `delta`. This design helps to adjust the update magnitude more responsively, which can improve training stability and convergence, particularly in the presence of noisy gradients. Kate also supports decoupled weight decay for more flexible regularization.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. When non-zero, weight decay is applied either in a decoupled manner if `weight_decouple` is True, or directly added to the gradients.
- **`delta`** *(float, default=0.0)*: A hyperparameter that modulates the update of the momentum term by incorporating additional gradient information.
- **`weight_decouple`** *(bool, default=True)*: If True, applies weight decay independently of the gradient update.
- **`fixed_decay`** *(bool, default=False)*: Uses a fixed weight decay instead of scaling it by the learning rate.
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Options to clip gradients to prevent explosion.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to the model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before updating parameters.
- **`name`** *(str, default="kate")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.kate import Kate

# Instantiate the Kate optimizer
optimizer = Kate(
    learning_rate=1e-3,
    epsilon=1e-8,
    weight_decay=0.0,
    delta=0.1,
    weight_decouple=True,
    fixed_decay=False
)

# Compile a model with the Kate optimizer
model.compile(optimizer=optimizer, 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# MSVAG

**Overview**:

The `MSVAG` optimizer implements a variant of adaptive gradient methods that adjusts the effective update by accounting for the variance of the gradients. It maintains exponential moving averages of the gradients and their squares to estimate both the mean (m) and variance (v) of the gradients. Using a derived statistic, ρ, it computes a scaling factor that balances the squared mean against the variance. This factor is then used to modulate the gradient update, potentially improving convergence and stability when training neural networks with noisy or high-variance gradients.

**Parameters**:

- **`learning_rate`** *(float, default=1e-2)*: The base step size for parameter updates.
- **`beta`** *(float, default=0.9)*: Exponential decay rate used for both the first moment (moving average of gradients) and the second moment (moving average of squared gradients).
- **`epsilon`** *(float, default not explicitly set in code, but used internally as 1e-?; here default=1e-6)*: Small constant for numerical stability.
- **`weight_decay`**: Not applicable in MSVAG (set to None).
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum factor for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before an update.
- **`name`** *(str, default="msvag")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.msvag import MSVAG

# Instantiate the MSVAG optimizer
optimizer = MSVAG(
    learning_rate=1e-2,
    beta=0.9
)

# Compile a model using the MSVAG optimizer
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Muon

**Overview**:

The `Muon` optimizer is a novel optimization algorithm designed for deep learning, leveraging Newton-Schulz iterations for orthogonalization and integrating momentum-based updates with adaptive learning rate strategies. It supports weight decay, Nesterov momentum, and an optional AdamW-based update step for enhanced performance and stability in training deep neural networks.

**Parameters**:

- **`learning_rate`** *(float, default=2e-2)*: The primary learning rate used for updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimate.
- **`beta2`** *(float, default=0.95)*: Exponential decay rate for the second moment estimate.
- **`weight_decay`** *(float, default=1e-2)*: Weight decay coefficient for regularization.
- **`momentum`** *(float, default=0.95)*: Momentum factor for gradient updates.
- **`weight_decouple`** *(bool, default=True)*: Whether to use decoupled weight decay as in AdamW.
- **`nesterov`** *(bool, default=True)*: Whether to apply Nesterov momentum.
- **`ns_steps`** *(int, default=5)*: Number of Newton-Schulz iterations for orthogonalization.
- **`use_adjusted_lr`** *(bool, default=False)*: Whether to adjust the learning rate dynamically based on parameter shape.
- **`adamw_params`** *(list, optional)*: Parameters for AdamW-based updates.
- **`adamw_lr`** *(float, default=3e-4)*: Learning rate for the AdamW update step.
- **`adamw_wd`** *(float, default=0.0)*: Weight decay for AdamW.
- **`adamw_eps`** *(float, default=1e-8)*: Small constant for numerical stability in AdamW.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients.
- **`name`** *(str, default="muon")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.muon import Muon

# Instantiate optimizer
optimizer = Muon(
    learning_rate=2e-2,
    weight_decay=1e-2,
    momentum=0.95,
    nesterov=True,
    ns_steps=5
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Nero

**Overview**:

The `Nero` optimizer is an adaptive gradient method that operates on a per-neuron basis by normalizing parameters and gradients across each neuron. It maintains a running average of squared neuron-wise gradient norms and scales updates according to these statistics, ensuring that each neuron’s update is adjusted relative to its historical gradient magnitude. Optional per-layer constraints re-center and re-normalize weights after each update, promoting stable training dynamics and mitigating internal covariate shift.

**Parameters**:

- **`learning_rate`** *(float, default=0.01)*: The base step size for parameter updates.
- **`beta`** *(float, default=0.999)*: Exponential decay rate for the moving average of squared neuron gradient norms.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in denominator computations.
- **`constraints`** *(bool, default=True)*: If True, applies per-layer re-centering and re-normalization constraints to weight tensors after each update.
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Options for gradient clipping by norm, value, or global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before updating.
- **`name`** *(str, default="nero")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.nero import Nero

# Instantiate the Nero optimizer
optimizer = Nero(
    learning_rate=0.01,
    beta=0.999,
    epsilon=1e-8,
    constraints=True
)

# Build a simple model\ nmodel = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model with Nero optimizer
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# OrthoGrad

**Overview**:

The `OrthoGrad` optimizer is a wrapper around any existing optimizer that enforces gradient orthogonality to the current parameter vectors before applying updates. Based on the method introduced in "Grokking at the Edge of Numerical Stability," OrthoGrad projects each gradient onto the subspace orthogonal to its corresponding weight vector, and then rescales it to preserve the original norm. This orthogonalization helps prevent updates that reinforce the current weight direction, which can improve generalization and reduce overfitting in deep networks.

**Parameters**:

- **`base_optimizer`** *(Optimizer, required)*: The underlying optimizer instance (e.g., `Adam`, `SGD`, `AdamW`) that will apply the orthogonalized gradient updates.
- **`name`** *(str, default="orthograd")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.orthograd import OrthoGrad

# Define a base optimizer
base_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Wrap it with OrthoGrad
optimizer = OrthoGrad(base_optimizer=base_opt)

# Build and compile a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# PAdam

**Overview**:

The `PAdam` (Partially Adaptive Momentum Estimation) optimizer is a modification of Adam that interpolates between SGD and Adam updates by applying a partial exponent to the adaptive denominator. Specifically, PAdam raises the second moment term to the power of `p * 2` (where `p` is a hyperparameter in [0,1]), allowing control over the degree of adaptivity. This partial adaptation can improve generalization by avoiding the extreme per-parameter learning rate variance sometimes seen in fully adaptive methods.

**Parameters**:

- **`learning_rate`** *(float, default=1e-1)*: Base step size for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment (mean) estimate.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment (variance) estimate.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`partial`** *(float, default=0.25)*: Exponent applied to the denominator’s square root term (i.e., uses `denom^(2*p)`), controlling the adaptivity.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 regularization; when non-zero, decay is applied to the parameters.
- **`weight_decouple`** *(bool, default=False)*: If True, applies weight decay in a decoupled manner (as in AdamW).
- **`fixed_decay`** *(bool, default=False)*: If True, uses a fixed decay rate rather than scaling by the learning rate.
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Options for gradient clipping by norm or value.
- **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Steps over which gradients are accumulated before updating.
- **`name`** *(str, default="padam")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.padam import PAdam

# Instantiate the PAdam optimizer\ noptimizer = PAdam(
    learning_rate=0.1,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    partial=0.25,
    weight_decay=1e-4,
    weight_decouple=True
)

# Build and compile a model\ nmodel = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# PCGrad

**Overview**:

The `PCGrad` (Projected Conflicting Gradients) method is a gradient surgery technique designed for multi-task learning. It addresses the issue of conflicting gradients—where gradients from different tasks point in opposing directions—by projecting each task’s gradient onto the normal plane of any other task’s gradient with which it conflicts. This projection mitigates gradient interference, leading to more stable and efficient multi-task optimization.

**Parameters**:

- **`reduction`** *(str, default="mean")*: Method to merge non-conflicting gradients. Options:
  - **`"mean"`**: Averages the shared gradient components.
  - **`"sum"`**: Sums the shared gradient components.

**Key Methods**:

- **`pack_grad(tape, losses, variables)`**: Computes and flattens gradients for each task loss.
  - **`tape`**: A `tf.GradientTape` instance (persistent if reused).
  - **`losses`**: List of loss tensors for each task.
  - **`variables`**: List of model variables.
  - **Returns**: Tuple `(grads_list, shapes, has_grads_list)` where:
    - `grads_list` is a list of flattened gradients per task.
    - `shapes` records original variable shapes for unflattening.
    - `has_grads_list` indicates presence of gradients (mask).

- **`project_conflicting(grads, has_grads)`**: Applies gradient surgery across tasks.
  - **`grads`**: List of flattened task gradients.
  - **`has_grads`**: List of masks for existing gradients.
  - **Returns**: A single merged flattened gradient after resolving conflicts.

- **`pc_backward(tape, losses, variables)`**: End-to-end API to compute PCGrad-adjusted gradients.
  - **Returns**: List of unflattened gradients matching `variables`.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.pcgrad import PCGrad

# Define model and tasks
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Instantiate PCGrad
pcgrad = PCGrad(reduction='mean')
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Custom training step with PCGrad
@tf.function
def train_step(x_batch, y_batch_tasks):
    # y_batch_tasks is a list of labels per task
    with tf.GradientTape(persistent=True) as tape:
        losses = [
            tf.keras.losses.sparse_categorical_crossentropy(y, model(x_batch), from_logits=True)
            for y in y_batch_tasks
        ]
    # Compute PCGrad-adjusted gradients
    pc_grads = pcgrad.pc_backward(tape, losses, model.trainable_variables)
    # Apply gradients
    optimizer.apply_gradients(zip(pc_grads, model.trainable_variables))

# Example training loop
for epoch in range(10):
    for x_batch, y_batch_tasks in train_dataset:
        train_step(x_batch, y_batch_tasks)
```

# PPCGrad

**Overview**:

The `PPCGrad` (Parallel Projected Conflicting Gradients) optimizer extends the PCGrad method by using tf.vectorized_map to parallelize the gradient surgery step. PPCGrad identifies and resolves conflicts among task-specific gradients by projecting each gradient onto the normal plane of any other conflicting gradient, similar to PCGrad. However, PPCGrad distributes the projection computation across multiple processes, which can accelerate the gradient adjustment in multi-core environments, especially when dealing with large models or many tasks.

**Parameters**:

- **`reduction`** *(str, default="mean")*: Method to merge non-conflicting gradient components:
  - **`"mean"`**: Averages the shared gradient components across tasks.
  - **`"sum"`**: Sums the shared gradient components across tasks.

**Key Methods**:

- **`pack_grad(tape, losses, variables)`**: Computes and flattens gradients for each task loss.
  - **`tape`**: A `tf.GradientTape` instance (persistent if reused for multiple losses).
  - **`losses`**: List of loss tensors for each task.
  - **`variables`**: List of model variables.
  - **Returns**: `(grads_list, shapes, has_grads_list)`:
    - `grads_list`: Flattened gradients per task.
    - `shapes`: Original shapes for unflattening.
    - `has_grads_list`: Masks indicating presence of gradients.

- **`project_conflicting(grads, has_grads)`**: Parallel gradient surgery using tf.vectorized_map.
  - **`grads`**: List of flattened task gradients.
  - **`has_grads`**: List of masks for existing gradients.
  - **Returns**: Merged flattened gradient after conflict resolution.

- **`pc_backward(tape, losses, variables)`**: End-to-end API to compute PPCGrad-adjusted gradients.
  - **Returns**: List of unflattened gradients matching `variables`.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.pcgrad import PPCGrad

# Define model and tasks\ nmodel = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Instantiate PPCGrad
ppcgrad = PPCGrad(reduction='mean')
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Custom training step with PPCGrad
@tf.function
def train_step(x_batch, y_batch_tasks):
    with tf.GradientTape(persistent=True) as tape:
        losses = [
            tf.keras.losses.sparse_categorical_crossentropy(y, model(x_batch), from_logits=True)
            for y in y_batch_tasks
        ]
    # Compute PPCGrad-adjusted gradients
    ppc_grads = ppcgrad.pc_backward(tape, losses, model.trainable_variables)
    # Apply gradients
    optimizer.apply_gradients(zip(ppc_grads, model.trainable_variables))

# Example training loop\ nfor epoch in range(10):
    for x_batch, y_batch_tasks in train_dataset:
        train_step(x_batch, y_batch_tasks)
```

# PNM

**Overview**:

The `PNM` optimizer implements a predictive negative momentum strategy tailored for deep learning optimization. It extends the conventional momentum approach by maintaining two momentum accumulators—one for positive and one for negative momentum—and alternating their roles every update. At each step, the optimizer computes a noise norm (based on the beta2 parameter) and combines the two momentum estimates to generate a predictive update. This mechanism aims to counteract overshooting and stabilize training, especially in noisy gradient environments. Additionally, PNM supports decoupled weight decay, which can be applied independently of the gradient update.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The base step size for updating parameters.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate applied to momentum; used for the positive momentum update.
- **`beta2`** *(float, default=1.0)*: Hyperparameter influencing the noise normalization in the update; used to compute a noise norm that scales the negative momentum contribution.
- **`epsilon`** *(float, default=1e-8)*: A small constant added for numerical stability to prevent division by zero.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay regularization, applied to parameters.
- **`weight_decouple`** *(bool, default=True)*: If True, applies weight decay in a decoupled manner, similar to AdamW.
- **`fixed_decay`** *(bool, default=False)*: Whether to use a fixed weight decay rate instead of scaling it by the learning rate.
- **`clipnorm`** *(float, optional)*: Clips gradients by norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Whether to maintain an Exponential Moving Average (EMA) of the model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum factor for EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for updating EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation (useful in mixed precision training).
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps over which gradients are accumulated before performing an update.
- **`name`** *(str, default="pnm")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.pnm import PNM

# Instantiate the PNM optimizer
optimizer = PNM(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=1.0,
    epsilon=1e-8,
    weight_decay=1e-2,
    weight_decouple=True,
    fixed_decay=False
)

# Build a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model using the PNM optimizer
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Prodigy

**Overview**:

The `Prodigy` optimizer is an adaptive optimization algorithm that extends traditional adaptive methods by dynamically recalibrating its effective learning rate through an additional scaling mechanism. It maintains extra state variables—such as d (current scaling factor), d0 (initial scaling factor), d_max (maximum observed scaling), and d_hat (a temporary estimate)—to adaptively adjust the update magnitude based on the relationship between parameter changes and gradients. This dynamic scaling, combined with options for decoupled weight decay and bias correction, aims to improve training stability and generalization, especially in the presence of noisy or complex gradients.

**Parameters**:

- **`learning_rate`** *(float, default=1.0)*: The base learning rate for updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`beta3`** *(float, optional)*: Exponential decay rate for additional momentum in scaling; if not provided, it defaults to √(beta2).
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay regularization.
- **`d0`** *(float, default=1e-6)*: Initial scaling factor; used to initialize the dynamic scaling variables.
- **`d_coef`** *(float, default=1.0)*: Multiplier applied to the computed scaling factor.
- **`growth_rate`** *(float, default=`inf`)*: Upper bound on the allowed growth of the scaling factor.
- **`weight_decouple`** *(bool, default=True)*: If True, applies weight decay decoupled from the gradient update (as in AdamW).
- **`fixed_decay`** *(bool, default=False)*: Uses fixed weight decay instead of scaling it by the learning rate.
- **`bias_correction`** *(bool, default=False)*: Whether to apply bias correction when computing the adaptive scaling factor.
- **`safeguard_warmup`** *(bool, default=False)*: If True, enables safeguard mechanisms during initial training steps.
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Options for clipping gradients by norm, by value, or by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to apply an Exponential Moving Average (EMA) to model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients before an update.
- **`name`** *(str, default="prodigy")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.prodigy import Prodigy

# Instantiate the Prodigy optimizer
optimizer = Prodigy(
    learning_rate=1.0,
    beta1=0.9,
    beta2=0.999,
    beta3=None,             # If None, beta3 defaults to sqrt(beta2)
    epsilon=1e-8,
    weight_decay=1e-2,
    d0=1e-6,
    d_coef=1.0,
    growth_rate=float('inf'),
    weight_decouple=True,
    fixed_decay=False,
    bias_correction=False,
    safeguard_warmup=False
)

# Build a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model using the Prodigy optimizer
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# SAM

**Overview**:

The `SAM` optimizer (Sharpness-Aware Minimization) is designed to improve model generalization by seeking parameters that lie in neighborhoods with uniformly low loss. SAM works by first perturbing the model weights in the direction of the gradients (scaled by a hyperparameter ρ) to explore the sharpness of the loss landscape. It then computes the gradients at these perturbed parameters and finally applies the update using an underlying base optimizer. Optional modifications include adaptive scaling of the perturbation and gradient centralization to further stabilize the update. This two-step process (perturb then update) helps to avoid sharp minima and improves robustness against noise.

**Parameters**:

- **`base_optimizer`** *(Optimizer, required)*: The underlying optimizer (e.g., SGD, Adam, AdamW) that performs the parameter updates after the SAM perturbation step.
- **`rho`** *(float, default=0.05)*: The radius defining the neighborhood around the current parameters where the loss is evaluated; it controls the magnitude of the weight perturbation.
- **`adaptive`** *(bool, default=False)*: If True, scales the perturbation by the element-wise absolute value of the weights, leading to adaptive perturbations that account for parameter scale.
- **`use_gc`** *(bool, default=False)*: If True, applies gradient centralization to the gradients (by subtracting the mean) before computing the perturbation.
- **`perturb_eps`** *(float, default=1e-12)*: A small constant added to avoid division by zero when computing the gradient norm for the perturbation step.
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Options to clip gradients by norm or by value to control gradient explosion.
- **`use_ema`** *(bool, default=False)*: Whether to maintain an Exponential Moving Average (EMA) of model weights, which can help improve stability.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for the EMA computation.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in training steps) at which the EMA weights are updated.
- **`loss_scale_factor`** *(float, optional)*: A factor for scaling the loss during gradient computation, useful in mixed precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: The number of steps for which gradients are accumulated before performing an update.
- **`name`** *(str, default="sam")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.sam import SAM

# Define a base optimizer (e.g., Adam)
base_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Wrap the base optimizer with SAM
optimizer = SAM(
    base_optimizer=base_opt,
    rho=0.05,
    adaptive=False,
    use_gc=False,
    perturb_eps=1e-12
)

# Build and compile a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
```

# GSAM

**Overview**:

The `GSAM` optimizer extends the Sharpness-Aware Minimization (SAM) framework by incorporating gradient decomposition into the weight perturbation process. GSAM perturbs the model weights by temporarily zeroing out certain dynamics (e.g., reducing BatchNormalization momentum) and decomposing the gradients into components that are then recombined. This surgery reduces harmful gradient conflicts and encourages convergence toward flatter minima, thereby improving model generalization.

**Parameters**:

- **`model`** *(tf.keras.Model, required)*: The model being optimized. Used for accessing layers (e.g., BatchNormalization) during perturbation.
- **`base_optimizer`** *(Optimizer, required)*: The underlying optimizer (e.g., Adam, SGD) that performs the actual weight updates after SAM perturbation.
- **`rho_scheduler`** *(object, required)*: A scheduler that provides the current perturbation radius (rho) at each update.
- **`alpha`** *(float, default=0.4)*: A weighting factor controlling the contribution of gradient decomposition in the SAM update.
- **`adaptive`** *(bool, default=False)*: If True, scales the perturbation adaptively based on weight magnitudes.
- **`perturb_eps`** *(float, default=1e-12)*: A small constant added to avoid division by zero in the perturbation computation.
- **Standard parameters** such as `clipnorm`, `clipvalue`, `global_clipnorm`, `use_ema`, `ema_momentum`, `ema_overwrite_frequency`, `loss_scale_factor`, and `gradient_accumulation_steps` are also available.
- **`name`** *(str, default="gsam")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.sam import GSAM

# Instantiate a base optimizer
base_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Define a rho scheduler (example: exponential decay)
rho_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.05, decay_steps=1000, decay_rate=0.95)

# Instantiate GSAM optimizer
optimizer = GSAM(
    model=model,
    base_optimizer=base_opt,
    rho_scheduler=rho_scheduler,
    alpha=0.4,
    adaptive=False,
    perturb_eps=1e-12
)

# Compile and train the model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# WSAM

**Overview**:

The `WSAM` (Weighted SAM) optimizer is a variant of SAM that refines the sharpness-aware update by incorporating a corrective term derived from a decomposition of the gradient. WSAM perturbs the weights and then calculates a corrective gradient based on the differences between the original and perturbed weights. Additionally, it synchronizes BatchNormalization momentum during the update to stabilize training. This two-phase update helps guide the model toward flatter regions of the loss landscape.

**Parameters**:

- **`model`** *(tf.keras.Model, required)*: The model to be optimized; used for synchronizing BatchNormalization layers during perturbation.
- **`base_optimizer`** *(Optimizer, required)*: The underlying optimizer that performs the weight update after perturbation.
- **`rho`** *(float, default=0.1)*: Perturbation radius for the SAM update.
- **`k`** *(int, default=10)*: A parameter that controls the frequency of corrective updates in WSAM.
- **`alpha`** *(float, default=0.7)*: Weighting factor for blending the corrective gradient with the original gradient.
- **`adaptive`** *(bool, default=False)*: Whether to use adaptive scaling based on weight magnitudes in the perturbation step.
- **`use_gc`** *(bool, default=False)*: If True, applies Gradient Centralization to the gradients.
- **`perturb_eps`** *(float, default=1e-12)*: A small constant to avoid division by zero in the perturbation computation.
- **Standard parameters** for clipping, EMA, and gradient accumulation are also supported.
- **`name`** *(str, default="wsam")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.sam import WSAM

# Instantiate a base optimizer
base_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Instantiate WSAM optimizer
optimizer = WSAM(
    model=model,
    base_optimizer=base_opt,
    rho=0.1,
    k=10,
    alpha=0.7,
    adaptive=False,
    use_gc=False,
    perturb_eps=1e-12
)

# Compile the model using WSAM
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# BSAM

**Overview**:

The `BSAM` (Batch Sharpness-Aware Minimization) optimizer employs a three-step update process designed to further reduce sharpness in the weight space. First, BSAM injects controlled random noise into the weights based on the dataset size. In the second step, it computes an intermediate update by adjusting for the noisy gradient. Finally, a momentum-based correction refines the updates. This layered approach aims to produce more robust and stable training, particularly useful when handling large datasets with noisy gradient estimates.

**Parameters**:

- **`num_data`** *(int, required)*: The total number of training samples; used to scale the injected noise in the first step.
- **`lr`** *(float, default=0.5)*: The learning rate for the momentum update step.
- **`beta1`** *(float, default=0.9)*: Decay rate for the momentum accumulator.
- **`beta2`** *(float, default=0.999)*: Decay rate for the secondary gradient accumulation for variance estimation.
- **`weight_decay`** *(float, default=1e-4)*: Coefficient for weight decay regularization.
- **`rho`** *(float, default=0.05)*: Perturbation strength used during the second update phase.
- **`adaptive`** *(bool, default=False)*: Whether to apply adaptive scaling based on weight magnitudes.
- **`damping`** *(float, default=0.1)*: Damping factor for smoothing the momentum update in the third step.
- **Standard parameters** such as gradient clipping, EMA settings, etc., are also supported.
- **`name`** *(str, default="bsam")*: Name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.sam import BSAM

# Instantiate the BSAM optimizer
optimizer = BSAM(
    num_data=50000,
    lr=0.5,
    beta1=0.9,
    beta2=0.999,
    weight_decay=1e-4,
    rho=0.05,
    adaptive=False,
    damping=0.1
)

# Build a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model using BSAM
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# LookSAM

**Overview**:  
The `LookSAM` optimizer extends Sharpness‑Aware Minimization (SAM) by performing the costly perturbation and retraction steps only every *k* iterations, trading off a small delay in the sharpness correction for substantial compute savings. It first “looks” ahead by perturbing parameters along the gradient direction scaled by ρ, then upon the next *k*th step it “retracts” to the original weights and applies a combined direction that mixes the old and new gradient components by factor α; this retains most of SAM’s generalization benefits at a fraction of the extra cost.

**Parameters**:

- **`base_optimizer`** *(tf.keras.optimizers.Optimizer, required)*: Underlying optimizer to which orthogonalized updates are delegated.  
- **`rho`** *(float, default=0.1)*: Neighborhood radius for sharpness perturbation.  
- **`k`** *(int, default=10)*: Interval (in steps) at which to perform the SAM perturbation/retraction cycle.  
- **`alpha`** *(float, default=0.7)*: Mixing coefficient for the gradient decomposition on non‑perturbation steps.  
- **`adaptive`** *(bool, default=False)*: Scale perturbations per‑parameter by \|w\| to obtain ASAM‑style updates.  
- **`use_gc`** *(bool, default=False)*: Whether to apply Gradient Centralization before perturbation.  
- **`perturb_eps`** *(float, default=1e-12)*: Small epsilon added when normalizing gradients to avoid division by zero.  
- **`step`** *(bool, default=True)*: If `True`, uses `base_optimizer.iterations` to decide when `k` divides; otherwise always performs first‑step perturbations.  
- **`name`** *(str, default="looksam")*: Name identifier for this optimizer instance.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.looksam import LookSAM

# 1. Define your base optimizer
base_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

# 2. Wrap it with LookSAM
opt = LookSAM(
    base_optimizer=base_opt,
    rho=0.1,
    k=10,
    alpha=0.7,
    adaptive=False,
    use_gc=False
)

# 3. Compile and train your model as usual
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10),
])
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# SCION

**Overview**:  
The `SCION` optimizer integrates norm-based gradient rescaling with an optional momentum mechanism and constraint application. It constructs a norm projector via a configurable norm type (e.g., AUTO, SPECTRAL, SIGN, etc.) to process each gradient before applying an update. Optionally, it uses a momentum update by blending the current gradient with a running scalar (tracked in a per-variable variable \(d\)). Additionally, when constraints are enabled, SCION scales down the weights prior to the update to enforce stability. These mechanisms help preserve the structure of the weights and stabilize training when gradients are noisy.

**Parameters**:  
- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.  
- **`weight_decay`** *(float, default=0.0)*: Weight decay coefficient applied to the parameters.  
- **`momentum`** *(float, default=0.1)*: Factor for momentum-based smoothing; a running scalar \(d\) is updated as a convex combination of its previous value and the current gradient.  
- **`constraint`** *(bool, default=False)*: If set to True, applies an additional pre-update constraint by scaling the weights.  
- **`norm_type`** *(LMONorm, default=LMONorm.AUTO)*: Specifies the type of norm-based projection to apply (see LMONorm enum for options such as AUTO, SPECTRAL, SIGN, etc.).  
- **`norm_kwargs`** *(dict, optional)*: Additional keyword arguments for building the norm projector.  
- **`scale`** *(float, default=1.0)*: Factor by which the normalized gradient is scaled before updating the weights.  
- **`weight_decouple`** *(bool, default=True)*: If True, applies weight decay separately from the gradient update.  
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Options for clipping gradients by norm, value, or global norm, respectively.  
- **`use_ema`** *(bool, default=False)*: Whether to use an Exponential Moving Average (EMA) of the model weights.  
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.  
- **`ema_overwrite_frequency`** *(int, optional)*: The frequency (in steps) at which EMA weights are updated.  
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.  
- **`gradient_accumulation_steps`** *(int, optional)*: The number of steps to accumulate gradients before an update.  
- **`name`** *(str, default="scion")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.scion import SCION

# Instantiate the SCION optimizer
optimizer = SCION(
    learning_rate=1e-3,
    weight_decay=0.0,
    momentum=0.1,
    constraint=False,
    norm_kwargs={},  # Additional parameters for the norm projector can be provided here.
    scale=1.0,
    weight_decouple=True
)

# Build a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model with SCION optimizer
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# SCIONLight

**Overview**:  
The `SCIONLight` optimizer is a streamlined version of SCION that applies norm-based gradient scaling without an explicit momentum mechanism. It uses a learned norm operator—constructed via a selectable norm type—to process the gradients and directly computes the update from the normalized gradients scaled by a factor \(scale\). Optionally, it applies a constraint by scaling down the weights before the update and supports decoupled weight decay. SCIONLight is well-suited for cases where a simpler normalization-based update is preferred.

**Parameters**:  
- **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.  
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay, if used.  
- **`momentum`** *(float, default=0.1)*: A value that is available for potential scaling of the gradient, although SCIONLight primarily uses direct gradient normalization.  
- **`constraint`** *(bool, default=False)*: If True, applies a weight constraint by scaling the weights down before the update.  
- **`norm_type`** *(LMONorm, default=LMONorm.AUTO)*: Specifies the type of norm projector to use (e.g., AUTO, SPECTRAL, SIGN, etc.).  
- **`norm_kwargs`** *(dict, optional)*: Additional keyword arguments used in constructing the norm projector.  
- **`scale`** *(float, default=1.0)*: Factor by which the normalized gradient is scaled before it is applied as an update.  
- **`weight_decouple`** *(bool, default=True)*: If True, weight decay is applied separately from the gradient update.  
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Options for gradient clipping by norm, value, or by global norm.  
- **`use_ema`** *(bool, default=False)*: Whether to maintain an Exponential Moving Average (EMA) of model weights.  
- **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA updates.  
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for EMA weight updates.  
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling loss during gradient computations.  
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before performing an update.  
- **`name`** *(str, default="scionlight")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.scion import SCIONLight

# Instantiate the SCIONLight optimizer
optimizer = SCIONLight(
    learning_rate=1e-3,
    weight_decay=0.0,
    momentum=0.1,
    constraint=False,
    norm_kwargs={},  # Optionally pass norm-specific arguments
    scale=1.0,
    weight_decouple=True
)

# Build a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model with SCIONLight
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# SM3

**Overview**:  
SM3 is a memory‑efficient adaptive optimizer that retains the per‑parameter adaptivity of methods like Adam while dramatically reducing optimizer state memory by factorizing the second‑moment estimates across tensor dimensions. citeturn17view0 SM3 maintains separate accumulators for each dimension (e.g., rows and columns) of parameter tensors instead of a full‑size accumulator per parameter. Despite this compression, SM3 matches or surpasses the convergence behavior of full‑memory optimizers on large‑scale language and vision tasks citeturn25search8.

**Parameters**:
- **`learning_rate`** *(float, default=1e-1)*: Base step size for updates.  
- **`epsilon`** *(float, default=1e-30)*: Small constant for numerical stability in denominator.  
- **`momentum`** *(float, default=0.0)*: Momentum coefficient for update smoothing.  
- **`beta`** *(float, default=0.0)*: Exponential decay rate for accumulator updates.  
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Gradient clipping thresholds.  
- **`use_ema`** *(bool, default=False)*: Whether to maintain an exponential moving average of parameters for evaluation.  
- **`ema_momentum`** *(float, default=0.99)*: Decay rate for parameter EMA.  
- **`ema_overwrite_frequency`** *(int, optional)*: Steps between overwriting model weights with EMA.  
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling loss in mixed‑precision training.  
- **`gradient_accumulation_steps`** *(int, optional)*: Steps over which gradients are accumulated before applying an update.  
- **`name`** *(str, default="sm3")*: Identifier for the optimizer instance.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.sm3 import SM3

# Instantiate SM3 optimizer
optimizer = SM3(
    learning_rate=0.1,
    epsilon=1e-30,
    momentum=0.9,
    beta=0.5
)

# Compile a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(1024,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```

# SophiaH

**Overview**  
The `SophiaH` optimizer is a scalable, stochastic second‑order optimizer based on the **Sophia** method, extended with a Hutchinson estimator for the Hessian diagonal. It maintains moving averages of both gradients and Hessian estimates, applies these as pre‑conditioners, and uses element‑wise clipping to control update magnitudes. Hutchinson’s method is run every `update_period` iterations, balancing second‑order information with computational cost citeturn0search8.

**Parameters**:
- **`learning_rate`** *(float, default=6e-2)*: Base learning rate for parameter updates.
- **`beta1`** *(float, default=0.96)*: Exponential decay rate for the first moment (gradient) estimates.
- **`beta2`** *(float, default=0.99)*: Exponential decay rate for the second moment (Hessian) estimates.
- **`epsilon`** *(float, default=1e-12)*: Small constant added to denominators for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay (L2 regularization).
- **`weight_decouple`** *(bool, default=True)*: Whether to apply decoupled weight decay (as in AdamW).
- **`fixed_decay`** *(bool, default=False)*: If `True`, uses a fixed weight decay independent of the learning rate.
- **`p`** *(float, default=1e-2)*: Scaling factor for Hutchinson’s diagonal estimator.
- **`update_period`** *(int, default=10)*: Number of iterations between Hessian estimator updates.
- **`num_samples`** *(int, default=1)*: Number of random vectors used per Hutchinson estimate.
- **`hessian_distribution`** *(str, default="gaussian")*: Distribution for Hutchinson random vectors: `"gaussian"` or `"rademacher"`.
- **`clipnorm`** *(float, optional)*: Clip gradients by norm.
- **`clipvalue`** *(float, optional)*: Clip gradients by value.
- **`global_clipnorm`** *(float, optional)*: Clip gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Whether to maintain an exponential moving average of weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for the EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) to overwrite model weights with EMA weights.
- **`loss_scale_factor`** *(float, optional)*: Factor for scaling loss when computing gradients.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before applying an update.
- **`name`** *(str, default="sophia")*: Name identifier for the optimizer instance.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.sophia import SophiaH

# Define model and loss
model = tf.keras.Sequential([...])
loss_fn = tf.keras.losses.MeanSquaredError()

# Instantiate the SophiaH optimizer
optimizer = SophiaH(
    learning_rate=0.06,
    beta1=0.96,
    beta2=0.99,
    epsilon=1e-12,
    weight_decay=1e-2,
    update_period=20,
    num_samples=2,
    hessian_distribution='rademacher'
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

# SRMM

**Overview**:

The **SRMM** (Stochastic Regularized Majorization–Minimization) optimizer implements the algorithm introduced in “Stochastic regularized majorization‑minimization with weakly convex and multi‑convex surrogates.” It maintains moving averages of both gradients and parameters, weighting them by a decaying factor that depends on the iteration count and a tunable memory length. This approach blends new gradient information with historical parameter values, resulting in smoother parameter trajectories and improved convergence properties in nonconvex settings.

**Parameters**:
- **`learning_rate`** *(float, default=0.01)*: Base step size for parameter updates.  
- **`beta`** *(float, default=0.5)*: Exponent that controls how quickly past information is discounted when computing the moving averages.  
- **`memory_length`** *(int or None, default=100)*: The period over which past gradients and parameters are mixed. When `None`, behaves like standard momentum.  
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Gradient clipping thresholds.  
- **`use_ema`** *(bool, default=False)*: Whether to maintain an exponential moving average of the model weights for evaluation.  
- **`ema_momentum`** *(float, default=0.99)*: Momentum factor for the EMA.  
- **`ema_overwrite_frequency`** *(int, optional)*: How often to overwrite the EMA weights with the current weights.  
- **`loss_scale_factor`** *(float, optional)*: Multiplier for loss scaling in mixed‑precision training.  
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before applying an update.  
- **`name`** *(str, default="srmm")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.srmm import SRMM

# Instantiate SRMM optimizer
optimizer = SRMM(
    learning_rate=1e-2,
    beta=0.4,
    memory_length=50
)

# Build and compile a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```

# AdaBelief (parallel)

**Overview**:

The `AdaBelief` optimizer is a variant of Adam that adapts the learning rate by tracking the "belief" in observed gradients (i.e., the deviation of gradients from their exponential moving average). This results in more responsive updates that can improve convergence stability and generalization. Additionally, `AdaBelief` uses Python's `multiprocessing.Manager` to manage optimizer state across worker processes, enabling parallel parameter updates.

**Parameters**:
- **`learning_rate`** *(float, default=1e-3)*: Base learning rate.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-16)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: L2 penalty coefficient applied either decoupled or via gradient addition.
- **`amsgrad`** *(bool, default=False)*: Enables AMSGrad; tracks the maximum of past second moments.
- **`decoupled_decay`** *(bool, default=True)*: Uses AdamW-style decoupled weight decay.
- **`fixed_decay`** *(bool, default=False)*: Applies a fixed decay factor rather than scaling by the learning rate.
- **`rectify`** *(bool, default=True)*: Enables RAdam rectification of the adaptive learning rate.
- **`degenerated_to_sgd`** *(bool, default=True)*: Falls back to SGD-like updates when variance is low.
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Gradient clipping thresholds.
- **`use_ema`**, **`ema_momentum`**, **`ema_overwrite_frequency`**: Controls weight EMA for evaluation.
- **`loss_scale_factor`** *(float, optional)*: Multiplicative factor for loss scaling in mixed precision.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before applying an update.
- **`name`** *(str, default="adabelief")*: Name of the optimizer instance.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.parallel.adabelief import AdaBelief

# Create AdaBelief optimizer
opt = AdaBelief(
    learning_rate=1e-3,
    weight_decay=1e-2,
    amsgrad=True,
    rectify=True,
    decoupled_decay=True
)

# Compile a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    run_eagerly=True
)

# Train with standard or multiprocessing-based data loaders
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AdaBoundW (parallel)

**Overview**:

The `AdaBoundW` optimizer is a variant of AdamW that incorporates dynamic bounds on the learning rate, smoothly transitioning from an adaptive optimizer to SGD over time. By tracking both the first and second moments of gradients, it applies parameter-specific step sizes, while weight decay is handled in a decoupled fashion. Additionally, `AdaBoundW` uses Python's `multiprocessing.Manager` to manage optimizer state across worker processes, enabling parallel parameter updates.

**Parameters**:
- **`learning_rate`** *(float, default=1e-3)*: Base learning rate for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment (mean) estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment (variance) estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant added for numerical stability when normalizing by the second moment.
- **`weight_decay`** *(float, default=0)*: Coefficient for decoupled L2 weight decay (AdamW-style).
- **`final_lr`** *(float, default=0.1)*: Target learning rate that the adaptive bounds will converge toward.
- **`gamma`** *(float, default=1e-3)*: Convergence speed of the lower and upper bound functions.
- **`amsbound`** *(bool, default=False)*: If `True`, uses the AMSBound variant to track the maximum of past squared gradients.
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Thresholds for gradient clipping to prevent exploding gradients.
- **`use_ema`** *(bool, default=False)*: Whether to maintain an exponential moving average (EMA) of model weights for evaluation.
- **`ema_momentum`** *(float, default=0.99)*: Smoothing factor for the EMA computation.
- **`ema_overwrite_frequency`** *(int, optional)*: Number of steps between EMA weight overwrites.
- **`loss_scale_factor`** *(float, optional)*: Factor by which to scale the loss (useful for mixed-precision training).
- **`gradient_accumulation_steps`** *(int, optional)*: Number of mini-batches to accumulate gradients over before applying an update.
- **`name`** *(str, default="adaboundw")*: Name identifier for the optimizer instance.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.parallel.adaboundw import AdaBoundW

# Instantiate the optimizer with bounds converging to SGD
opt = AdaBoundW(
    learning_rate=2e-4,
    final_lr=0.05,
    gamma=5e-4,
    weight_decay=1e-2,
    amsbound=True,
    clipnorm=1.0,
    use_ema=True,
    ema_momentum=0.995
)

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model with AdaBoundW
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    run_eagerly=True
)

# Train the model, potentially across multiple processes
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20
)
```

# Adalite (parallel)

**Overview**:

The `Adalite` optimizer is a hybrid adaptive optimizer that integrates ideas from Adam, AdaBelief, LAMB, and Adafactor. It adapts the learning rate by tracking first and second moment estimates of gradients like Adam, while employing a trust ratio mechanism (inspired by LAMB) to scale updates based on the ratio of parameter norms to gradient norms. For high-dimensional parameters, it uses factorized second moment estimation (like Adafactor) to reduce memory requirements by maintaining separate row and column statistics. Additionally, `Adalite` uses Python's `multiprocessing.Manager` to manage optimizer state across worker processes, enabling parallel parameter updates in multi-process training setups.

**Parameters**:
- **`learning_rate`** *(float, default=1e-3)*: Base learning rate for parameter updates.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment (mean) estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment (variance) estimates.
- **`weight_decay`** *(float, default=1e-2)*: L2 regularization coefficient applied to weights.
- **`weight_decouple`** *(bool, default=False)*: If `True`, applies weight decay in a decoupled manner (AdamW-style).
- **`fixed_decay`** *(bool, default=False)*: If `True`, uses a fixed decay factor independent of the learning rate.
- **`g_norm_min`** *(float, default=1e-10)*: Minimum gradient norm to prevent division by zero when computing trust ratios.
- **`ratio_min`** *(float, default=1e-4)*: Minimum trust ratio to avoid overly small update scaling.
- **`tau`** *(float, default=1.0)*: Temperature parameter for the softmax computation of importance weights in factorized updates.
- **`eps1`** *(float, default=1e-6)*: Small epsilon added in the denominator for normalization by the square root of the second moment.
- **`eps2`** *(float, default=1e-10)*: Small epsilon for numerical stability in sum-based operations (e.g., factorized variance sums).
- **`clipnorm`** *(float, optional)*: Maximum norm for gradient clipping per variable.
- **`clipvalue`** *(float, optional)*: Maximum absolute value for gradient clipping per variable.
- **`global_clipnorm`** *(float, optional)*: Maximum global norm across all gradients for clipping.
- **`use_ema`** *(bool, default=False)*: Whether to maintain an exponential moving average (EMA) of the model weights for evaluation or inference.
- **`ema_momentum`** *(float, default=0.99)*: Momentum factor for the weight EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Number of steps between overwriting model weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss computations in mixed-precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of gradient accumulation steps before performing an optimizer update.
- **`name`** *(str, default="adalite")*: Name of this optimizer instance.
- **`**kwargs`**: Additional keyword arguments forwarded to the base `optimizer.Optimizer` class.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.parallel.adalite import Adalite

# Instantiate the Adalite optimizer
opt = Adalite(
    learning_rate=3e-4,
    beta1=0.9,
    beta2=0.999,
    weight_decay=1e-2,
    weight_decouple=True,
    fixed_decay=False,
    tau=0.5
)

# Build and compile a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    run_eagerly=True
)

# Train with a parallel data pipeline
model.fit(train_dataset, validation_data=val_dataset, epochs=5)
```

# AdaMod (parallel)

**Overview**:

The `AdaMod` optimizer is an enhanced variant of Adam that introduces a momentum-based bound on the effective learning rates, preventing sudden large updates and improving training stability. It maintains exponential moving averages of gradients (`exp_avg`), squared gradients (`exp_avg_sq`), and adaptive learning rates (`exp_avg_lr`) to constrain step sizes. Additionally, `AdaMod` uses Python's `multiprocessing.Manager` to share optimizer state across multiple processes for parallel training.

**Parameters**:
- **`learning_rate`** *(float, default=1e-3)*: Base learning rate for updating parameters.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates (mean of gradients).
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates (variance of gradients).
- **`beta3`** *(float, default=0.999)*: Exponential decay rate for bounding the adaptive learning rates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability when dividing by the root of the second moment.
- **`weight_decay`** *(float, default=0)*: L2 regularization coefficient applied to parameters.
- **`clipnorm`** *(float, optional)*: Maximum norm for clipping gradients per variable.
- **`clipvalue`** *(float, optional)*: Maximum absolute value for clipping gradients per variable.
- **`global_clipnorm`** *(float, optional)*: Maximum global norm across all gradients for clipping.
- **`use_ema`** *(bool, default=False)*: Whether to maintain an exponential moving average of the model weights for evaluation.
- **`ema_momentum`** *(float, default=0.99)*: Momentum factor for the weight EMA.
- **`ema_overwrite_frequency`** *(int, optional)*: Number of steps between overwriting model weights with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss in mixed-precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before applying an update.
- **`name`** *(str, default="adamod")*: Name identifier for this optimizer instance.
- **`**kwargs`**: Additional keyword arguments forwarded to the base `optimizer.Optimizer` class.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.parallel.adamod import AdaMod

# Instantiate the AdaMod optimizer
opt = AdaMod(
    learning_rate=2e-4,
    beta1=0.9,
    beta2=0.999,
    beta3=0.999,
    epsilon=1e-8,
    weight_decay=1e-4
)

# Build and compile a neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    run_eagerly=True
)

# Train with a parallel data pipeline
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AdamP (parallel)

**Overview**:

The `AdamP` optimizer is a variant of Adam designed to reduce the increase of weight norm by projecting the update direction away from the parameter vector when they are too aligned. This helps maintain generalization performance, especially in convolutional models. The optimizer implements the method proposed in the paper [*Slowing Down the Weight Norm Increase in Momentum-based Optimizers*](https://arxiv.org/abs/2006.08217). Additionally, `AdamP` uses Python's `multiprocessing.Manager` to share optimizer state across multiple processes for parallel training.

**Parameters**:
- **`learning_rate`** *(float, default=1e-3)*: Base learning rate.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant added to denominator for numerical stability.
- **`weight_decay`** *(float, default=0)*: Decoupled weight decay coefficient.
- **`delta`** *(float, default=0.1)*: Threshold value for projection condition; controls how "aligned" gradients and weights can be before projection is applied.
- **`wd_ratio`** *(float, default=0.1)*: Weight decay ratio to apply during projection adjustment.
- **`nesterov`** *(bool, default=False)*: Whether to apply Nesterov momentum-style updates.
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Gradient clipping thresholds for stability.
- **`use_ema`**, **`ema_momentum`**, **`ema_overwrite_frequency`**: Controls exponential moving average (EMA) for weights, typically used during evaluation.
- **`loss_scale_factor`** *(float, optional)*: Factor to scale the loss, useful for mixed precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before applying an update.
- **`name`** *(str, default="adamp")*: Name identifier for the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.parallel.adamp import AdamP

# Create AdamP optimizer
opt = AdamP(
    learning_rate=1e-3,
    weight_decay=1e-2,
    delta=0.1,
    wd_ratio=0.1,
    nesterov=True
)

# Build a sample model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
])

# Compile the model with AdamP
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    run_eagerly=True
)

# Fit with standard or multiprocessing-ready dataset
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# RAdam (parallel)

**Overview**:

The `RAdam` optimizer is a variant of Adam that introduces a rectification term to explicitly reduce the variance of the adaptive learning rate in the early stages of training, addressing the instability of vanilla Adam updates and thereby improving convergence stability and generalization across tasks. Additionally, `RAdam` uses Python's `multiprocessing.Manager` to share optimizer state across multiple processes for parallel training.

**Parameters**:
- **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates.  
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment (mean) estimates.  
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment (variance) estimates.  
- **`epsilon`** *(float, default=1e-8)*: Small constant added to the denominator for numerical stability.  
- **`weight_decay`** *(float, default=0)*: L2 penalty coefficient applied directly to the weights each step.  
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Thresholds for per‑gradient or global gradient clipping.  
- **`use_ema`** *(bool, default=False)*: Whether to maintain an exponential moving average of weights for evaluation.  
- **`ema_momentum`** *(float, default=0.99)*: Decay rate for weight EMA when `use_ema=True`.  
- **`ema_overwrite_frequency`** *(int, optional)*: Number of steps between EMA overwrites.  
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss in mixed‑precision training.  
- **`gradient_accumulation_steps`** *(int, optional)*: Number of mini‑batches over which to accumulate gradients before applying an update.  
- **`name`** *(str, default="radam")*: Name given to this optimizer instance.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.parallel.radam import RAdam

# Instantiate RAdam optimizer
opt = RAdam(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=1e-2,
    clipnorm=1.0,
    use_ema=True,
    ema_momentum=0.999
)

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile with RAdam
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    run_eagerly=True
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```

# AdaBound (parallel)

**Overview**:

The `AdaBound` optimizer is an adaptive learning rate method that dynamically bounds the learning rate between a lower and upper bound, transitioning smoothly from adaptive methods like Adam to SGD over time. This approach aims to combine the fast convergence of adaptive methods with the superior generalization performance of SGD. Additionally, `AdaBound` uses Python's `multiprocessing.Manager` to share optimizer state across multiple processes for parallel training.

**Parameters**:
- **`learning_rate`** *(float, default=1e-3)*: Initial learning rate.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0)*: L2 penalty coefficient applied to the weights.
- **`final_lr`** *(float, default=0.1)*: Final (SGD-like) learning rate towards which the optimizer transitions.
- **`gamma`** *(float, default=1e-3)*: Convergence speed of the bound functions.
- **`amsbound`** *(bool, default=False)*: Enables AMSBound variant, which maintains the maximum of past squared gradients.
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float, optional)*: Gradient clipping thresholds.
- **`use_ema`**, **`ema_momentum`**, **`ema_overwrite_frequency`**: Controls exponential moving average (EMA) of weights for evaluation.
- **`loss_scale_factor`** *(float, optional)*: Multiplicative factor for loss scaling in mixed precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before applying an update.
- **`name`** *(str, default="adabound")*: Name of the optimizer instance.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.parallel.adabound import AdaBound

# Create AdaBound optimizer
opt = AdaBound(
    learning_rate=1e-3,
    final_lr=0.1,
    gamma=1e-3,
    weight_decay=1e-4,
    amsbound=True
)

# Compile a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    run_eagerly=True
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
``` 

# SOAP

**Overview**:

SOAP (“ShampoO with Adam in the Preconditioner’s eigenbasis”) is motivated by the formal equivalence between Shampoo (with ½ power) and Adafactor run in Shampoo’s eigenbasis. Shampoo delivers superior preconditioning by capturing second-order curvature via Kronecker-factored statistics, but at the cost of extra hyperparameters and compute. SOAP alleviates this by updating the second-moment running average continuously—just like Adam—but performing those updates in Shampoo’s current eigenbasis, avoiding repeated expensive eigendecompositions. The result is an algorithm that matches Shampoo’s superior convergence while approaching Adam’s efficiency, introducing only the “precondition_frequency” hyperparameter beyond Adam’s standard set.

**Parameters**:
- **`learning_rate`** *(float, default=3e-3)*: Base step size for parameter updates.
- **`beta1`** *(float, default=0.95)*: Exponential decay rate for the first-moment (mean) estimate.
- **`beta2`** *(float, default=0.95)*: Exponential decay rate for the second-moment (variance) estimate.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in denominator.
- **`weight_decay`** *(float, default=1e-2)*: Coefficient for decoupled weight decay (AdamW style).
- **`shampoo_beta`** *(float or None, default=None)*: Decay rate for Shampoo’s preconditioner; if `None`, uses `beta2`.
- **`precondition_frequency`** *(int, default=10)*: Number of steps between full eigendecomposition updates of the preconditioner.
- **`max_precondition_dim`** *(int, default=10000)*: Maximum dimension for which to apply full matrix preconditioning; larger dims use diagonal fallback.
- **`merge_dims`** *(bool, default=False)*: Whether to collapse small tensor dimensions before preconditioning to reduce cost.
- **`precondition_1d`** *(bool, default=False)*: Enable 1D parameter preconditioning when dimension ≤ `max_precondition_dim`.
- **`correct_bias`** *(bool, default=True)*: Apply bias-correction factors as in Adam.
- **`normalize_gradient`** *(bool, default=False)*: Renormalize gradient magnitude after projection.
- **`data_format`** *(str, default='channels_last')*: Input format for convolutional weights; affects reshape/transpose in preconditioning.
- **`clipnorm`** *(float or None)*: Clip gradients to a maximum L2‐norm.
- **`clipvalue`** *(float or None)*: Clip gradients to a maximum absolute value.
- **`global_clipnorm`** *(float or None)*: Clip all gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Maintain an exponential moving average of weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for weight EMA.
- **`ema_overwrite_frequency`** *(int or None)*: Steps between overwriting model weights with EMA weights.
- **`loss_scale_factor`** *(float or None)*: Scale applied to loss for mixed-precision training.
- **`gradient_accumulation_steps`** *(int or None)*: Number of steps to accumulate gradients before applying update.
- **`name`** *(str, default="soap")*: Optional name for the optimizer instance.

---

**Example Usage**:
```python
import tensorflow as tf
from optimizers.soap import SOAP

# Instantiate the SOAP optimizer
optimizer = SOAP(
    learning_rate=3e-3,
    weight_decay=1e-2,
    precondition_frequency=20,
    merge_dims=True,
)

# Compile a Keras model with SOAP
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=5)
```  

# TAM

**Overview**:

Torque‑Aware Momentum (TAM) modifies classical SGD with momentum by computing a smoothed cosine similarity between the new gradient and the previous momentum, then using this “correlation” as a damping factor to down‑weight misaligned gradients and preserve exploration along dominant directions.  By treating momentum as velocity and gradients as forces in a mechanical analogy, TAM applies anisotropic friction to reduce the impact of “torqued” (i.e. misaligned) gradients, leading to more consistent parameter updates and better escape from sharp minima.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: Base step size for updates.
- **`epsilon`** *(float, default=1e-8)*: Small constant to ensure non‑zero update even when damping factor is zero.
- **`weight_decay`** *(float, default=0.0)*: Decoupled weight‑decay coefficient (applied before or after update according to `weight_decouple` and `fixed_decay`).
- **`momentum`** *(float, default=0.9)*: Classical momentum coefficient β controlling the inertia of past updates.
- **`decay_rate`** *(float, default=0.9)*: Smoothing factor γ for the running average of gradient–momentum correlation.
- **`weight_decouple`** *(bool, default=True)*: If True, applies decoupled weight decay (AdamW style); else integrates decay into gradient.
- **`fixed_decay`** *(bool, default=False)*: If True, uses a fixed weight‑decay independent of learning rate; otherwise scales decay by `learning_rate`.
- **`clipnorm`** *(float or None)*: Clip gradients to a maximum L2‐norm.
- **`clipvalue`** *(float or None)*: Clip gradients to a maximum absolute value.
- **`global_clipnorm`** *(float or None)*: Clip all gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Maintain an exponential moving average of weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for weight EMA.
- **`ema_overwrite_frequency`** *(int or None)*: Steps between overwriting model weights with EMA weights.
- **`loss_scale_factor`** *(float or None)*: Scale applied to loss for mixed-precision training.
- **`gradient_accumulation_steps`** *(int or None)*: Number of steps to accumulate gradients before applying update.
- **`name`** *(str, default="soap")*: Optional name for the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.tam import TAM

# Create TAM optimizer
optimizer = TAM(
    learning_rate=5e-4,
    momentum=0.95,
    decay_rate=0.8,
    weight_decay=1e-2,
    weight_decouple=True,
    fixed_decay=False,
    name="tam_optimizer"
)

# Compile and train a model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

---

# AdaTAM

**Overview**:

AdaTAM extends TAM by incorporating Adam‑style first (`β₁`) and second (`β₂`) moment estimates of the (damped) gradient, allowing per‑parameter adaptive learning rates on top of torque‑aware damping.  This combination preserves Adam’s fast convergence in noisy settings while benefiting from TAM’s stability against misaligned gradients, leading to improved generalization and robustness across vision and NLP tasks.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: Base step size.
- **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Decoupled weight‑decay coefficient (applied before or after update according to `weight_decouple` and `fixed_decay`).
- **`decay_rate`** *(float, default=0.9)*: Smoothing factor γ for TAM’s correlation term.
- **`weight_decouple`** *(bool, default=True)*: If True, applies decoupled weight decay (AdamW style); else integrates decay into gradient.
- **`fixed_decay`** *(bool, default=False)*: If True, uses a fixed weight‑decay independent of learning rate; otherwise scales decay by `learning_rate`.
- **`clipvalue`** *(float or None)*: Clip gradients to a maximum absolute value.
- **`global_clipnorm`** *(float or None)*: Clip all gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Maintain an exponential moving average of weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for weight EMA.
- **`ema_overwrite_frequency`** *(int or None)*: Steps between overwriting model weights with EMA weights.
- **`loss_scale_factor`** *(float or None)*: Scale applied to loss for mixed-precision training.
- **`gradient_accumulation_steps`** *(int or None)*: Number of steps to accumulate gradients before applying update.
- **`name`** *(str, default="soap")*: Optional name for the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.tam import AdaTAM

# Create AdaTAM optimizer
optimizer = AdaTAM(
    learning_rate=2e-4,
    beta1=0.9,
    beta2=0.999,
    decay_rate=0.85,
    weight_decay=5e-3,
    weight_decouple=True,
    fixed_decay=True,
    name="adatam_optimizer"
)

# Compile and train with AdaTAM
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_dataset, validation_data=val_dataset, epochs=8)
```

# Ranger21

**Overview**:

The `Ranger21` optimizer integrates a suite of advanced techniques into a single, unified update rule built on top of AdamW (or optionally MadGrad). It combines adaptive gradient clipping, gradient centralization & normalization, positive‑negative momentum, stable weight decay, norm‑based loss regularization, linear warm‑up & explore‑exploit scheduling, lookahead, Softplus smoothing of denominators, and corrected Adam denominators to deliver robust, well‑conditioned optimization across a wide variety of deep‑learning tasks.

**Parameters**:

- **`num_iterations`** *(int)*: Total number of training iterations for scheduling warm‑up, warm‑down, and lookahead cycles.
- **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates.
- **`epsilon`** *(float, default=1e-8)*: Small constant added for numerical stability in denominator computations.
- **`weight_decay`** *(float, default=1e-4)*: Coefficient for decoupled weight‑decay (as in AdamW).
- **`beta0`** *(float, default=0.9)*: Coefficient for positive–negative momentum combination baseline.
- **`betas`** *((float, float), default=(0.9, 0.999))*: Exponential decay rates for first and second moment estimates.
- **`use_softplus`** *(bool, default=True)*: Whether to apply a Softplus transform to the denominator (improves stability).
- **`beta_softplus`** *(float, default=50.0)*: Sharpness parameter for the Softplus transform when `use_softplus=True`.
- **`disable_lr_scheduler`** *(bool, default=False)*: If True, disables linear warm‑up and warm‑down scheduling.
- **`num_warm_up_iterations`** *(int, optional)*: Number of iterations for linear learning‑rate warm‑up. Auto‑computed if None.
- **`num_warm_down_iterations`** *(int, optional)*: Number of iterations for linear learning‑rate warm‑down. Auto‑computed if None.
- **`warm_down_min_lr`** *(float, default=3e-5)*: Final minimum learning rate at end of warm‑down phase.
- **`agc_clipping_value`** *(float, default=1e-2)*: Maximum ratio of gradient norm to parameter norm for Adaptive Gradient Clipping.
- **`agc_eps`** *(float, default=1e-3)*: Epsilon floor for unit‑wise parameter norm in AGC.
- **`centralize_gradients`** *(bool, default=True)*: Whether to subtract mean across axes for each gradient (gradient centralization).
- **`normalize_gradients`** *(bool, default=True)*: Whether to divide gradients by their standard deviation (gradient normalization).
- **`lookahead_merge_time`** *(int, default=5)*: Number of steps between lookahead slow‑weight merges.
- **`lookahead_blending_alpha`** *(float, default=0.5)*: Interpolation factor between fast weights and lookahead slow weights.
- **`weight_decouple`** *(bool, default=True)*: If True, applies decoupled weight decay (AdamW style) before parameter step.
- **`fixed_decay`** *(bool, default=False)*: If True, uses fixed weight‑decay factor rather than scaling by learning rate.
- **`norm_loss_factor`** *(float, default=1e-4)*: Coefficient for additional norm‑based loss regularization on parameter magnitudes.
- **`adam_debias`** *(bool, default=False)*: If True, does not apply bias‑correction to the learning‑rate scaling.
- **`clipnorm`** *(float, optional)*: Global norm threshold to clip gradients before any other processing.
- **`clipvalue`** *(float, optional)*: Value threshold to clip individual gradient elements.
- **`global_clipnorm`** *(float, optional)*: Alias for `clipnorm`, for compatibility.
- **`use_ema`** *(bool, default=False)*: Whether to maintain an exponential moving average of model weights.
- **`ema_momentum`** *(float, default=0.99)*: Decay factor for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Number of steps between EMA‐to‐model weight overwrites.
- **`loss_scale_factor`** *(float, optional)*: Multiplicative factor for dynamic loss‐scaling (mixed precision).
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before applying an update.
- **`name`** *(str, default="ranger21")*: Optional name scope for the optimizer.

**Example Usage**:

```python
import tensorflow as tf
from optimizer.ranger21 import Ranger21

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

# Instantiate Ranger21 optimizer
optimizer = Ranger21(
    num_iterations=10000,
    learning_rate=3e-4,
    weight_decay=1e-2,
    betas=(0.9, 0.999),
    use_softplus=True,
    agc_clipping_value=1e-2,
    norm_loss_factor=1e-4,
    lookahead_merge_time=6,
    lookahead_blending_alpha=0.6
)

# Compile and train
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```

# Tiger

**Overview**:

The `Tiger` optimizer (Tight‑fisted Optimizer) is a budget‑conscious sign‑based optimizer that combines momentum with decoupled weight decay. By tracking an exponential moving average of past gradients (`exp_avg`) and updating parameters by the sign of this average, Tiger minimizes memory overhead—especially under gradient accumulation—while still providing adaptive, per‑parameter step directions. It can emulate both fixed and learning‑rate–scaled weight decay, and supports optional EMA of weights for stabilized evaluation.

**Parameters**:

- **`learning_rate`** *(float, default=1e‑3)*: Base step size for updates.
- **`beta`** *(float, default=0.965)*: Exponential decay rate for the gradient moving average.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay. If `weight_decouple=True`, decay is applied directly to weights; otherwise it is added to the gradient.
- **`weight_decouple`** *(bool, default=True)*: If `True`, applies decoupled weight decay (as in AdamW) before gradient update.
- **`fixed_decay`** *(bool, default=False)*: If `True`, uses a fixed decay factor independent of the learning rate; otherwise scales decay by the current learning rate.
- **`clipnorm`** *(float, optional)*: Maximum norm for gradient clipping.
- **`clipvalue`** *(float, optional)*: Maximum absolute value for gradient clipping.
- **`global_clipnorm`** *(float, optional)*: Maximum global norm for all gradients.
- **`use_ema`** *(bool, default=False)*: Whether to maintain an Exponential Moving Average (EMA) of model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for the EMA of weights.
- **`ema_overwrite_frequency`** *(int, optional)*: How often (in steps) to overwrite model weights with their EMA.
- **`loss_scale_factor`** *(float, optional)*: Static scaling factor applied to the loss (useful for mixed‑precision).
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps over which to accumulate gradients before applying an update.
- **`name`** *(str, default="tiger")*: Optional name prefix for the optimizer.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.tiger import Tiger

# Instantiate Tiger optimizer
optimizer = Tiger(
    learning_rate=5e-4,
    beta=0.98,
    weight_decay=1e-2,
    weight_decouple=True,
    fixed_decay=True,
    clipnorm=1.0,
    use_ema=True,
    ema_momentum=0.995,
    gradient_accumulation_steps=4,
    name="tiger_opt"
)

# Build and compile a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```

# TRAC

**Overview**:

The `TRAC` optimizer wraps any existing Keras optimizer and augments its parameter updates with a time‑reversible adaptive correction mechanism based on a complex error‑function approximation (ERF1994). By tracking both parameter deltas and gradients over multiple decay rates (`betas`), TRAC computes a dynamic scaling factor that modulates each update step—enabling more robust adaptation to non‑stationary or noisy gradient dynamics.

**Parameters**:

- **`optimizer`** *(tf.keras.optimizers.Optimizer)*: The underlying “base” optimizer whose gradients and updates TRAC will correct (e.g. `Adam`, `SGD`).
- **`epsilon`** *(float, default=1e-8)*: Small constant added for numerical stability in divisions and square‑root computations.
- **`betas`** *(tuple of floats, default=(0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999))*: Sequence of decay coefficients controlling how past update–gradient products influence the adaptive scale. Higher values give longer memory.
- **`num_coefs`** *(int, default=128)*: Number of coefficients used in the polynomial approximation of the complex error function (ERF1994). Larger values increase approximation fidelity at the cost of compute.
- **`s_prev`** *(float, default=1e-8)*: Initial value for the cumulative scale term. Ensures nonzero scaling on the first step.
- **`name`** *(str, default="trac”)*: Optional name scope for the optimizer, useful for multi‑optimizer training or logging.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.trac import TRAC

# 1. Instantiate a base optimizer, e.g. Adam
base_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

# 2. Wrap it with TRAC
optimizer = TRAC(
    optimizer=base_opt,
    epsilon=1e-8,
    betas=(0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999),
    num_coefs=128,
    s_prev=1e-8,
    name="trac"
)

# 3. Compile your model with the TRAC optimizer
model = tf.keras.models.Sequential([ ... ])
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 4. Train as usual
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AdaGC

**Overview**:

The `AdaGC` optimizer is an adaptive gradient clipping and rescaling method built on top of standard moment‑based optimizers. During an initial warmup phase, it applies absolute gradient clipping to enforce a maximum update magnitude. After warmup, it dynamically adjusts per‑parameter clipping thresholds based on a running estimate of past clipped gradient norms, blending relative and absolute criteria. This mechanism stabilizes training under highly variable or heavy‑tailed gradient distributions, while optional weight‑decoupled decay and exponential moving averages further improve generalization.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: The base step size for parameter updates.
- **`betas`** *(tuple of two floats, default=(0.9, 0.999))*:
  * First element is the decay rate for the exponential moving average of the (clipped) gradient.
  * Second element is the decay rate for the exponential moving average of the squared (clipped) gradient.
- **`beta`** *(float, default=0.98)*: Momentum coefficient used when updating the running threshold (`gamma`) after warmup.
- **`epsilon`** *(float, default=1e-8)*: Small constant added inside norms and denominators for numerical stability.
- **`weight_decay`** *(float, default=1e-1)*: Coefficient for L2 weight decay. If `weight_decouple=True`, decay is applied directly to parameters before gradient update; otherwise it is added to gradients.
- **`lambda_abs`** *(float, default=1.0)*: Maximum allowed squared‐gradient norm during the absolute clipping (warmup) phase.
- **`lambda_rel`** *(float, default=1.05)*: Multiplicative factor for relative clipping after warmup, scaling the running threshold `gamma`.
- **`warmup_steps`** *(int, default=100)*: Number of initial iterations using absolute clipping (`lambda_abs`) before switching to relative clipping.
- **`weight_decouple`** *(bool, default=True)*: If True, applies weight decay in a decoupled manner (as in AdamW) before gradient update; else adds decay term into the gradient.
- **`fixed_decay`** *(bool, default=False)*: When `weight_decouple=True`, if True uses a fixed decay independent of learning rate; else scales decay by the learning rate.
- **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(float or None)*: Optional built‑in Keras gradient clipping settings (by norm or value).
- **`use_ema`** *(bool, default=False)*: Whether to maintain and apply an exponential moving average of model weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum factor for the weights EMA.
- **`ema_overwrite_frequency`** *(int or None)*: Frequency (in steps) to overwrite model weights with EMA weights.
- **`loss_scale_factor`** *(float or None)*: Factor by which to scale the loss for mixed‑precision training.
- **`gradient_accumulation_steps`** *(int or None)*: Number of steps over which to accumulate gradients before applying an update.
- **`name`** *(str, default="adagc")*: Name identifier for the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.adagc import AdaGC

# 1. Instantiate the AdaGC optimizer
optimizer = AdaGC(
    learning_rate=5e-4,
    betas=(0.9, 0.999),
    beta=0.98,
    epsilon=1e-8,
    weight_decay=1e-2,
    lambda_abs=0.5,
    lambda_rel=1.1,
    warmup_steps=50,
    weight_decouple=True,
    fixed_decay=False
)

# 2. Compile a model with AdaGC
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 3. Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```

# Ranger25

**Overview**:

`Ranger25` is an experimental composite optimizer that blends together seven advanced optimization techniques—ADOPT, AdEMAMix, Cautious updates, StableAdamW/Adam‑atan2, OrthoGrad, adaptive gradient clipping (AGC), and Lookahead—to achieve more reliable convergence, improved stability, and faster training across a wide range of deep‑learning tasks. By combining theoretical convergence fixes (ADOPT) with enhanced utilization of past gradients (AdEMAMix), directional masking (Cautious), numerical stability (Adam‑atan2), gradient decorrelation (OrthoGrad), unit‑wise clipping (AGC), and periodic weight averaging (Lookahead), Ranger25 aims to deliver the best of each world in a single optimizer.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates.
- **`betas`** *(tuple of three floats, default=(0.9, 0.98, 0.9999))*:
  * `beta1` for first‑moment EMA (momentum)
  * `beta2` for second‑moment EMA (RMS scaling)
  * `beta3` for slow EMA used in the “mix” component (AdEMAMix).
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in denominator (and in StableAdamW/Adam‑atan2 branch).
- **`weight_decay`** *(float, default=1e-3)*: Coefficient for decoupled weight‑decay regularization (AdamW style).
- **`alpha`** *(float, default=5.0)*: Mixing coefficient magnitude for the slow EMA in AdEMAMix.
- **`t_alpha_beta3`** *(int or None, default=None)*: Number of steps over which to warm up `alpha` and `beta3`; if `None`, no warmup.
- **`lookahead_merge_time`** *(int, default=5)*: Number of steps between Lookahead slow‑weight merges.
- **`lookahead_blending_alpha`** *(float, default=0.5)*: Interpolation factor between fast and slow weights at each Lookahead merge.
- **`cautious`** *(bool, default=True)*: Enable Cautious updates—masking out parameter updates whose sign conflicts with the raw gradient.
- **`stable_adamw`** *(bool, default=True)*: Use StableAdamW variant, which rescales step size by measured gradient variance for numerical stability.
- **`orthograd`** *(bool, default=True)*: Enable OrthoGrad, projecting each gradient to be orthogonal to its parameter vector before update.
- **`weight_decouple`** *(bool, default=True)*: Apply weight decay in a decoupled fashion (AdamW) rather than via loss augmentation.
- **`fixed_decay`** *(bool, default=False)*: Use fixed weight‑decay (not scaled by learning rate) when `weight_decouple` is True.
- **`clipnorm`** *(float or None)*: Clip gradients by global L‑2 norm.
- **`clipvalue`** *(float or None)*: Clip gradients by value.
- **`global_clipnorm`** *(float or None)*: Alias for clipping by global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Maintain an exponential moving average of model weights.
- **`ema_momentum`** *(float, default=0.99)*: Decay rate for weight EMA.
- **`ema_overwrite_frequency`** *(int or None)*: How often to overwrite model weights with EMA weights.
- **`loss_scale_factor`** *(float or None)*: Static loss‑scaling factor for mixed‑precision training.
- **`gradient_accumulation_steps`** *(int or None)*: Number of steps to accumulate gradients before applying an update.
- **`name`** *(str, default="ranger25")*: Name identifier for the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.ranger25 import Ranger25

# Instantiate the Ranger25 optimizer with custom settings
optimizer = Ranger25(
    learning_rate=3e-4,
    betas=(0.9, 0.98, 0.9999),
    epsilon=1e-8,
    weight_decay=1e-4,
    alpha=4.0,
    t_alpha_beta3=10000,
    lookahead_merge_time=6,
    lookahead_blending_alpha=0.6,
    cautious=True,
    stable_adamw=True,
    orthograd=True,
    fixed_decay=False,
    clipnorm=1.0,
    use_ema=True,
    ema_momentum=0.995,
    gradient_accumulation_steps=2,
    name="ranger25_custom"
)

# Compile a Keras model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```

# LOMO

**Overview**:

The `LOMO` (LOw-Memory Optimization) optimizer is designed to reduce the memory footprint during training, particularly in large-scale distributed settings like ZeRO stage 3. It achieves this by fusing the gradient computation and parameter update steps for each parameter individually, avoiding the need to store all gradients simultaneously. It supports gradient clipping by norm or value and incorporates optional dynamic loss scaling for mixed-precision training (`tf.float16`).

**Parameters**:

-  **`model`** *(tf.keras.Model)*: The Keras model whose parameters will be optimized.
-  **`lr`** *(float, default=1e-3)*: The learning rate or step size for parameter updates.
-  **`clip_grad_norm`** *(float, optional, default=None)*: If set to a positive value, gradients will be clipped globally based on the total norm of all gradients. Requires calling `optimizer.grad_norm()` before `optimizer.fused_backward()`.
-  **`clip_grad_value`** *(float, optional, default=None)*: If set to a positive value, gradients will be clipped element-wise to stay within the range `[-clip_grad_value, +clip_grad_value]`.
-  **`zero3_enabled`** *(bool, default=True)*: If `True`, enables ZeRO stage 3 style optimization where gradients are reduced across replicas and only the relevant partition of the parameter is updated locally. If `False`, performs standard updates on the full parameters.
-  **`name`** *(str, default="lomo")*: The name for the optimizer instance.

*(Note: For `tf.float16` parameters, dynamic loss scaling is automatically enabled to prevent underflow.)*

**Example Usage**:

```python
import tensorflow as tf
from optimizers.lomo import LOMO # Assuming LOMO is in your_module

# --- Model Definition ---
# inputs = tf.keras.Input(shape=(...))
# outputs = YourModelLayers(inputs)
# model = tf.keras.Model(inputs, outputs)
# ------------------------

# Instantiate optimizer
optimizer = LOMO(model, lr=1e-3, clip_grad_norm=1.0)

# --- Training Step ---
# @tf.function # Decorate for performance
def train_step(inputs, labels):
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(inputs, training=True)
        # Ensure loss computation uses float32 for stability
        loss = tf.keras.losses.your_loss_function(labels, tf.cast(predictions, tf.float32))

    # If clip_grad_norm is used, calculate norm first
    if optimizer.clip_grad_norm is not None and optimizer.clip_grad_norm > 0.0:
         # Pass loss before potential scaling if LOMO handles it internally
         # Note: The provided LOMO code seems to scale loss *inside* grad_norm/fused_backward
         optimizer.grad_norm(tape, loss, model.trainable_variables)
         # fused_backward will use the calculated clip_coef

    # Perform fused backward pass and update
    # Pass the original, potentially unscaled loss if LOMO handles scaling
    optimizer.fused_backward(tape, loss, model.trainable_variables, lr=optimizer.lr) # Pass current lr

    return loss
# ---------------------

# --- Training Loop ---
# for epoch in range(num_epochs):
#     for step, (x_batch, y_batch) in enumerate(train_dataset):
#         loss_value = train_step(x_batch, y_batch)
#         if step % log_interval == 0:
#             print(f"Epoch {epoch}, Step {step}, Loss: {loss_value.numpy()}")
# ---------------------
```

*(Note: LOMO requires a custom training loop because it uses `fused_backward` and potentially `grad_norm` instead of the standard Keras `optimizer.apply_gradients` used within `model.compile`/`model.fit`.)*

# AdaLOMO

**Overview**:

The `AdaLOMO` optimizer combines the low-memory optimization strategy of LOMO with adaptive learning rate methods. It approximates the second moment of gradients using row and column averages (similar to Adafactor) to adapt the learning rate for each parameter, aiming for improved convergence and stability while maintaining low memory usage. It includes features like weight decay, adaptive gradient clipping based on update norms, and learning rate scaling based on parameter norms (similar to LAMB).

**Parameters**:

-  **`model`** *(tf.keras.Model)*: The Keras model whose parameters will be optimized.
-  **`lr`** *(float, default=1e-3)*: The base learning rate. The actual step size is adapted based on parameter norms and second moment estimates.
-  **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 weight decay (applied additively to the gradient before the update step).
-  **`loss_scale`** *(float, default=1024.0)*: Static loss scaling factor used to prevent gradient underflow, especially during mixed-precision training. Gradients are unscaled internally before updates.
-  **`clip_threshold`** *(float, default=1.0)*: Threshold for adaptive gradient clipping. The normalized update is clipped based on this value.
-  **`decay_rate`** *(float, default=-0.8)*: Exponent used to compute the decay factor (`beta2_t`) for the running averages of squared gradients. `beta2_t = 1.0 - steps ** decay_rate`.
-  **`clip_grad_norm`** *(float, optional, default=None)*: If set to a positive value, gradients will be clipped globally based on the total norm of all gradients *before* adaptive calculations. Requires calling `optimizer.grad_norm()` before `optimizer.fused_backward()`.
-  **`clip_grad_value`** *(float, optional, default=None)*: If set to a positive value, gradients will be clipped element-wise *before* adaptive calculations.
-  **`eps1`** *(float, default=1e-30)*: A small epsilon added to the squared gradients before computing row/column means, ensuring numerical stability.
-  **`eps2`** *(float, default=1e-3)*: A small epsilon used when scaling the learning rate by the parameter norm (`lr_scaled = lr * max(eps2, p_rms)`), preventing division by zero or overly large learning rates for small parameters.
-  **`zero3_enabled`** *(bool, default=True)*: If `True`, enables ZeRO stage 3 style optimization where gradients are reduced, second moments are potentially calculated on full gradients, and only the relevant partition of the parameter is updated locally using partitioned updates. If `False`, performs standard updates on the full parameters.
-  **`name`** *(str, default="adalomo")*: The name for the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.lomo import AdaLOMO # Assuming AdaLOMO is in your_module

# --- Model Definition ---
# inputs = tf.keras.Input(shape=(...))
# outputs = YourModelLayers(inputs)
# model = tf.keras.Model(inputs, outputs)
# ------------------------

# Instantiate optimizer
optimizer = AdaLOMO(
    model,
    lr=1e-3,
    weight_decay=0.01,
    clip_threshold=1.0,
    clip_grad_norm=1.0 # Example: enabling global grad norm clipping
)

# --- Training Step ---
# @tf.function # Decorate for performance
def train_step(inputs, labels):
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(inputs, training=True)
        # Ensure loss computation uses float32 for stability
        loss = tf.keras.losses.your_loss_function(labels, tf.cast(predictions, tf.float32))

    # If clip_grad_norm is used, calculate norm first
    if optimizer.clip_grad_norm is not None and optimizer.clip_grad_norm > 0.0:
        # Pass loss before potential scaling if AdaLOMO handles it internally
        # Note: The provided AdaLOMO code scales loss *inside* grad_norm/fused_backward
        optimizer.grad_norm(tape, loss, model.trainable_variables)
        # fused_backward will use the calculated clip_coef

    # Perform fused backward pass and update
    # Pass the original, potentially unscaled loss if AdaLOMO handles scaling
    optimizer.fused_backward(tape, loss, model.trainable_variables, lr=optimizer.lr) # Pass base lr

    return loss
# ---------------------

# --- Training Loop ---
# optimizer.num_steps = 0 # Initialize step counter
# for epoch in range(num_epochs):
#     for step, (x_batch, y_batch) in enumerate(train_dataset):
#         loss_value = train_step(x_batch, y_batch)
#         if step % log_interval == 0:
#             print(f"Epoch {epoch}, Step {step}, Loss: {loss_value.numpy()}")
# ---------------------
```

# A2Grad

**Overview**:

The A2Grad optimizer implements the “Adaptive Accelerated Stochastic Gradient” algorithm. It combines Nesterov‑style acceleration (to achieve optimal deterministic convergence) with per‑coordinate adaptive scaling of gradient innovations (to match the optimal stochastic convergence rate). Three variants of moving‑average (“uni”, “inc”, “exp”) let you trade off bias versus variance in the adaptive term. A2Grad fixes its base learning rate internally and automatically adapts to both smoothness and noise, often yielding faster, more stable training than standard Adam or AMSGrad.

**Parameters**:

-  **`beta`** *(float, default=10.0)*: Weight on the adaptive (diagonal) proximal term. Larger values increase the influence of past gradient variability.
-  **`lips`** *(float, default=10.0)*: Estimate of the Lipschitz constant L of the gradient. Used to set the acceleration step size γₖ = 2·L/(k+1).
-  **`rho`** *(float, default=0.5)*: Smoothing factor for the exponential‑moving‑average variant. Only used when `variant='exp'`.
-  **`variant`** *(str, default='uni')*: Choice of moving‑average scheme for the squared “innovation” term:
-  **`clipnorm`** *(float or None)*: If set, each tensor’s gradient is clipped by its own L₂ norm before scaling.
-  **`clipvalue`** *(float or None)*: If set, each tensor’s gradient values are individually clipped to \[–clipvalue, clipvalue].
-  **`global_clipnorm`** *(float or None)*: If set, all gradients are jointly clipped by their global norm.
-  **`use_ema`** *(bool, default=False)*: Maintain an exponential moving average of model weights for evaluation or stability.
-  **`ema_momentum`** *(float, default=0.99)*: Decay rate for the weight EMA when `use_ema=True`.
-  **`ema_overwrite_frequency`** *(int or None)*: How many steps between replacing model weights with their EMA values.
-  **`loss_scale_factor`** *(float or None)*: Static multiplier applied to the loss for mixed‑precision training.
-  **`gradient_accumulation_steps`** *(int or None)*: Number of steps to accumulate gradients before performing an update.
-  **`name`** *(str, default="a2grad")*: Identifier name for this optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.a2grad import A2Grad

# Create an A2Grad optimizer using the 'exp' variant
optimizer = A2Grad(
    beta=8.0,
    lips=5.0,
    rho=0.9,
    variant='exp',
    clipnorm=1.0,
    clipvalue=0.5,
    use_ema=True,
    ema_momentum=0.995,
    name="a2grad_exp"
)

# Build a simple Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(784,)),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile with A2Grad
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(train_dataset, validation_data=val_dataset, epochs=15)
```

# AdamG

**Overview**:

The `AdamG` optimizer is a generalized extension of the Adam family that introduces an additional third moment accumulator and exponentiated numerator scaling. By incorporating parameters $p$ and $q$ into the numerator function and a separate decay control for weights, `AdamG` adapts more flexibly to gradient distributions, potentially improving convergence on both smooth and noisy objectives.

**Parameters**:

-  **`learning_rate`** *(float, default=1.0)*: Base step size for parameter updates.
-  **`betas`** *(tuple of three floats, default=(0.95, 0.999, 0.95))*: Exponential decay rates for the first, second, and third moment estimates, respectively.
-  **`epsilon`** *(float, default=1e-8)*: Small constant added for numerical stability in denominator.
-  **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 weight decay. Applied either in a decoupled fashion or directly to the gradient.
-  **`p`** *(float, default=0.2)*: Scaling factor in the numerator function $f(x) = p \, x^q$.
-  **`q`** *(float, default=0.24)*: Exponent in the numerator function $f(x) = p \, x^q$.
-  **`weight_decouple`** *(bool, default=False)*: If `True`, applies weight decay by directly scaling the variable (decoupled decay).
-  **`fixed_decay`** *(bool, default=False)*: If `True` with decoupled decay, uses a fixed decay rate rather than scaling by learning rate.
-  **`clipnorm`** *(float, optional)*: Maximum norm for gradient clipping per variable.
-  **`clipvalue`** *(float, optional)*: Maximum absolute value for gradient clipping per variable.
-  **`global_clipnorm`** *(float, optional)*: Maximum global norm for all gradients before update.
-  **`use_ema`** *(bool, default=False)*: Whether to maintain an exponential moving average of model weights.
-  **`ema_momentum`** *(float, default=0.99)*: Momentum factor for exponential moving average.
-  **`ema_overwrite_frequency`** *(int, optional)*: Number of steps between overwriting model weights with their EMA.
-  **`loss_scale_factor`** *(float, optional)*: Factor by which to scale the loss for mixed‑precision or dynamic scaling.
-  **`gradient_accumulation_steps`** *(int, optional)*: Number of steps over which to accumulate gradients before applying an update.
-  **`name`** *(str, default="adamg")*: Identifier for the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from keras.src.optimizers import optimizer
from optimizers.adamg import AdamG  # assume AdamG is defined in this module

# Instantiate the AdamG optimizer
optimizer = AdamG(
    learning_rate=0.001,
    betas=(0.9, 0.999, 0.9),
    epsilon=1e-7,
    weight_decay=1e-4,
    p=0.3,
    q=0.25,
    weight_decouple=True,
    fixed_decay=False,
    clipnorm=1.0,
    use_ema=True,
    ema_momentum=0.98
)

# Compile a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```

# AdaMax

**Overview**:

The `AdaMax` optimizer is a variant of Adam based on the infinity norm (max‐norm) of past gradients. By replacing the second‐moment estimate with an exponentially weighted infinity norm, AdaMax obtains more stable parameter updates when gradients have heavy tails or outliers. This often yields improved convergence in practice, especially on models with sparse or highly variable gradients. ([arXiv][1], [PyTorch][2])

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates. ([PyTorch][2])
* **`betas`** *(tuple of two floats, default=(0.9, 0.999))*: Exponential decay rates for the first‐moment (mean) and the infinity‐norm estimates, respectively. ([arXiv][1])
* **`epsilon`** *(float, default=1e-8)*: Small constant added to the denominator for numerical stability. ([PyTorch][2])
* **`weight_decay`** *(float, default=0.0)*: L2 penalty coefficient; applied either by modifying gradients or via decoupled weight‐scale.
* **`r`** *(float, default=0.95)*: Smoothing constant for the optional adaptive gradient‐norm scaling (`adanorm`).
* **`adanorm`** *(bool, default=False)*: If `True`, rescales each gradient by the ratio of its running norm to its instantaneous norm, mitigating large gradient spikes.
* **`adam_debias`** *(bool, default=False)*: If `True`, omits the usual bias‐correction on the first‐moment estimate, using raw learning‐rate scaling instead.
* **`weight_decouple`** *(bool, default=False)*: If `True`, applies decoupled weight decay (as in AdamW) by directly scaling parameters before the update.
* **`fixed_decay`** *(bool, default=False)*: When using decoupled decay, applies a fixed decay factor rather than scaling by the learning rate.
* **`clipnorm`** *(float, optional)*: Maximum norm for per‐variable gradient clipping.
* **`clipvalue`** *(float, optional)*: Maximum absolute value for per‐variable gradient clipping.
* **`global_clipnorm`** *(float, optional)*: Maximum global norm for all gradients combined.
* **`use_ema`** *(bool, default=False)*: Maintain an exponential moving average of model weights for evaluation or ensembling.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for the weight EMA.
* **`ema_overwrite_frequency`** *(int, optional)*: Number of steps between copying EMA weights back to the model.
* **`loss_scale_factor`** *(float, optional)*: Multiply the loss by this factor before computing gradients (useful for mixed precision).
* **`gradient_accumulation_steps`** *(int, optional)*: Number of forward/backward passes to accumulate gradients before applying an update.
* **`name`** *(str, default="adamax")*: Identifier name for the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.adamax import AdaMax  # assuming this module

# Instantiate AdaMax with decoupled weight decay and adanorm enabled
optimizer = AdaMax(
    learning_rate=2e-4,
    betas=(0.9, 0.999),
    epsilon=1e-8,
    weight_decay=1e-3,
    r=0.9,
    adanorm=True,
    adam_debias=False,
    weight_decouple=True,
    fixed_decay=False,
    clipnorm=1.0,
    use_ema=True,
    ema_momentum=0.97
)

# Build and compile a model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=15)
```

# AdamMini

**Overview**:

The `AdamMini` optimizer is a variant of the Adam family tailored for large-scale transformer models with sharded parameters and specialized update rules for embedding and attention layers. It extends the standard Adam algorithm by decoupling weight decay, supporting tensor model parallelism, applying grouped updates for query/key projections, and maintaining separate moment statistics per attention head. This design improves convergence behavior in massive transformer architectures while enabling efficient distributed training and optional exponential moving average (EMA) of model weights.

**Parameters**:

* **`learning_rate`** *(float, default=1.0)*: Base step size for parameter updates.
* **`betas`** *(tuple of two floats, default=(0.9, 0.999))*: Exponential decay rates for the first and second moment estimates.
* **`epsilon`** *(float, default=1e-8)*: Small constant added to the denominator for numerical stability.
* **`weight_decay`** *(float, default=0.1)*: Coefficient for L2 regularization; can be decoupled or applied directly to gradients.
* **`model_sharding`** *(bool, default=False)*: Whether to gather and reduce gradient norms across replicas for sharded model parallelism.
* **`num_embeds`** *(int, default=2048)*: Hidden dimension size, used to reshape parameters for per-head statistics.
* **`num_heads`** *(int, default=32)*: Number of attention heads, determining the grouping of query/key projections.
* **`num_query_groups`** *(int or None, default=None)*: Number of groups for query/key projections; defaults to `num_embeds` if not specified.
* **`clipnorm`** *(float or None, default=None)*: Clip gradients by norm threshold.
* **`clipvalue`** *(float or None, default=None)*: Clip gradients by absolute value.
* **`global_clipnorm`** *(float or None, default=None)*: Clip gradients by global norm across all parameters.
* **`use_ema`** *(bool, default=False)*: Whether to maintain an exponential moving average of model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for EMA updates.
* **`ema_overwrite_frequency`** *(int or None, default=None)*: How often (in steps) to overwrite model weights with EMA weights.
* **`loss_scale_factor`** *(float or None, default=None)*: Scaling factor for loss when using mixed precision.
* **`gradient_accumulation_steps`** *(int or None, default=None)*: Number of micro-steps to accumulate gradients before applying an update.
* **`name`** *(str, default="adammini")*: Optional name prefix for optimizer variables.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.adam_mini import AdamMini

# Instantiate the AdamMini optimizer
optimizer = AdamMini(
    learning_rate=0.5,
    betas=(0.9, 0.98),
    epsilon=1e-6,
    weight_decay=0.05,
    model_sharding=True,
    num_embeds=4096,
    num_heads=16,
    use_ema=True,
    ema_momentum=0.995
)

# Compile a transformer-style Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=4096),
    tf.keras.layers.MultiHeadAttention(num_heads=16, key_dim=256),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model on a distributed strategy
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model.fit(train_dataset, validation_data=val_dataset, epochs=5)
```
