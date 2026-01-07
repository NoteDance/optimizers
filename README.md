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
- **`cautious`** *(bool, default=True)*: Use cautious masking to reduce updates that conflict in sign with the gradient (stabilizes updates).
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

The `GaLore` optimizer is a memory-efficient variant of the Adam optimizer that projects gradients into a low-rank subspace using orthogonal matrices derived from SVD. This reduces the memory footprint for storing momentum states, making it suitable for training large-scale models while maintaining performance.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
* **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
* **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
* **`epsilon`** *(float, default=1e-6)*: Small constant for numerical stability.
* **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay.
* **`rank`** *(int, optional, default=None)*: The rank of the low-rank projection; if None, no projection is applied.
* **`update_proj_gap`** *(int, optional, default=None)*: The frequency (in steps) for updating the projection matrix; if None, no projection is applied.
* **`scale`** *(float, optional, default=None)*: Scaling factor applied to the projected-back gradient.
* **`projection_type`** *(str, optional, default=None)*: Type of projection scheme ('std', 'reverse_std', 'right', 'left', 'full', 'random'); if None, no projection is applied.
* **`clipnorm`** *(float, optional)*: Clips gradients by norm.
* **`clipvalue`** *(float, optional)*: Clips gradients by value.
* **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
* **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
* **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients.
* **`name`** *(str, default="galore")*: Name of the optimizer.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.galore import GaLore

# Instantiate optimizer
optimizer = GaLore(
    learning_rate=1e-3,
    rank=128,
    update_proj_gap=50,
    scale=1.0,
    projection_type='std'
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

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

The `Muon` optimizer applies a hybrid update strategy:

1. **Muon Updates** for matrix‐shaped parameters (rank ≥ 2): uses Nesterov momentum combined with Newton–Schulz “zero‐power” normalization over micro‑batches distributed across devices, optionally scaling per‑tensor learning rates by shape.
2. **AdamW‑style Updates** for the remaining parameters: standard Adam with decoupled weight decay and bias‐corrected moments plus an epsilon floor.

This design accelerates large weight matrices with normalized momentum while retaining AdamW’s robustness on biases and vectors.

**Parameters**:

* **`params`** *(list of `tf.Variable`)*: Primary parameters for Muon updates; any `adamw_params` are appended for AdamW updates.
* **`learning_rate`** *(float, default=2e-2)*: Base learning rate for Muon group.
* **`beta1`** *(float, default=0.9)*: Decay rate for momentum in both Muon and AdamW groups.
* **`beta2`** *(float, default=0.95)*: Decay rate for second‐moment in Muon normalization and AdamW.
* **`weight_decay`** *(float, default=1e-2)*: L2 regularization coefficient for Muon updates (decoupled if `weight_decouple=True`).
* **`momentum`** *(float, default=0.95)*: Nesterov momentum factor for Muon updates.
* **`weight_decouple`** *(bool, default=True)*: If `True`, apply decoupled weight decay to Muon parameters; otherwise include in gradient.
* **`nesterov`** *(bool, default=True)*: Applies Nesterov lookahead in momentum.
* **`ns_steps`** *(int, default=5)*: Number of Newton–Schulz iterations for zero‐power gradient normalization.
* **`use_adjusted_lr`** *(bool, default=False)*: If `True`, scales Muon learning rates by 0.2 × √(max(dim0,dim1)); otherwise uses uniform `learning_rate`.
* **`adamw_params`** *(list of `tf.Variable` or `None`, default=None)*: Secondary parameters for AdamW updates.
* **`adamw_lr`** *(float, default=3e-4)*: Learning rate for AdamW updates on `adamw_params`.
* **`adamw_wd`** *(float, default=0.0)*: Weight‐decay coefficient for AdamW.
* **`adamw_eps`** *(float, default=1e-8)*: Epsilon for AdamW denominator stability.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional floats)*: Gradient clipping thresholds.
* **`use_ema`**, **`ema_momentum`**, **`ema_overwrite_frequency`** *(optional)*: EMA of weights settings.
* **`loss_scale_factor`**, **`gradient_accumulation_steps`** *(optional)*: Mixed‑precision loss scaling and accumulation.
* **`name`** *(str, default="muon")*: Name prefix for optimizer variables.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.muon import Muon

# 1. Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10)
])
params = model.trainable_variables

# 2. Instantiate Muon optimizer
optimizer = Muon(
    params=params,
    learning_rate=2e-2,
    beta1=0.9,
    beta2=0.95,
    weight_decay=1e-2,
    momentum=0.95,
    weight_decouple=True,
    nesterov=True,
    ns_steps=5,
    use_adjusted_lr=True,
    adamw_params=None,
    adamw_lr=3e-4,
    adamw_wd=1e-3,
    adamw_eps=1e-8,
    clipnorm=1.0,
    use_ema=True,
    ema_momentum=0.99,
    gradient_accumulation_steps=4
)

# 3. Compile model
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 4. Train
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
-  **`clipvalue`** *(float or None)*: If set, each tensor’s gradient values are individually clipped to \[–clipvalue, clipvalue].
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
-  **`betas`** *(tuple of three floats, default=(0.95, 0.999, 0.95))*: Exponential decay rates for the first, second, and third moment estimates, respectively.
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
* **`betas`** *(tuple of two floats, default=(0.9, 0.999))*: Exponential decay rates for the first‐moment (mean) and the infinity‐norm estimates, respectively. ([arXiv][1])
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

# Fira

**Overview**:

The `Fira` optimizer extends the Adam family by integrating low‐rank gradient projections via the GaLore projector. It adaptively scales gradients with moment estimates and selectively projects and reconstructs updates for matrix parameters, which can improve training efficiency and generalization in large models. Optional maximization, mixed‐precision loss scaling, gradient accumulation, and Exponential Moving Average (EMA) of weights are also supported.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates.
* **`betas`** *(tuple of two floats, default=(0.9, 0.999))*: Exponential decay rates for the first and second moment estimates.
* **`epsilon`** *(float, default=1e-6)*: Small constant for numerical stability in denominator.
* **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 regularization applied to variables after update.
* **`maximize`** *(bool, default=False)*: If `True`, performs gradient ascent instead of descent.
* **`rank`** *(int or None, default=None)*: Rank of the low‐rank approximation for 2D parameters; if `None`, no projection is applied.
* **`update_proj_gap`** *(int or None, default=None)*: Number of steps between projector refreshes; only used when `rank` is set.
* **`scale`** *(float or None, default=None)*: Scaling factor applied within the GaLore projection; only used when `rank` is set.
* **`projection_type`** *(str or None, default=None)*: Type of projection strategy passed to the GaLore projector.
* **`clipnorm`** *(float, optional)*: Clip gradients by their norm before any other update.
* **`clipvalue`** *(float, optional)*: Clip gradients by absolute value before any other update.
* **`global_clipnorm`** *(float, optional)*: Clip gradients by global norm across all parameters.
* **`use_ema`** *(bool, default=False)*: Whether to maintain an Exponential Moving Average of model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for EMA updates when `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) at which the model weights are overwritten with EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss in mixed‐precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of micro‐batches to accumulate gradients over before applying an update.
* **`name`** *(str, default="fira")*: Optional name prefix for all optimizer variables.

**Example Usage**:

```python
import tensorflow as tf
from keras.src.optimizers import optimizer
from optimizers.fira import Fira

# Instantiate the Fira optimizer with low-rank projection
optimizer = Fira(
    learning_rate=5e-4,
    betas=(0.9, 0.999),
    epsilon=1e-6,
    weight_decay=1e-2,
    maximize=False,
    rank=64,
    update_proj_gap=100,
    scale=0.1,
    projection_type="svd",
    use_ema=True,
    ema_momentum=0.995,
    gradient_accumulation_steps=4
)

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile with Fira
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```

# RACS

**Overview**:

The `RACS` optimizer is a coordinate‐separable adaptive method that normalizes gradients by factorizing their squared values along rows and columns. It maintains two sets of exponential moving averages—one per row and one per column—and uses them to rescale the gradient before applying updates. Optional features include weight decoupling, fixed decay, maximization (gradient ascent), clipping, EMA of weights, mixed‐precision loss scaling, and gradient accumulation.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates.
* **`beta`** *(float, default=0.9)*: Exponential decay rate for both row‐ and column‐wise second moment estimates.
* **`epsilon`** *(float, default=1e-8)*: Small constant added to denominators for numerical stability.
* **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 regularization.
* **`alpha`** *(float, default=0.05)*: Scaling factor for the normalized gradient during the update.
* **`gamma`** *(float, default=1.01)*: Threshold multiplier to limit drastic step‐size increases.
* **`maximize`** *(bool, default=False)*: If `True`, the optimizer performs gradient ascent rather than descent.
* **`weight_decouple`** *(bool, default=True)*: If `True`, applies decoupled weight decay (as in AdamW); otherwise standard decay adds to the gradient.
* **`fixed_decay`** *(bool, default=False)*: When using decoupled decay, fixes the weight decay coefficient independent of learning rate.
* **`clipnorm`** *(float, optional)*: Clip gradients by their norm before any other processing.
* **`clipvalue`** *(float, optional)*: Clip gradients by absolute value before any other processing.
* **`global_clipnorm`** *(float, optional)*: Clip gradients by global norm across all parameters.
* **`use_ema`** *(bool, default=False)*: Maintain an Exponential Moving Average of model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for EMA updates.
* **`ema_overwrite_frequency`** *(int, optional)*: How often (in steps) to overwrite model weights with EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss in mixed‐precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of micro‐batches to accumulate before applying an update.
* **`name`** *(str, default="racs")*: Optional name prefix for optimizer variables.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.racs import RACS

optimizer = RACS(
    learning_rate=2e-3,
    beta=0.95,
    epsilon=1e-7,
    weight_decay=1e-2,
    alpha=0.1,
    gamma=1.02,
    weight_decouple=True,
    fixed_decay=True,
    use_ema=True,
    ema_momentum=0.99,
    gradient_accumulation_steps=2
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_dataset, validation_data=val_dataset, epochs=15)
```

# Alice

**Overview**:

The `Alice` optimizer combines three mechanisms: low‐rank subspace updates for large matrices, adaptive moment estimation in the projected subspace, and a compensation step for the orthogonal residual. It maintains separate low‐rank bases (`U`, `Q`), moment estimates (`m`, `v`), and a residual scaling factor (`phi`) for each 2D parameter. This enables efficient large‐matrix optimization with controlled update directions. Additional features include weight decoupling, fixed decay, maximization, clipping, EMA, mixed‐precision loss scaling, and gradient accumulation.

**Parameters**:

* **`learning_rate`** *(float, default=0.02)*: Base learning rate for parameter updates.
* **`betas`** *(tuple of three floats, default=(0.9, 0.9, 0.999))*: Exponential decay rates for subspace residual update (`beta1`), subspace moment estimate (`beta2`), and subspace correlation matrix (`beta3`).
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
* **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 regularization.
* **`alpha`** *(float, default=0.3)*: Scaling factor for the subspace update component.
* **`alpha_c`** *(float, default=0.4)*: Scaling factor for the compensation (residual) update.
* **`update_interval`** *(int, default=200)*: Steps between full subspace basis refreshes.
* **`rank`** *(int, default=256)*: Dimension of the low‐rank subspace basis used for projection.
* **`gamma`** *(float, default=1.01)*: Threshold multiplier to regulate compensation magnitude.
* **`leading_basis`** *(int, default=40)*: Number of top eigenvectors to retain in the basis during refresh.
* **`maximize`** *(bool, default=False)*: If `True`, performs gradient ascent.
* **`weight_decouple`** *(bool, default=True)*: Use decoupled weight decay when `True`.
* **`fixed_decay`** *(bool, default=False)*: When decoupled, apply fixed decay independent of learning rate.
* **`clipnorm`** *(float, optional)*: Clip gradients by norm.
* **`clipvalue`** *(float, optional)*: Clip gradients by value.
* **`global_clipnorm`** *(float, optional)*: Clip gradients by global norm.
* **`use_ema`** *(bool, default=False)*: Maintain EMA of weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency to overwrite weights with EMA.
* **`loss_scale_factor`** *(float, optional)*: For mixed‐precision loss scaling.
* **`gradient_accumulation_steps`** *(int, optional)*: Steps to accumulate gradients before update.
* **`name`** *(str, default="alice")*: Optional name for optimizer variables.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.racs import Alice

optimizer = Alice(
    learning_rate=1e-2,
    betas=(0.9, 0.9, 0.999),
    epsilon=1e-8,
    alpha=0.5,
    alpha_c=0.3,
    update_interval=100,
    rank=128,
    gamma=1.05,
    leading_basis=32,
    weight_decouple=True,
    fixed_decay=False,
    use_ema=True,
    ema_momentum=0.995,
    gradient_accumulation_steps=4
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_dataset, validation_data=val_dataset, epochs=25)
```

# SPAM

**Overview**:

The `SPAM` optimizer integrates sparse parameter updates with adaptive moment estimation and cosine‐decayed “death” of update masks. At each update, a fixed fraction (`density`) of a weight matrix is selected via a random mask; only those entries are updated using Adam‐style moment estimates, while the rest remain frozen. Masks are periodically refreshed, and a cosine decay warms up the mask refresh rate. Optional features include gradient clipping, decoupled weight decay, maximization (ascent), EMA of weights, mixed‐precision loss scaling, and gradient accumulation.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates.
* **`betas`** *(tuple of two floats, default=(0.9, 0.999))*: Exponential decay rates for the first and second moment estimates.
* **`epsilon`** *(float, default=1e-6)*: Small constant for numerical stability in denominator.
* **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 regularization applied decoupled after the update.
* **`density`** *(float, default=1.0)*: Fraction of weights in each 2D tensor to update (mask density).
* **`warmup_epoch`** *(int, default=50)*: Number of epochs over which the mask “death rate” warms up via cosine decay.
* **`threshold`** *(float, default=5000)*: Clipping threshold multiplier for large gradient squares; gradients above `threshold × exp_avg_sq` are clipped in magnitude.
* **`grad_accu_steps`** *(int, default=20)*: Minimum step before threshold clipping is applied.
* **`update_proj_gap`** *(int, default=500)*: Number of optimization steps between mask refreshes.
* **`maximize`** *(bool, default=False)*: If `True`, performs gradient ascent instead of descent.
* **`clipnorm`** *(float, optional)*: Clip gradients by global norm before any update.
* **`clipvalue`** *(float, optional)*: Clip gradients by absolute value before any update.
* **`global_clipnorm`** *(float, optional)*: Clip gradients by global norm across all parameters.
* **`use_ema`** *(bool, default=False)*: Maintain an Exponential Moving Average of model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for EMA updates when `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) at which model weights are overwritten with EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss in mixed‐precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of micro‐batches to accumulate before applying an update.
* **`name`** *(str, default="spam")*: Optional name prefix for optimizer variables.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.spam import SPAM

# Instantiate the SPAM optimizer with 50% sparsity and periodic mask updates
optimizer = SPAM(
    learning_rate=1e-3,
    betas=(0.9, 0.999),
    epsilon=1e-6,
    weight_decay=1e-2,
    density=0.5,
    warmup_epoch=20,
    threshold=1000,
    grad_accu_steps=10,
    update_proj_gap=200,
    maximize=False,
    use_ema=True,
    ema_momentum=0.995,
    gradient_accumulation_steps=4
)

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile with SPAM
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=30)
```

# StableSPAM

**Overview**:

The `StableSPAM` optimizer builds on sparse adaptive moment estimation by adding dynamic clipping based on historical gradient norms and magnitudes. It tracks maximum per‐parameter gradients (`m_max_t`), maintains normalized gradient norms (`m_norm_t`, `v_norm_t`), and applies thresholded clipping before standard Adam‐style updates. An optional cosine‐decayed “death rate” warms up the stability controls over `t_max` steps. Features include decoupled weight decay, optional maximization (ascent), periodic reset of moment buffers, gradient clipping, EMA of weights, mixed‐precision loss scaling, and gradient accumulation.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates.
* **`betas`** *(tuple of two floats, default=(0.9, 0.999))*: Exponential decay rates for the first and second moment estimates.
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in denominators.
* **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 regularization applied decoupled after updates.
* **`gamma1`** *(float, default=0.7)*: Decay rate for the exponential moving average of gradient norms.
* **`gamma2`** *(float, default=0.9)*: Decay rate for the exponential moving average of squared gradient norms.
* **`theta`** *(float, default=0.999)*: Decay rate for tracking the maximum absolute gradient per step.
* **`t_max`** *(int or None, default=None)*: Total steps for cosine‐decayed warmup of stability controls; if `None`, no warmup.
* **`eta_min`** *(float, default=0.5)*: Minimum decay fraction at end of warmup when `t_max` is set.
* **`update_proj_gap`** *(int, default=1000)*: Steps between periodic resets of moment buffers and clipping thresholds.
* **`maximize`** *(bool, default=False)*: If `True`, performs gradient ascent instead of descent.
* **`clipnorm`** *(float, optional)*: Clip gradients by global norm before any update.
* **`clipvalue`** *(float, optional)*: Clip gradients by value before any update.
* **`global_clipnorm`** *(float, optional)*: Clip gradients by global norm across all parameters.
* **`use_ema`** *(bool, default=False)*: Maintain an Exponential Moving Average of model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for EMA updates when `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) at which model weights are overwritten with EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss in mixed‐precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of micro‐batches to accumulate before applying an update.
* **`name`** *(str, default="stablespam")*: Optional name prefix for optimizer variables.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.spam import StableSPAM

# Instantiate the StableSPAM optimizer with warmup and stability controls
optimizer = StableSPAM(
    learning_rate=2e-3,
    betas=(0.9, 0.999),
    epsilon=1e-8,
    weight_decay=1e-2,
    gamma1=0.6,
    gamma2=0.95,
    theta=0.995,
    t_max=10000,
    eta_min=0.1,
    update_proj_gap=500,
    maximize=False,
    use_ema=True,
    ema_momentum=0.99,
    gradient_accumulation_steps=2
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```

# ASGD

**Overview**:

The `ASGD` (Adaptive Step‐Size Gradient Descent) optimizer dynamically adjusts its learning rate based on changes in parameter and gradient norms across iterations. At each step, it computes the Euclidean norm of all trainable parameters and gradients as a group, then uses differences in those norms to modulate a per‐variable learning rate. Optional decoupled weight decay, maximization (gradient ascent), and mixed‐precision or distributed features (gradient accumulation, EMA) are supported.

**Parameters**:

* **`learning_rate`** *(float, default=1e-2)*: Initial base learning rate for all parameters.
* **`epsilon`** *(float, default=1e-5)*: Small constant added to adjusted learning rate to avoid division by zero.
* **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 regularization. When `weight_decouple=True`, decay is applied multiplicatively after each update; otherwise it is added to gradients.
* **`amplifier`** *(float, default=0.02)*: Scaling factor for amplifying the ratio between new and previous learning rate when computing an unconstrained increase (`new_lr = lr * sqrt(1 + amplifier * theta)`).
* **`theta`** *(float, default=1.0)*: Initial ratio of new learning rate to old learning rate; updated each step to track how much the learning rate has changed.
* **`dampening`** *(float, default=1.0)*: Factor in the denominator when constraining learning‐rate increase. In effect, larger `dampening` slows down the allowed jump in learning rate.
* **`weight_decouple`** *(bool, default=True)*: If `True`, applies L2 weight decay multiplicatively (`param ← param × (1 – weight_decay × lr)`); otherwise, adds `variable × weight_decay` into the gradient.
* **`fixed_decay`** *(bool, default=False)*: When used with `weight_decouple=True`, if `True` then uses a fixed decay factor of `(1 – weight_decay)` instead of scaling by the current learning rate.
* **`maximize`** *(bool, default=False)*: If `True`, performs gradient ascent (negates gradients before update).
* **`clipnorm`** *(float, optional)*: If set, clips each gradient tensor by the given global L2 norm before any other operation.
* **`clipvalue`** *(float, optional)*: If set, clips each gradient tensor element‐wise to the range \[–`clipvalue`, `clipvalue`] before any other operation.
* **`global_clipnorm`** *(float, optional)*: If set, clips the global norm of all gradients combined across variables before any other operation.
* **`use_ema`** *(bool, default=False)*: If `True`, maintains an Exponential Moving Average (EMA) of model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for updating EMA weights when `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Number of optimizer steps between overwriting model weights with EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Factor by which to scale the loss for mixed‐precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of micro‐batches to accumulate gradients over before applying an update.
* **`name`** *(str, default="asgd")*: Optional name prefix for all optimizer‐related variables.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.asgd import ASGD

# Instantiate the ASGD optimizer
optimizer = ASGD(
    learning_rate=5e-3,
    epsilon=1e-6,
    weight_decay=1e-2,
    amplifier=0.05,
    theta=1.0,
    dampening=1.0,
    weight_decouple=True,
    fixed_decay=False,
    maximize=False
)

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile with ASGD
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# SignSGD

**Overview**:

The `SignSGD` optimizer updates parameters by taking the sign of a momentum‐smoothed gradient, effectively moving each weight in the positive or negative direction of the gradient. This simple scheme can improve robustness to noisy gradients and reduce communication in distributed settings. Optional weight decay (either decoupled or standard), maximization (gradient ascent), gradient clipping, EMA of weights, and gradient accumulation are also supported.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Step size for each parameter update.
* **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 regularization. If `weight_decouple=True`, decay is applied multiplicatively after each update; otherwise it is added to gradients.
* **`momentum`** *(float, default=0.9)*: Momentum factor (0 ≤ `momentum` < 1). The buffer is updated and the sign of this buffer determines the update direction.
* **`weight_decouple`** *(bool, default=True)*: If `True`, applies weight decay after the sign update. If `False` and `weight_decay>0`, adds `param × weight_decay` into the gradient.
* **`maximize`** *(bool, default=False)*: If `True`, performs gradient ascent by negating the gradient before computing the sign.
* **`clipnorm`** *(float, optional)*: Clip each gradient tensor by its L2 norm before momentum accumulation.
* **`clipvalue`** *(float, optional)*: Clip each gradient tensor element‐wise to the range \[–`clipvalue`, `clipvalue`] before momentum accumulation.
* **`global_clipnorm`** *(float, optional)*: Clip the global norm of all gradients before momentum accumulation.
* **`use_ema`** *(bool, default=False)*: If `True`, maintains an Exponential Moving Average (EMA) of model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for EMA updates of weights when `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Number of optimizer steps between overwriting model weights with EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Factor to scale the loss for mixed‐precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of micro‐batches to accumulate gradients before applying an update.
* **`name`** *(str, default="signsgd")*: Optional name prefix for optimizer variables.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.signsgd import SignSGD

# Instantiate the SignSGD optimizer
optimizer = SignSGD(
    learning_rate=1e-3,
    weight_decay=1e-4,
    momentum=0.9,
    weight_decouple=True,
    maximize=False,
    use_ema=True,
    ema_momentum=0.99,
    gradient_accumulation_steps=2
)

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile with SignSGD
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# SGDSaI

**Overview**:

The `SGDSaI` optimizer enhances standard SGD by adapting the update magnitude with a per-parameter signal-to-noise ratio (SNR). After an initial “warmup” pass that computes the SNR for each parameter, subsequent updates scale the usual momentum‐smoothed gradient by this SNR. Optional decoupled weight decay, gradient clipping, EMA of weights, and gradient accumulation are supported.

**Parameters**:

* **`learning_rate`** *(float, default=1e-2)*: Base step size for parameter updates.
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability when computing SNR.
* **`weight_decay`** *(float, default=1e-2)*: L2 regularization coefficient. When `weight_decouple=True`, decay is applied multiplicatively after update; otherwise it is added to the gradient.
* **`momentum`** *(float, default=0.9)*: Momentum factor for smoothing gradients.
* **`weight_decouple`** *(bool, default=True)*: If `True`, applies weight decay as `param ← param * (1 – weight_decay * lr)` after the update; if `False` and `weight_decay>0`, adds `param * weight_decay` into the gradient.
* **`maximize`** *(bool, default=False)*: If `True`, performs gradient ascent by negating gradients before updates.
* **`clipnorm`** *(float, optional)*: Clips each gradient tensor by its L2 norm before momentum accumulation.
* **`clipvalue`** *(float, optional)*: Clips gradient values element-wise to `[–clipvalue, clipvalue]` before momentum accumulation.
* **`global_clipnorm`** *(float, optional)*: Clips the global norm of all gradients before momentum accumulation.
* **`use_ema`** *(bool, default=False)*: If `True`, maintains an Exponential Moving Average of model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for EMA updates when `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Number of optimizer steps between overwriting model weights with EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Factor by which to scale the loss for mixed-precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of micro-batches to accumulate gradients before applying an update.
* **`name`** *(str, default="sgdsai")*: Optional name prefix for optimizer variables.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.sgdsai import SGDSaI

# 1. Instantiate SGDSaI with momentum and weight decay
optimizer = SGDSaI(
    learning_rate=1e-2,
    epsilon=1e-8,
    weight_decay=1e-3,
    momentum=0.8,
    weight_decouple=True,
    maximize=False,
    use_ema=True,
    ema_momentum=0.99,
    gradient_accumulation_steps=4
)

# 2. Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 3. Compile with SGDSaI
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```

# VSGD

**Overview**:

The `VSGD` optimizer applies a variance‐stabilized SGD update, maintaining running estimates of gradient statistics to adaptively scale updates. It tracks parameter‐specific accumulators (`mug`, `bg`, `bhg`) and uses time‐decayed learning rates influenced by two decay exponents (`tau1`, `tau2`). Optional decoupled weight decay, maximization (ascent), gradient clipping, EMA of weights, mixed‐precision loss scaling, and gradient accumulation are supported.

**Parameters**:

* **`learning_rate`** *(float, default=1e-1)*: Base step size for parameter updates.
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability when dividing by the estimated variance term.
* **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 regularization. When `weight_decouple=True`, applied as `param = param * (1 – weight_decay * lr)` before the update; otherwise added to gradient.
* **`weight_decouple`** *(bool, default=True)*: If `True`, uses decoupled weight decay (as in AdamW) before gradient scaling; if `False`, adds `param * weight_decay` into the gradient.
* **`ghattg`** *(float, default=30.0)*: A constant used to initialize and scale the accumulator `bhg` related to gradient history.
* **`ps`** *(float, default=1e-8)*: A small positive constant used in the initial variance computations and accumulator updates.
* **`tau1`** *(float, default=0.81)*: Exponent for time‐decay of the accumulator `bg`. Controls how rapidly past gradient variance estimates are discounted in `bg`.
* **`tau2`** *(float, default=0.9)*: Exponent for time‐decay of the accumulator `bhg`. Controls how rapidly past gradient‐squared terms are discounted in `bhg`.
* **`maximize`** *(bool, default=False)*: If `True`, performs gradient ascent by negating the gradient before computing statistics and updates.
* **`clipnorm`** *(float, optional)*: If specified, clips each gradient tensor by its L2 norm before any other processing.
* **`clipvalue`** *(float, optional)*: If specified, clips gradient tensor values element‐wise to \[–`clipvalue`, `clipvalue`] before any other processing.
* **`global_clipnorm`** *(float, optional)*: If specified, clips the global norm of all gradients combined before any other processing.
* **`use_ema`** *(bool, default=False)*: If `True`, maintains an Exponential Moving Average of model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for updating EMA weights when `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Number of optimizer steps between overwriting model weights with EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Scaling factor for the loss in mixed‐precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of micro‐batches to accumulate gradients before applying an update.
* **`name`** *(str, default="vsgd")*: Optional name prefix for optimizer variables.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.vsgd import VSGD

# Instantiate VSGD optimizer with custom settings
optimizer = VSGD(
    learning_rate=0.1,
    epsilon=1e-8,
    weight_decay=1e-3,
    weight_decouple=True,
    ghattg=20.0,
    ps=1e-7,
    tau1=0.8,
    tau2=0.9,
    maximize=False,
    use_ema=True,
    ema_momentum=0.99,
    gradient_accumulation_steps=2
)

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile with VSGD
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```

# AdamC

**Overview**:

The `AdamC` optimizer is an Adam variant that supports decoupled weight decay, optional fixed decay, and an AMS-bound option. It maintains exponential moving averages of gradients and squared gradients, with optional clipping and maximization support. When `ams_bound=True`, it tracks the maximum of second-moment estimates to provide a more stable denominator. Additional features include decoupled or standard weight decay, optional gradient clipping, EMA of weights, mixed-precision loss scaling, and gradient accumulation.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates.
* **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates (moving average of gradients).
* **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates (moving average of squared gradients).
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in denominator.
* **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 regularization. When `weight_decouple=True`, applied decoupled after moment updates; otherwise added to gradient.
* **`weight_decouple`** *(bool, default=True)*: If `True`, applies decoupled weight decay (`param = param * (1 – weight_decay * lr)` or fixed); if `False`, adds `param * weight_decay` into the gradient.
* **`fixed_decay`** *(bool, default=False)*: When using decoupled decay, if `True` applies a fixed factor independent of learning rate; if `False`, scales decay by current learning rate.
* **`ams_bound`** *(bool, default=False)*: If `True`, tracks the maximum of the second-moment estimates (`max_exp_avg_sq`) and uses it in denominator for a more stable update (AMSBound variant).
* **`maximize`** *(bool, default=False)*: If `True`, performs gradient ascent by negating gradients before updates.
* **`clipnorm`** *(float, optional)*: If set, clips each gradient tensor by its L2 norm before moment updates.
* **`clipvalue`** *(float, optional)*: If set, clips gradient tensor values element-wise to \[–clipvalue, clipvalue] before moment updates.
* **`global_clipnorm`** *(float, optional)*: If set, clips the global norm of all gradients before moment updates.
* **`use_ema`** *(bool, default=False)*: If `True`, maintains an Exponential Moving Average of model weights alongside updates.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for updating EMA weights when `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Steps between overwriting model weights with EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss in mixed-precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of micro-batches to accumulate before applying an update.
* **`name`** *(str, default="adamc")*: Optional name prefix for optimizer variables.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.adamc import AdamC

# Instantiate AdamC optimizer
optimizer = AdamC(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=1e-2,
    weight_decouple=True,
    fixed_decay=False,
    ams_bound=True,
    maximize=False,
    use_ema=True,
    ema_momentum=0.99,
    gradient_accumulation_steps=2
)

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile with AdamC
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AdamWSN

**Overview**:

The `AdamWSN` optimizer is a variant of Adam that integrates subset-based second-moment estimation (subset normalization) for large parameter tensors. When enabled (`sn=True`), it partitions each tensor into subsets of size `subset_size` (or a computed divisor) and computes second-moment updates over those subsets rather than element-wise. This can reduce memory or computation overhead and provide group-wise normalization. It also supports decoupled or standard weight decay, optional fixed decay, maximization (ascent), gradient clipping, EMA of weights, mixed-precision loss scaling, and gradient accumulation.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates.
* **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates (moving average of gradients).
* **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates (moving average of squared gradients or subset sums).
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability when dividing by the estimated second moment.
* **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 regularization. When `weight_decouple=True`, applied decoupled after moment updates; otherwise added to the gradient before updating.
* **`weight_decouple`** *(bool, default=True)*: If `True`, applies decoupled weight decay (`param = param * (1 – weight_decay * lr)` or fixed); if `False`, adds `param * weight_decay` into the gradient.
* **`fixed_decay`** *(bool, default=False)*: When using decoupled decay, if `True`, applies a fixed factor independent of the current learning rate; if `False`, scales decay by current learning rate.
* **`subset_size`** *(int, default=-1)*: Desired subset size for grouping elements when computing second-moment updates. If >0, used directly; if ≤0, a divisor close to sqrt(size) is computed.
* **`sn`** *(bool, default=False)*: If `True`, enable subset normalization: partition each tensor into subsets of length `subset_size` (or computed), reshape gradient accordingly, and compute second-moment update per subset. If `False`, falls back to element-wise squared gradients.
* **`maximize`** *(bool, default=False)*: If `True`, performs gradient ascent by negating the gradient before updates.
* **`clipnorm`** *(float, optional)*: Clip each gradient tensor by its L2 norm before moment updates.
* **`clipvalue`** *(float, optional)*: Clip gradient tensor values element-wise to \[–clipvalue, clipvalue] before moment updates.
* **`global_clipnorm`** *(float, optional)*: Clip the global norm of all gradients combined before moment updates.
* **`use_ema`** *(bool, default=False)*: If `True`, maintain an Exponential Moving Average of model weights alongside updates.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for updating EMA weights when `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Steps between overwriting model weights with EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss in mixed-precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of micro-batches to accumulate before applying an update.
* **`name`** *(str, default="adamwsn")*: Optional name prefix for optimizer variables.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.adamwsn import AdamWSN

# Instantiate AdamWSN optimizer with subset normalization enabled
optimizer = AdamWSN(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=1e-4,
    weight_decouple=True,
    fixed_decay=False,
    subset_size=256,
    sn=True,
    maximize=False,
    use_ema=True,
    ema_momentum=0.99,
    gradient_accumulation_steps=2
)

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile with AdamWSN
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=15)
```

# RangerQH_sn

**Overview**:

The `RangerQH_sn` optimizer integrates the QHAdam (Quasi-Hyperbolic Adam) algorithm with Lookahead and optional Subset Normalization (SN). QHAdam uses two “nu” hyperparameters to mix instantaneous gradients and their moving averages before scaling by the root of mixed second moments. Lookahead periodically interpolates fast weights toward a slow weight buffer every `k` steps. When `sn=True`, second‐moment statistics are computed over subsets of each tensor to reduce per-element overhead.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates.
* **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment moving average in QHAdam.
* **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment moving average in QHAdam.
* **`epsilon`** *(float, default=1e-8)*: Small constant added to the denominator for numerical stability.
* **`weight_decay`** *(float, default=0.0)*: L2 regularization coefficient. If `decouple_weight_decay=True`, decay is applied to parameters independently of gradients; otherwise it is added into the gradient before update.
* **`nus`** *(tuple of two floats, default=(0.7, 1.0))*: QHAdam mixing coefficients `(nu1, nu2)` that weight the contribution of the moving averages vs. instant values for numerator and denominator.
* **`k`** *(int, default=6)*: Number of steps between Lookahead synchronizations.
* **`alpha`** *(float, default=0.5)*: Lookahead interpolation factor—fraction by which the slow buffer moves toward fast weights at each synchronization.
* **`decouple_weight_decay`** *(bool, default=False)*: If `True`, applies decoupled weight decay; otherwise includes weight decay in the gradient.
* **`subset_size`** *(int, default=-1)*: Desired subset length for Subset Normalization. If > 0, used directly; if ≤ 0, a divisor near √(tensor\_size) is computed.
* **`sn`** *(bool, default=True)*: Whether to enable Subset Normalization—compute second moments over subsets of elements; if `False`, uses per‐element squared gradients.
* **`clipnorm`** *(float, optional)*: Clip each gradient tensor by its L2 norm before any computation.
* **`clipvalue`** *(float, optional)*: Clip gradient values element‐wise to \[–clipvalue, clipvalue] before any computation.
* **`global_clipnorm`** *(float, optional)*: Clip the global norm of all gradients before any computation.
* **`use_ema`** *(bool, default=False)*: Maintain an Exponential Moving Average of model weights alongside updates.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for updating EMA weights when `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Steps between overwriting model weights with EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss in mixed‐precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of micro‐batches to accumulate before applying an update.
* **`name`** *(str, default="rangerqh\_sn")*: Optional name prefix for optimizer variables.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.rangerqh_sn import RangerQH_sn

# Instantiate RangerQH_sn with QHAdam, Lookahead, and subset normalization
optimizer = RangerQH_sn(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=1e-4,
    nus=(0.7, 1.0),
    k=6,
    alpha=0.5,
    decouple_weight_decay=True,
    subset_size=256,
    sn=True,
    use_ema=True,
    ema_momentum=0.99,
    gradient_accumulation_steps=2
)

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile with RangerQH_sn
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```

# RangerVA_sn

**Overview**:

The `RangerVA_sn` optimizer extends RAdam with Variance-Aware transformation, adaptive moment rectification (AMSGrad), Lookahead, and optional Subset Normalization (SN). It supports alternative gradient transformers (e.g., square or absolute), smooths the denominator via a customizable activation (e.g., Softplus), and maintains both exponential moving averages of gradients and second moments (with an AMS bound). Lookahead synchronizes fast and slow weights every `k` steps. When `sn=True`, second‐moment statistics are computed over subsets of each tensor to reduce per‐element overhead.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates.
* **`beta1`** *(float, default=0.95)*: Decay rate for the first moment moving average.
* **`beta2`** *(float, default=0.999)*: Decay rate for the second moment moving average.
* **`epsilon`** *(float, default=1e-5)*: Small constant added to denominators for numerical stability.
* **`weight_decay`** *(float, default=0)*: L2 regularization coefficient applied to model weights.
* **`alpha`** *(float, default=0.5)*: Lookahead interpolation factor—fraction by which the slow buffer moves toward fast weights at each synchronization.
* **`k`** *(int, default=6)*: Number of steps between Lookahead synchronization events.
* **`n_sma_threshhold`** *(int, default=5)*: Minimum value of the variance statistic (N\_sma) to apply the rectified update; below this threshold, falls back to unrectified update.
* **`amsgrad`** *(bool, default=True)*: If `True`, maintains the maximum of all second-moment estimates for a stable denominator.
* **`transformer`** *(str, default="softplus")*: Type of activation to smooth the denominator; `"softplus"` applies a Softplus transform, otherwise falls back to sqrt.
* **`smooth`** *(float, default=50)*: Smoothness parameter (beta) or threshold for the Softplus transformer when `transformer="softplus"`.
* **`grad_transformer`** *(str, default="square")*: How to transform raw gradients for second-moment updates; options are `"square"` (gradient²) or `"abs"` (|gradient|).
* **`subset_size`** *(int, default=-1)*: Desired subset length for Subset Normalization. If > 0, used directly; if ≤ 0, computes a divisor near √(tensor\_size).
* **`sn`** *(bool, default=True)*: Whether to enable Subset Normalization—compute second moments over subsets rather than per-element.
* **`clipnorm`** *(float, optional)*: Clip each gradient tensor by its L2 norm before updates.
* **`clipvalue`** *(float, optional)*: Clip gradient values element-wise to \[–clipvalue, clipvalue] before updates.
* **`global_clipnorm`** *(float, optional)*: Clip the global norm of all gradients before updates.
* **`use_ema`** *(bool, default=False)*: If `True`, maintain an Exponential Moving Average of model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for updating EMA weights when `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Steps between overwriting model weights with EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss in mixed‐precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of micro‐batches to accumulate before applying an update.
* **`name`** *(str, default="rangerva\_sn")*: Optional name prefix for optimizer variables.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.rangerva_sn import RangerVA_sn

# Instantiate RangerVA_sn with Softplus smoothing and square-gradient transformer
optimizer = RangerVA_sn(
    learning_rate=1e-3,
    beta1=0.95,
    beta2=0.999,
    epsilon=1e-5,
    weight_decay=1e-4,
    alpha=0.5,
    k=6,
    n_sma_threshhold=5,
    amsgrad=True,
    transformer='softplus',
    smooth=50,
    grad_transformer='square',
    subset_size=256,
    sn=True,
    use_ema=True,
    ema_momentum=0.99,
    gradient_accumulation_steps=2
)

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile with RangerVA_sn
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```

# AdaMuon

**Overview**:

The `AdaMuon` optimizer blends Nesterov‐accelerated adaptive moment estimation with a “zero‐power” gradient normalization and optional decoupled AdamW updates on a secondary parameter set. For tensors of rank ≥ 2 (the “muon” group), it applies Nesterov momentum combined with a Newton–Schulz‐based normalization (`ns_steps`), distributing updates across devices. Remaining parameters are handled by a standard or decoupled AdamW update. Optional per‐layer learning‐rate adjustment based on tensor shape further refines the step size.

**Parameters**:

* **`params`** *(list of `tf.Variable`)*: Primary parameter list for Muon updates; any additional `adamw_params` are added to this list.
* **`learning_rate`** *(float, default=1e-3)*: Base learning rate for Muon‐group parameters.
* **`beta1`** *(float, default=0.9)*: Nesterov momentum decay factor for Muon updates.
* **`beta2`** *(float, default=0.999)*: Second‐moment decay for Muon‐group zero‐power normalization.
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in denominators.
* **`weight_decay`** *(float, default=1e-2)*: L2 regularization coefficient for Muon updates (decoupled if `weight_decouple=True`).
* **`weight_decouple`** *(bool, default=True)*: If `True`, apply decoupled weight decay to Muon parameters; otherwise add to gradient.
* **`nesterov`** *(bool, default=True)*: If `True`, use Nesterov‐style lookahead in momentum accumulation.
* **`ns_steps`** *(int, default=5)*: Number of Newton–Schulz iterations for zero‐power gradient normalization.
* **`use_adjusted_lr`** *(bool, default=False)*: If `True`, scale per‐tensor learning rates by a factor √(max(input, output) / min(input, output)); otherwise use a fallback factor.
* **`adamw_params`** *(list of `tf.Variable` or `None`, default=None)*: Secondary parameter list to update via AdamW.
* **`adamw_betas`** *(tuple of two floats, default=(0.9, 0.999))*: `(beta1, beta2)` for the AdamW update on `adamw_params`.
* **`adamw_lr`** *(float, default=3e-4)*: Learning rate for the AdamW update on `adamw_params`.
* **`adamw_wd`** *(float, default=0.0)*: Weight‐decay coefficient for the AdamW update.
* **`maximize`** *(bool, default=False)*: If `True`, performs gradient ascent instead of descent on both groups.
* **`clipnorm`** *(float, optional)*: Clip each gradient tensor by its L2 norm before any update.
* **`clipvalue`** *(float, optional)*: Clip gradient values element‐wise to \[–clipvalue, clipvalue] before any update.
* **`global_clipnorm`** *(float, optional)*: Clip the global norm of all gradients before any update.
* **`use_ema`** *(bool, default=False)*: If `True`, maintain an Exponential Moving Average of all model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for updating EMA weights when `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Steps between overwriting model weights with EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss in mixed‐precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of micro‐batches to accumulate before applying an update.
* **`name`** *(str, default="adamuon")*: Optional name prefix for optimizer variables.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.muon import AdaMuon

# Define model parameters
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10)
])
params = model.trainable_variables

# Instantiate AdaMuon with a small AdamW secondary group
optimizer = AdaMuon(
    params=params,
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=1e-2,
    weight_decouple=True,
    nesterov=True,
    ns_steps=5,
    use_adjusted_lr=True,
    adamw_params=None,            # e.g., no secondary group
    adamw_betas=(0.9, 0.999),
    adamw_lr=3e-4,
    adamw_wd=0.01,
    maximize=False,
    clipnorm=1.0,
    use_ema=True,
    ema_momentum=0.99,
    gradient_accumulation_steps=4
)

# Compile and train
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# SPlus

**Overview**:

The `SPlus` optimizer is a Stable Whitening method that adapts updates by re‐whitening activations for efficient neural network training. It maintains momentum of gradients and an exponential moving average (EMA) of parameters. For 2D tensors (e.g., weight matrices), it tracks per‑dimension covariance (`sides`) and their eigenbases (`q_sides`), allowing sign‑based updates along principal directions. It supports decoupled or standard weight decay, optional maximization, and switching between training and evaluation modes via EMA.

**Parameters**:

* **`params`** *(iterable of `tf.Variable`)*: Parameters to optimize.
* **`learning_rate`** *(float, default=1e-1)*: Base step size for updates.
* **`betas`** *(tuple of two floats, default=(0.9, 0.999))*: `(beta1, beta2)` decay rates for the momentum (`m`) and EMA of parameters.
* **`epsilon`** *(float, default=1e-30)*: Small constant added for numerical stability.
* **`weight_decay`** *(float, default=1e-2)*: L2 regularization coefficient.
* **`weight_decouple`** *(bool, default=True)*: If `True`, applies decoupled weight decay (`param *= 1 – wd * lr`); otherwise adds `param * wd` to the gradient.
* **`fixed_decay`** *(bool, default=False)*: When decoupled, if `True` uses a fixed decay factor independent of learning rate.
* **`ema_rate`** *(float, default=0.999)*: Decay rate for the EMA of parameters.
* **`inverse_steps`** *(int, default=100)*: Number of steps between eigendecomposition of `sides` to update `q_sides`.
* **`nonstandard_constant`** *(float, default=1e-3)*: Scale factor for learning rate on non‑2D tensors or large dimensions.
* **`max_dim`** *(int, default=10000)*: Maximum dimension size for which whitening is applied; larger dims are skipped.
* **`maximize`** *(bool, default=False)*: If `True`, performs gradient ascent rather than descent.
* **`train_mode`** *(bool, default=True)*: If `False`, uses EMA values for parameters (evaluation mode).
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional floats)*: Gradient clipping thresholds.
* **`use_ema`**, **`ema_momentum`**, **`ema_overwrite_frequency`** *(optional)*: EMA of weights settings.
* **`loss_scale_factor`**, **`gradient_accumulation_steps`** *(optional)*: Mixed‑precision scaling and accumulation.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.splus import SPlus

# 1. Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)
])
params = model.trainable_variables

# 2. Instantiate SPlus optimizer
optimizer = SPlus(
    params=params,
    learning_rate=0.1,
    betas=(0.9, 0.999),
    epsilon=1e-30,
    weight_decay=1e-2,
    weight_decouple=True,
    fixed_decay=False,
    ema_rate=0.999,
    inverse_steps=100,
    nonstandard_constant=1e-3,
    max_dim=10000,
    maximize=False,
    train_mode=True
)

# 3. Compile model
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 4. Train
model.fit(train_dataset, validation_data=val_dataset, epochs=20)

# 5. Switch to evaluation mode (using EMA weights)
optimizer.eval()
# ... run validation/inference ...
optimizer.train()  # switch back to training
```

# EmoNavi

**Overview**:

The `EmoNavi` optimizer is an emotion-driven variant of AdamW that dynamically interpolates model parameters toward a “shadow” copy based on the network’s recent loss behavior. It maintains a short- and long-term EMA of the loss to compute an emotional scalar, which governs occasional “mood-based” look-ahead steps. Standard AdamW updates follow for every step.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates.
* **`betas`** *(tuple of two floats, default=(0.9, 0.999))*: (`beta1`, `beta2`) decay rates for first and second moment estimates.
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in denominator.
* **`weight_decay`** *(float, default=1e-2)*: L2 penalty coefficient.
* **`weight_decouple`** *(bool, default=True)*: If `True`, applies decoupled weight decay (AdamW style); otherwise adds `param*wd` to gradient.
* **`fixed_decay`** *(bool, default=False)*: When decoupled, if `True` uses a fixed decay factor rather than scaling by `lr`.
* **`shadow_weight`** *(float, default=0.05)*: Interpolation rate for shadow parameter updates when emotion triggers.
* **`maximize`** *(bool, default=False)*: If `True`, performs gradient ascent rather than descent.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional floats)*: Gradient clipping thresholds by norm or value.
* **`use_ema`**, **`ema_momentum`**, **`ema_overwrite_frequency`** *(optional)*: EMA settings for optimizer state (not to be confused with loss EMAs).
* **`loss_scale_factor`**, **`gradient_accumulation_steps`** *(optional)*: Mixed-precision and accumulation.
* **`name`** *(str, default="emonavi")*: Name prefix for optimizer variables.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.emonavi import EmoNavi

# 1) Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# 2) Instantiate EmoNavi optimizer
optimizer = EmoNavi(
    learning_rate=1e-3,
    betas=(0.9, 0.999),
    epsilon=1e-8,
    weight_decay=1e-2,
    weight_decouple=True,
    fixed_decay=False,
    shadow_weight=0.05,
    maximize=False,
    clipnorm=1.0
)

# 3) Prepare loss function and metrics
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc   = tf.keras.metrics.SparseCategoricalAccuracy()

# 4) Custom training loop
epochs = 5
for epoch in range(epochs):
    # -- Training --
    train_acc.reset_states()
    for x_batch, y_batch in train_ds:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        # optimizer.apply_gradients expects (grads_and_vars, loss)
        grads_and_vars = zip(grads, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars, loss)
        train_acc.update_state(y_batch, logits)

    # -- Validation --
    val_acc.reset_states()
    for x_batch, y_batch in val_ds:
        logits = model(x_batch, training=False)
        val_acc.update_state(y_batch, logits)

    print(
        f"Epoch {epoch+1}/{epochs} — "
        f"train_acc={train_acc.result():.4f}, val_acc={val_acc.result():.4f}"
    )
```

# EmoNavi_sn

**Overview**:

`EmoNavi_sn` augments `EmoNavi` with Subset Normalization (SN) on the second moment in AdamW steps. It shares the same emotion-driven shadow interpolation logic but, for each gradient, computes its variance over fixed-size subsets, which can improve stability on large vector parameters. Mood-triggered shadow updates and SN-powered AdamW follow each mini-batch.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step size.
* **`betas`** *(tuple of two floats, default=(0.9, 0.999))*: Decay rates for moment estimates.
* **`epsilon`** *(float, default=1e-8)*: Numerical stability constant.
* **`weight_decay`** *(float, default=1e-2)*: L2 penalty coefficient.
* **`weight_decouple`** *(bool, default=True)*: Use AdamW’s decoupled weight decay.
* **`fixed_decay`** *(bool, default=False)*: Use fixed rather than `lr`-scaled decay when decoupled.
* **`shadow_weight`** *(float, default=0.05)*: Shadow interpolation rate on emotional triggers.
* **`sn`** *(bool, default=True)*: If `True`, applies Subset Normalization on the squared gradients for the second‐moment buffer.
* **`maximize`** *(bool, default=False)*: If `True`, maximizes the loss.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional floats)*: Gradient clipping settings.
* **`use_ema`**, **`ema_momentum`**, **`ema_overwrite_frequency`** *(optional)*: EMA usage for optimizer.
* **`loss_scale_factor`**, **`gradient_accumulation_steps`** *(optional)*: Mixed precision and accumulation.
* **`name`** *(str, default="emonavi\_sn")*: Optimizer name prefix.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.emonavi import EmoNavi_sn

# 1) Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# 2) Instantiate EmoNavi_sn optimizer
optimizer = EmoNavi_sn(
    learning_rate=1e-3,
    betas=(0.9, 0.999),
    epsilon=1e-8,
    weight_decay=1e-2,
    weight_decouple=True,
    fixed_decay=False,
    shadow_weight=0.1,
    sn=True,                 # enable Subset Normalization
    maximize=False,
    clipvalue=0.5
)

# 3) Prepare loss function and metrics
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc   = tf.keras.metrics.SparseCategoricalAccuracy()

# 4) Custom training loop
epochs = 8
for epoch in range(epochs):
    train_acc.reset_states()
    for x_batch, y_batch in train_ds:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        grads_and_vars = zip(grads, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars, loss)
        train_acc.update_state(y_batch, logits)

    val_acc.reset_states()
    for x_batch, y_batch in val_ds:
        logits = model(x_batch, training=False)
        val_acc.update_state(y_batch, logits)

    print(
        f"Epoch {epoch+1}/{epochs} — "
        f"train_acc={train_acc.result():.4f}, val_acc={val_acc.result():.4f}"
    )
```

# EmoLynx

**Overview**:

The `EmoLynx` optimizer combines the lightweight “sign-based” update of Lion/Tiger with the emotion-driven shadow mechanism of EmoNavi.  In each step it first softly interpolates parameters toward a “shadow” copy based on the network’s recent “emotional” loss history, then applies a sign-based gradient update blended with a running average.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates.
* **`betas`** *(tuple of two floats, default=(0.9, 0.99))*:

  * First element is the interpolation coefficient for the blended gradient (`β₁`).
  * Second element is the EMA coefficient for the running average of past gradients (`β₂`).
* **`epsilon`** *(float, default=1e-8)*: Small constant added for numerical stability.
* **`weight_decay`** *(float, default=1e-2)*: L2 penalty coefficient.
* **`weight_decouple`** *(bool, default=True)*: Whether to apply weight decay in a decoupled manner (as in AdamW).
* **`fixed_decay`** *(bool, default=False)*: If true, use a fixed decay factor independent of the learning rate.
* **`shadow_weight`** *(float, default=0.05)*: Interpolation rate between parameters and their “shadow” copy during an “emotion” step.
* **`maximize`** *(bool, default=False)*: If true, gradients are negated to maximize rather than minimize.
* **`clipnorm`** *(float, optional)*: If set, clip all gradients to have norm ≤ this value.
* **`clipvalue`** *(float, optional)*: If set, clip each gradient element to the range \[–clipvalue, clipvalue].
* **`global_clipnorm`** *(float, optional)*: If set, clip all gradients by the global norm.
* **`use_ema`** *(bool, default=False)*: Whether to maintain an exponential moving average of parameters for evaluation.
* **`ema_momentum`** *(float, default=0.99)*: Decay rate for the parameter EMA.
* **`ema_overwrite_frequency`** *(int, optional)*: How often (in steps) to overwrite the EMA weights with the current weights.
* **`loss_scale_factor`** *(float, optional)*: Static factor by which to scale the loss during backprop.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of batches to accumulate gradients over before applying an update.
* **`name`** *(str, default="emolynx")*: Optional name for the optimizer instance.

---

**Example Usage**:

```python
import tensorflow as tf
from optimizers.emonavi import EmoLynx

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Instantiate EmoLynx optimizer
optimizer = EmoLynx(
    learning_rate=1e-3,
    betas=(0.9, 0.99),
    epsilon=1e-8,
    weight_decay=1e-2,
    weight_decouple=True,
    fixed_decay=False,
    shadow_weight=0.05,
    maximize=False,
    clipnorm=1.0
)

# Prepare loss and metrics
loss_fn  = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc   = tf.keras.metrics.SparseCategoricalAccuracy()

# Custom training loop
for epoch in range(5):
    train_acc.reset_states()
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss   = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        # Pass (grads, vars) plus the current loss to trigger the emotion-driven update
        optimizer.apply_gradients(zip(grads, model.trainable_variables), loss)
        train_acc.update_state(y_batch, logits)

    val_acc.reset_states()
    for x_batch, y_batch in val_dataset:
        logits = model(x_batch, training=False)
        val_acc.update_state(y_batch, logits)

    print(f"Epoch {epoch+1}, "
          f"Train Acc: {train_acc.result():.4f}, "
          f"Val Acc:   {val_acc.result():.4f}")
```

# EmoFact

**Overview**:

The `EmoFact` optimizer adapts the memory efficiency of AdaFactor’s factored second-moment estimates into an emotion-driven framework.  It maintains row- and column-factorized running variances, applies an EmoNavi-style shadow interpolation based on the network’s loss “emotion,” and then rescales gradients by the factored RMS.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step size for updates.
* **`betas`** *(tuple of two floats, default=(0.9, 0.999))*: EMA coefficients for the row/column second-moment estimates.
* **`epsilon`** *(float, default=1e-8)*: Numerical stability constant.
* **`weight_decay`** *(float, default=1e-2)*: L2 penalty coefficient.
* **`weight_decouple`** *(bool, default=True)*: Decoupled weight decay (AdamW-style).
* **`fixed_decay`** *(bool, default=False)*: Fixed decay factor instead of scaling by the learning rate.
* **`shadow_weight`** *(float, default=0.05)*: Shadow-interpolation rate in the “emotion” step.
* **`maximize`** *(bool, default=False)*: If true, perform gradient ascent.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`**, **`use_ema`**, **`ema_momentum`**, **`ema_overwrite_frequency`**, **`loss_scale_factor`**, **`gradient_accumulation_steps`**, **`name`**: As in **EmoLynx**.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.emonavi import EmoFact

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Instantiate EmoFact optimizer
optimizer = EmoFact(
    learning_rate=5e-4,
    betas=(0.9, 0.999),
    epsilon=1e-8,
    weight_decay=1e-2,
    weight_decouple=True,
    fixed_decay=False,
    shadow_weight=0.1,
    maximize=False
)

# Loss and metrics
loss_fn  = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc   = tf.keras.metrics.SparseCategoricalAccuracy()

# Custom training loop
for epoch in range(10):
    train_acc.reset_states()
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss   = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables), loss)
        train_acc.update_state(y_batch, logits)

    val_acc.reset_states()
    for x_batch, y_batch in val_dataset:
        logits = model(x_batch, training=False)
        val_acc.update_state(y_batch, logits)

    print(f"Epoch {epoch+1}, "
          f"Train Acc: {train_acc.result():.4f}, "
          f"Val Acc:   {val_acc.result():.4f}")
```

# EmoFact_sn

**Overview**:

The `EmoFact_sn` optimizer extends `EmoFact` by incorporating “sign-based” updates for scalar parameters, combined with an emotion-driven shadow interpolation.  It uses factored second-moment estimates (row/column) for multi-dimensional weights, and simple sign-scaled updates for 1D parameters.  During each step, it optionally first shifts parameters toward a “shadow” copy based on recent loss “emotion,” then rescales gradients by the factored RMS (or sign for scalars) before applying the update.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step size for all updates.
* **`betas`** *(tuple of two floats, default=(0.9, 0.999))*:

  * First element is the decay rate for the row/column factored second moments (`β₁`).
  * Second element is the decay rate for the overall second-moment accumulator (`β₂`).
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
* **`weight_decay`** *(float, default=1e-2)*: L2 penalty coefficient.
* **`weight_decouple`** *(bool, default=True)*: Apply weight decay in a decoupled fashion (as in AdamW).
* **`fixed_decay`** *(bool, default=False)*: If true, use a fixed weight decay factor rather than scaling by the learning rate.
* **`shadow_weight`** *(float, default=0.05)*: Interpolation rate toward shadow parameters when emotion-driven update triggers.
* **`subset_size`** *(int, default=-1)*: Block size for sign-based updates on scalar tensors; if >0, splits 1D arrays into blocks for second-moment estimation.
* **`sn`** *(bool, default=True)*: Enable sign-based (“SN”) update mode for scalar parameters or 1D tensors.
* **`maximize`** *(bool, default=False)*: If true, gradients are negated to perform maximization.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`**, **`use_ema`**, **`ema_momentum`**, **`ema_overwrite_frequency`**, **`loss_scale_factor`**, **`gradient_accumulation_steps`**, **`name`**: Same behavior as in **EmoFact**.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.emonavi import EmoFact_sn

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Instantiate EmoFact_sn optimizer
optimizer = EmoFact_sn(
    learning_rate=2e-3,
    betas=(0.9, 0.999),
    epsilon=1e-8,
    weight_decay=1e-2,
    weight_decouple=True,
    fixed_decay=False,
    shadow_weight=0.1,
    subset_size=50,
    sn=True,
    maximize=False
)

# Prepare loss function and metrics
loss_fn  = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc   = tf.keras.metrics.SparseCategoricalAccuracy()

# Custom training loop
for epoch in range(8):
    # Training
    train_acc.reset_states()
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss   = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        # Pass loss to trigger emotion-driven shadow update
        optimizer.apply_gradients(zip(grads, model.trainable_variables), loss)
        train_acc.update_state(y_batch, logits)

    # Validation
    val_acc.reset_states()
    for x_batch, y_batch in val_dataset:
        logits = model(x_batch, training=False)
        val_acc.update_state(y_batch, logits)

    print(f"Epoch {epoch+1}, "
          f"Train Acc: {train_acc.result():.4f}, "
          f"Val Acc:   {val_acc.result():.4f}")
```

# EmoNeco

**Overview**:

The `EmoNeco` optimizer is an emotion-aware, lightweight optimizer inspired by Lion/Tiger/Lynx families and integrates a shadow-parameter interpolation mechanism driven by a small EMA ("emotion") of the loss. For multi-step training it adaptively blends parameters toward a shadow copy when recent loss dynamics indicate it, and uses a combined blended gradient / softsign / sign logic to compute updates. It supports decoupled weight decay and standard optimizer conveniences (gradient clipping, EMA of weights, etc.).

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates.
* **`betas`** *(tuple of two floats, default=(0.9, 0.99))*: Coefficients for running averages; first for the blended gradient mixture, second for the exponential moving average of gradients.
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
* **`weight_decay`** *(float, default=1e-2)*: L2 penalty coefficient applied to parameters.
* **`weight_decouple`** *(bool, default=True)*: If true, apply weight decay in a decoupled fashion (like AdamW).
* **`fixed_decay`** *(bool, default=False)*: If true, apply a fixed weight decay factor instead of scaling it by the learning rate.
* **`shadow_weight`** *(float, default=0.05)*: Interpolation rate used when updating the shadow copy after a shadow-blend step.
* **`maximize`** *(bool, default=False)*: If true, optimizer will maximize the objective (gradients are negated).
* **`clipnorm`** *(float, optional)*: Clip gradients by norm before applying updates.
* **`clipvalue`** *(float, optional)*: Clip gradients by value before applying updates.
* **`global_clipnorm`** *(float, optional)*: Clip all gradients by a global norm.
* **`use_ema`** *(bool, default=False)*: Whether to keep an exponential moving average of model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum for the EMA if `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency to overwrite model weights from EMA.
* **`loss_scale_factor`** *(float, optional)*: Factor for scaled-loss training (mixed precision).
* **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before applying an update.
* **`name`** *(str, default="emoneco")*: Name of the optimizer instance.

**Notes**:

* `EmoNeco` exposes `apply_gradients(grads_and_vars, loss)` and the example below uses a custom training loop — you must pass the current `loss` to `apply_gradients` so the optimizer can update its internal "emotion" EMA and perform shadow interpolation when appropriate.
* The optimizer internally keeps a shadow copy per parameter and will interpolate model parameters toward the shadow when recent loss dynamics indicate it (emotion-driven behavior).

**Example Usage**:

```python
import tensorflow as tf
from optimizers.emonavi import EmoNeco

# Simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Loss and metrics
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

# Instantiate EmoNeco optimizer
optimizer = EmoNeco(
    learning_rate=1e-3,
    betas=(0.9, 0.99),
    epsilon=1e-8,
    weight_decay=1e-2,
    weight_decouple=True,
    fixed_decay=False,
    shadow_weight=0.05,
    maximize=False,
    name="emoneco"
)

# Custom training loop: note we pass loss into apply_gradients so EmoNeco can update its internal EMA ("emotion")
epochs = 5
for epoch in range(epochs):
    train_acc.reset_states()
    # training
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        # Pass (grads, vars) and loss so the optimizer can update its "emotion" EMA and shadow state.
        optimizer.apply_gradients(zip(grads, model.trainable_variables), loss)
        train_acc.update_state(y_batch, logits)

    # validation
    val_acc.reset_states()
    for x_batch, y_batch in val_dataset:
        logits = model(x_batch, training=False)
        val_acc.update_state(y_batch, logits)

    print(f"Epoch {epoch+1}: Train Acc={train_acc.result():.4f}, Val Acc={val_acc.result():.4f}")
```

# EmoZeal

**Overview**:

The `EmoZeal` optimizer is an Adafactor-style, memory-friendly optimizer that combines per-tensor/per-factor second-moment tracking with an optional shadow-parameter interpolation driven by a tiny EMA-based "emotion" signal of the loss. It supports decoupled weight decay, optional shadow behavior, and mixed factor / full second-moment handling for multi-dimensional tensors.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates.
* **`betas`** *(tuple of two floats, default=(0.9, 0.999))*: Coefficients for exponential moving averages used in the optimizer (used for mean and variance tracking).
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
* **`weight_decay`** *(float, default=1e-2)*: L2 penalty coefficient applied to parameters.
* **`weight_decouple`** *(bool, default=True)*: If true, apply weight decay in a decoupled fashion (like AdamW).
* **`fixed_decay`** *(bool, default=False)*: If true, apply a fixed weight decay factor instead of scaling it by the learning rate.
* **`use_shadow`** *(bool, default=True)*: Enable shadow-parameter interpolation driven by the emotion EMA.
* **`shadow_weight`** *(float, default=0.05)*: Interpolation rate used to update the shadow copy when a shadow step occurs.
* **`maximize`** *(bool, default=False)*: If true, optimizer will maximize the objective (gradients are negated).
* **`clipnorm`** *(float, optional)*: Clip gradients by norm before applying updates.
* **`clipvalue`** *(float, optional)*: Clip gradients by value before applying updates.
* **`global_clipnorm`** *(float, optional)*: Clip all gradients by a global norm.
* **`use_ema`** *(bool, default=False)*: Whether to keep an exponential moving average of model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum for the EMA if `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency to overwrite model weights from EMA.
* **`loss_scale_factor`** *(float, optional)*: Factor for scaled-loss training (mixed precision).
* **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before applying an update.
* **`name`** *(str, default="emozeal")*: Name of the optimizer instance.

**Notes**:

* `EmoZeal` exposes `apply_gradients(grads_and_vars, loss)` — you **must** pass the current `loss` so the optimizer can update its internal emotion EMA and perform shadow interpolation when appropriate.
* For tensors with rank ≥ 2 it uses factorized statistics (rows/cols) to be VRAM efficient; for vectors it keeps per-element second moment buffers.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.emonavi import EmoZeal  # module path where your class lives

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()

# Instantiate the optimizer
opt = EmoZeal(
    learning_rate=1e-3,
    betas=(0.9, 0.999),
    epsilon=1e-8,
    weight_decay=1e-2,
    weight_decouple=True,
    use_shadow=True,
    shadow_weight=0.05,
    name="emozeal"
)

# Custom training loop — note: pass loss into apply_gradients
for epoch in range(5):
    train_acc.reset_states()
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        # PASS loss to apply_gradients so EmoZeal can update its internal EMA ("emotion")
        opt.apply_gradients(zip(grads, model.trainable_variables), loss)
        train_acc.update_state(y_batch, logits)
    print(f"Epoch {epoch+1}, Train Acc: {train_acc.result().numpy():.4f}")
```

# EmoZeal_sn

**Overview**:

`EmoZeal_sn` is the subset-normalized / "sharded-norm" variant of `EmoZeal`. It provides the same Adafactor-style and emotion-driven shadow interpolation features while optionally applying subset aggregation for second-moment statistics (useful to reduce memory and compute on very large parameter vectors). The `sn` suffix indicates support for subset normalization of per-element second moments.

**Parameters**:

All parameters from `EmoZeal`, plus:

* **`subset_size`** *(int, default=-1)*: If >0, number of elements per subset used when computing summed second-moment statistics for vectors; if -1 it derives a heuristic subset size (sqrt of vector length).
* **`sn`** *(bool, default=True)*: Enable subset-normalization mode for vector parameters (aggregated second-moment statistics).
* Other parameters (learning\_rate, betas, epsilon, weight\_decay, use\_shadow, shadow\_weight, maximize, etc.) behave the same as in `EmoZeal`.

**Notes**:

* Like `EmoZeal`, `EmoZeal_sn` exposes `apply_gradients(grads_and_vars, loss)` — you **must** pass the current `loss` for the internal emotion EMA and shadow updates.
* `subset_size` controls how vector parameters are reshaped into (n\_subsets × subset\_size) to compute reduced second-moment statistics for memory efficiency.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.emonavi import EmoZeal_sn  # module path where your class lives

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()

# Instantiate the sn variant with subset normalization enabled
opt = EmoZeal_sn(
    learning_rate=1e-3,
    betas=(0.9, 0.999),
    epsilon=1e-8,
    weight_decay=1e-2,
    weight_decouple=True,
    use_shadow=True,
    shadow_weight=0.05,
    subset_size=256,  # example subset size; set -1 to let the optimizer choose heuristically
    sn=True,
    name="emozeal_sn"
)

# Training loop — must pass loss into apply_gradients
for epoch in range(5):
    train_acc.reset_states()
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables), loss)
        train_acc.update_state(y_batch, logits)
    print(f"Epoch {epoch+1}, Train Acc: {train_acc.result().numpy():.4f}")
```

# DistributedMuon

**Overview**:

The `DistributedMuon` optimizer is a hybrid distributed optimizer designed for large-scale training. It employs **Muon-style updates** with momentum and zero-power (Newton-Schulz) normalization when `use_muon=True`, distributing parameter updates across replicas efficiently. Optionally, when Muon is disabled, it falls back to **AdamW-style updates** for stability. It supports cautious masking, Nesterov momentum, per-parameter adaptive learning rates, and EMA features.

**Parameters**:

* **`learning_rate`** *(float, default=2e-2)*: Base step size for optimization.
* **`weight_decay`** *(float, default=0.0)*: L2 penalty applied to parameters (decoupled if `weight_decouple=True`).
* **`momentum`** *(float, default=0.95)*: Momentum coefficient for Muon updates.
* **`weight_decouple`** *(bool, default=True)*: Apply weight decay in a decoupled manner (like AdamW).
* **`nesterov`** *(bool, default=True)*: Use Nesterov momentum in Muon updates.
* **`ns_steps`** *(int, default=5)*: Number of Newton-Schulz iterations for zero-power normalization in Muon.
* **`use_adjusted_lr`** *(bool, default=False)*: Whether to adapt learning rate per parameter shape.
* **`adamw_lr`** *(float, default=3e-4)*: Learning rate for AdamW fallback updates (when `use_muon=False`).
* **`adamw_betas`** *(tuple of two floats, default=(0.9, 0.95))*: Beta coefficients for first- and second-moment estimates in AdamW fallback.
* **`adamw_wd`** *(float, default=0.0)*: Weight decay for AdamW fallback updates.
* **`adamw_eps`** *(float, default=1e-10)*: Epsilon for numerical stability in AdamW denominator.
* **`use_muon`** *(bool, default=True)*: Enable Muon-style distributed updates across replicas.
* **`cautious`** *(bool, default=False)*: If true, applies a mask that suppresses update elements that disagree with the gradient sign.
* **`maximize`** *(bool, default=False)*: If true, optimizer performs gradient ascent.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional floats)*: Gradient clipping thresholds.
* **`use_ema`**, **`ema_momentum`**, **`ema_overwrite_frequency`** *(optional)*: EMA settings for tracking parameter averages.
* **`loss_scale_factor`**, **`gradient_accumulation_steps`** *(optional)*: Mixed-precision loss scaling and gradient accumulation.
* **`name`** *(str, default="distributedmuon")*: Name of the optimizer.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.muon import DistributedMuon

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Instantiate DistributedMuon optimizer within a distribution scope
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    optimizer = DistributedMuon(
        learning_rate=1e-2,
        weight_decay=1e-3,
        momentum=0.9,
        use_adjusted_lr=True,
        adamw_lr=5e-4,
        use_muon=True,
        cautious=True,
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    # Custom distributed training loop
    @tf.function
    def train_step(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables), loss)
        train_acc.update_state(y, logits)

    for epoch in range(5):
        train_acc.reset_states()
        for batch in train_dataset:
            strategy.run(train_step, args=(batch,))
        print(f"Epoch {epoch+1}: Train Acc = {train_acc.result():.4f}")
```

# DAdaptAdam_sn

**Overview**:

`DAdaptAdam_sn` is an adaptive, scale-free variant of Adam that includes an online automatic step-size scheduler (`D`-Adapt) and optional subset-normalization (SN) to reduce memory for very large tensors. The optimizer adaptively adjusts an internal scalar `d0` based on accumulated gradient statistics, and then uses this scale to modulate per-parameter updates computed from Adam-like first and second moments. When `sn=True`, second-moment statistics are computed over groups (subsets) of elements rather than elementwise to reduce memory and compute.

**Parameters**:

* **`learning_rate`** *(float, default=1.0)*: Global learning-rate multiplier. Final per-step effective scale is controlled by the internal `d0`.
* **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates.
* **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimates.
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in denominators.
* **`weight_decay`** *(float, default=0.0)*: L2 weight decay coefficient applied either decoupled or through gradients depending on `weight_decouple`.
* **`d0`** *(float, default=1e-6)*: Initial scale used by the D-Adapt mechanism.
* **`growth_rate`** *(float, default=inf)*: Maximum multiplicative growth allowed for `d0` between updates.
* **`weight_decouple`** *(bool, default=True)*: When `True`, apply decoupled weight decay (like AdamW).
* **`fixed_decay`** *(bool, default=False)*: If `True`, treat weight decay as a fixed multiplier instead of scaling it by the optimizer step.
* **`bias_correction`** *(bool, default=False)*: If `True`, the internal step calculation uses Adam-style bias correction when updating `d0`.
* **`subset_size`** *(int, default=-1)*: Subset size used for SN grouping; `-1` means automatic heuristic (sqrt of tensor size).
* **`sn`** *(bool, default=True)*: Enable subset-normalization (grouped second moments).
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional)*: Gradient clipping settings.
* **`use_ema`** *(bool, default=False)*: Track exponential moving averages of parameters.
* **`ema_momentum`** *(float, default=0.99)*: EMA momentum.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Loss scaling for mixed precision.
* **`gradient_accumulation_steps`** *(int, optional)*: Accumulate gradients for this many steps before applying an update.
* **`name`** *(str, default="dadaptadam\_sn")*: Name of the optimizer.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.dadaptadam import DAdaptAdam_sn  # assume module path

# Build model inside strategy.scope() if using distribution.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Instantiate optimizer
optimizer = DAdaptAdam_sn(
    learning_rate=1.0,
    beta1=0.9,
    beta2=0.999,
    d0=1e-6,
    subset_size=-1,  # automatic subset sizing for SN
    sn=True
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Example training step using tf.GradientTape (correctly computes loss, grads, applies grads)
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Typical loop (dataset yields (x, y))
for epoch in range(3):
    for x_batch, y_batch in train_dataset:
        loss_value = train_step(x_batch, y_batch)
```

# DAdaptAdan_sn

**Overview**:

`DAdaptAdan_sn` is a D-Adapt variant of the Adan family that combines adaptive step-size scheduling with Adan-style moments (first moment, running difference moments, and a third moment for squared differences). It maintains an internal adaptive scalar `d0` that is computed from accumulated gradient statistics and uses subset-normalization (SN) to compute grouped second moments for memory efficiency. This optimizer is tailored for scenarios where stable adaptive scaling and memory-efficient normalization matter.

**Parameters**:

* **`learning_rate`** *(float, default=1.0)*: Global learning-rate multiplier; the D-Adapt internal scale `d0` modulates effective step sizes.
* **`beta1`** *(float, default=0.98)*: Decay for the primary first moment.
* **`beta2`** *(float, default=0.92)*: Decay used for the difference-derived first moment (Adan-style).
* **`beta3`** *(float, default=0.99)*: Decay for the second-moment estimate used by Adan.
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
* **`weight_decay`** *(float, default=0.0)*: L2 weight decay coefficient.
* **`d0`** *(float, default=1e-6)*: Initial D-Adapt scalar.
* **`growth_rate`** *(float, default=inf)*: Maximum allowed growth factor for `d0`.
* **`weight_decouple`** *(bool, default=True)*: Use decoupled weight decay when `True`.
* **`fixed_decay`** *(bool, default=False)*: If `True`, weight decay is fixed rather than scaled by step.
* **`subset_size`** *(int, default=-1)*: SN grouping size; `-1` selects an automatic heuristic.
* **`sn`** *(bool, default=True)*: Enable subset-normalization.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional)*: Gradient clipping options.
* **`use_ema`** *(bool, default=False)*: Track EMA of parameters.
* **`ema_momentum`** *(float, default=0.99)*: EMA momentum.
* **`ema_overwrite_frequency`** *(int, optional)*: How often to overwrite EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Loss scaling factor for mixed precision.
* **`gradient_accumulation_steps`** *(int, optional)*: Accumulate gradients before applying.
* **`name`** *(str, default="dadaptadan\_sn")*: Name of the optimizer.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.dadaptadan import DAdaptAdan_sn  # assume module path

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

optimizer = DAdaptAdan_sn(
    learning_rate=1.0,
    beta1=0.98,
    beta2=0.92,
    beta3=0.99,
    d0=1e-6,
    subset_size=-1,
    sn=True
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for epoch in range(3):
    for x_batch, y_batch in train_dataset:
        loss_value = train_step(x_batch, y_batch)
```

# Ranger_e

**Overview**:

`Ranger_e` is a hybrid optimizer that combines RAdam-style rectified adaptive updates with Lookahead, optional Gradient Centralization (GC), optional subset-normalization (SN) for memory-efficient second-moment estimates, and an optional D-Adapt automatic step-size controller. It is intended for robust, stable training across architectures: when `use_gc` is enabled it recenters gradients (useful for convolutional and/or fully-connected layers), `sn=True` groups elements to reduce second-moment memory, Lookahead (slow buffer + interpolation) improves stability, and `DAdapt` automatically adjusts an internal scaling `d0` from observed gradient statistics.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base learning rate (global multiplier). Effective per-step scale may be modulated by D-Adapt when enabled.
* **`beta1`** *(float, default=0.95)*: Decay for the first moment (moving average of gradients).
* **`beta2`** *(float, default=0.999)*: Decay for the second moment (moving average of squared gradients or grouped second moments when SN is enabled).
* **`epsilon`** *(float, default=1e-5)*: Small constant added for numerical stability in denominators.
* **`weight_decay`** *(float, default=0)*: L2 weight decay coefficient (applied as a parameter-level subtraction inside the update step).
* **`alpha`** *(float, default=0.5)*: Lookahead interpolation factor between slow and fast weights.
* **`k`** *(int, default=6)*: Lookahead synchronization period (apply slow/fast interpolation every `k` steps).
* **`N_sma_threshhold`** *(int, default=5)*: RAdam N\_sma threshold controlling rectified vs. unrectified behavior.
* **`use_gc`** *(bool, default=True)*: Enable Gradient Centralization (GC). When enabled GC removes mean across axes for conv/fc gradients.
* **`gc_conv_only`** *(bool, default=False)*: If `True`, only apply GC to convolutional layers (checked via gradient dimensionality).
* **`subset_size`** *(int, default=-1)*: Subset size hint for subset-normalization (SN). `-1` uses an automatic heuristic (sqrt of tensor size).
* **`sn`** *(bool, default=False)*: Enable subset-normalization: group elements into subsets and compute grouped second moments to reduce memory.
* **`d0`** *(float, default=1e-6)*: Initial D-Adapt scalar (when `DAdapt=True`) that multiplies the global `learning_rate`.
* **`growth_rate`** *(float, default=inf)*: Maximum multiplicative growth for `d0` per adapt step.
* **`DAdapt`** *(bool, default=False)*: Enable the D-Adapt automatic step-size controller that adjusts `d0` from accumulated statistics.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional)*: Standard gradient clipping options passed to the base optimizer.
* **`use_ema`** *(bool, default=False)*: Whether to maintain exponential moving averages of parameters.
* **`ema_momentum`** *(float, default=0.99)*: EMA momentum if `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting weights from EMA (when used).
* **`loss_scale_factor`** *(float, optional)*: Loss scaling factor for mixed-precision workflows.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before applying an optimizer update.
* **`name`** *(str, default="ranger_e")*: Name of the optimizer.

**Behavior & Notes**:

* Gradient Centralization (GC): when `use_gc=True` the optimizer will subtract the mean over feature axes for gradients whose rank exceeds a threshold; set `gc_conv_only=True` to restrict GC to convolution-like gradients.
* Subset-Normalization (SN): when `sn=True` a grouped second-moment (`exp_avg_sq`) is stored per-subset instead of per-element. The optimizer chooses a practical subset size (heuristic: `sqrt(size)` unless `subset_size` is provided) and computes grouped squared norms to reduce memory.
* Lookahead: the optimizer keeps a `slow_buffer` (slow weights) and every `k` steps performs slow/fast interpolation `slow += alpha * (fast - slow)` and copies slow weights back to the fast parameters.
* RAdam rectification: the update uses the rectified `N_sma` schedule; when `N_sma <= N_sma_threshhold` the optimizer uses unrectified updates similar to SGD with momentum.
* D-Adapt (optional): when `DAdapt=True` the optimizer collects statistics across parameters to estimate and adapt an internal scalar `d0` that multiplies `learning_rate`. This is designed to produce a stable automatic step-size without manual tuning.
* Mixed precision: the implementation casts gradients and internal accumulators to `float32` for stability; parameters are updated and then cast back to their original dtype.
* Sparse gradients are **not supported**; the optimizer raises if a sparse gradient is encountered.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.ranger_e import Ranger_e

optimizer = Ranger_e(
    learning_rate=1e-3,
    beta1=0.95,
    beta2=0.999,
    epsilon=1e-5,
    alpha=0.5,         # lookahead interpolation
    k=6,               # lookahead sync period
    use_gc=True,       # gradient centralization
    gc_conv_only=False,
    sn=True,           # subset-normalization
    subset_size=-1,    # automatic subset sizing
    DAdapt=True,       # enable D-Adapt automatic scaling
    d0=1e-6,
    growth_rate=1e6
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Ranger2020_e

**Overview**:

`Ranger2020_e` is a hybrid optimizer combining RAdam-style rectified adaptive updates with Lookahead, optional Gradient Centralization (GC) (with local/global modes), optional subset-normalization (SN) to reduce second-moment memory, and an optional D-Adapt automatic step-size controller. It is designed to be robust across architectures: GC recenters gradients for convolutional and/or fully-connected layers, SN groups elements to reduce memory for second moments, Lookahead stabilizes optimization via a slow buffer, and D-Adapt adapts an internal scaling `d0` from observed gradient statistics.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base learning rate (global multiplier). Effective per-step scale may be modulated by D-Adapt when enabled.
* **`beta1`** *(float, default=0.95)*: Decay for the first moment (moving average of gradients).
* **`beta2`** *(float, default=0.999)*: Decay for the second moment (moving average of squared gradients or grouped second moments when SN is enabled).
* **`epsilon`** *(float, default=1e-5)*: Small constant added for numerical stability in denominators.
* **`weight_decay`** *(float, default=0)*: L2 weight decay coefficient (applied inside the update step if non-zero).
* **`alpha`** *(float, default=0.5)*: Lookahead interpolation factor between slow and fast weights (`slow += alpha * (fast - slow)`).
* **`k`** *(int, default=6)*: Lookahead synchronization period (apply slow/fast interpolation every `k` steps).
* **`N_sma_threshhold`** *(int, default=5)*: Threshold controlling rectified (RAdam) vs. unrectified behavior.
* **`use_gc`** *(bool, default=True)*: Enable Gradient Centralization (GC).
* **`gc_conv_only`** *(bool, default=False)*: If `True`, apply GC only to convolution-like gradients (determined by gradient rank).
* **`gc_loc`** *(bool, default=True)*: If `True`, GC is applied locally on the raw gradient; if `False`, GC may be applied to the computed update (alternate placement).
* **`subset_size`** *(int, default=-1)*: Subset size hint for subset-normalization (SN). `-1` uses an automatic heuristic (sqrt of tensor size).
* **`sn`** *(bool, default=False)*: Enable subset-normalization: group elements into subsets and compute grouped second moments to reduce memory.
* **`d0`** *(float, default=1e-6)*: Initial D-Adapt scalar (when `DAdapt=True`) that multiplies the global `learning_rate`.
* **`growth_rate`** *(float, default=inf)*: Maximum multiplicative growth allowed for `d0` per adapt step.
* **`DAdapt`** *(bool, default=False)*: Enable the D-Adapt automatic step-size controller that adjusts `d0` from accumulated statistics.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional)*: Standard gradient clipping options forwarded to the base optimizer.
* **`use_ema`** *(bool, default=False)*: Whether to maintain exponential moving averages of parameters.
* **`ema_momentum`** *(float, default=0.99)*: EMA momentum if `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting model weights from EMA (when enabled).
* **`loss_scale_factor`** *(float, optional)*: Loss scaling factor for mixed-precision workflows.
* **`gradient_accumulation_steps`** *(int, optional)*: Steps to accumulate gradients before applying an optimizer update.
* **`name`** *(str, default="ranger2020_e")*: Name of the optimizer.

**Behavior & Notes**:

* Gradient Centralization (GC): when `use_gc=True` the optimizer will subtract the mean across feature axes for gradients (either locally on raw gradients if `gc_loc=True`, or on computed updates otherwise). Use `gc_conv_only=True` to restrict GC to convolution-like gradients.
* Subset-Normalization (SN): when `sn=True` grouped second-moment statistics are stored per-subset rather than per-element. The optimizer chooses a practical subset size (heuristic: `sqrt(size)` unless `subset_size` is provided) and computes grouped squared norms to reduce memory footprint.
* Lookahead: the optimizer maintains `slow_buffer` (slow weights) and every `k` steps performs `slow += alpha * (fast - slow)` and copies slow weights back to the fast parameters.
* RAdam rectification: updates are rectified using the `N_sma` schedule; when `N_sma <= N_sma_threshhold` the optimizer falls back to an unrectified style update.
* D-Adapt: when `DAdapt=True` the optimizer gathers per-step statistics to adapt an internal scalar `d0` that rescales the effective step size automatically.
* Mixed precision: internal accumulators are maintained in float32; gradients and parameters are cast as needed for numerical stability.
* Sparse gradients are **not supported**; the optimizer will raise if a sparse gradient is passed.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.ranger2020_e import Ranger_e

optimizer = Ranger_e(
    learning_rate=1e-3,
    beta1=0.95,
    beta2=0.999,
    epsilon=1e-5,
    alpha=0.5,
    k=6,
    use_gc=True,
    gc_conv_only=False,
    gc_loc=True,
    sn=True,
    subset_size=-1,
    DAdapt=True,
    d0=1e-6,
    growth_rate=1e6
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Ranger21_e

**Overview**:

`Ranger21_e` is a composite optimizer that integrates many modern optimization techniques into a single, flexible optimizer. It combines AdamW-style adaptive updates (with optional Adam-debias), adaptive gradient clipping (AGC), gradient centralization, gradient normalization, positive–negative momentum (PN-momentum), subset-based second-moment estimation (subset normalization) to reduce memory, Lookahead, softplus transformation of denominators, stable weight decay, norm-based parameter regularization, linear warm-up / warm-down schedules, and an optional D-Adapt automatic step-size controller. It is intended for advanced use where many stability and performance improvements are desired together.

**Parameters**:

* **`num_iterations`** *(int, required)*: Total number of optimizer iterations (used to build warm-up / warm-down scheduling).
* **`learning_rate`** *(float, default=1e-3)*: Base learning rate (global multiplier).
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in denominators.
* **`weight_decay`** *(float, default=1e-4)*: L2 weight decay coefficient (stable / decoupled application is supported).
* **`beta0`** *(float, default=0.9)*: Additional beta parameter (reserved / used internally; e.g., for noise normalization).
* **`betas`** *(tuple, default=(0.9, 0.999))*: (beta1, beta2) pair for first- and second-moment moving averages.
* **`use_softplus`** *(bool, default=True)*: If `True`, applies a softplus transform to the denominator (improves stability).
* **`beta_softplus`** *(float, default=50.0)*: Softplus `beta` parameter when `use_softplus=True`.
* **`disable_lr_scheduler`** *(bool, default=False)*: Disable warm-up / warm-down LR scheduling when `True`.
* **`num_warm_up_iterations`** *(int, optional)*: Number of warm-up iterations (auto-computed if `None`).
* **`num_warm_down_iterations`** *(int, optional)*: Number of warm-down iterations (auto-computed if `None`).
* **`warm_down_min_lr`** *(float, default=3e-5)*: Minimum LR at the end of warm-down.
* **`agc_clipping_value`** *(float, default=1e-2)*: Clipping value used by Adaptive Gradient Clipping.
* **`agc_eps`** *(float, default=1e-3)*: Small epsilon for AGC unit-wise norm robustness.
* **`centralize_gradients`** *(bool, default=True)*: Enable gradient centralization.
* **`normalize_gradients`** *(bool, default=True)*: Enable gradient normalization (per-tensor standardization where applied).
* **`lookahead_merge_time`** *(int, default=5)*: Lookahead synchronization period (every `k` steps).
* **`lookahead_blending_alpha`** *(float, default=0.5)*: Lookahead interpolation factor (`slow += alpha * (fast - slow)`).
* **`weight_decouple`** *(bool, default=True)*: Use decoupled weight decay (AdamW style) when `True`.
* **`fixed_decay`** *(bool, default=False)*: If `True` apply fixed decay instead of scaling by LR.
* **`norm_loss_factor`** *(float, default=1e-4)*: Factor controlling the "norm loss" regularization applied to parameters.
* **`adam_debias`** *(bool, default=False)*: If `True`, apply Adam-style debiasing to the step-size.
* **`subset_size`** *(int, default=-1)*: Subset size hint for subset-normalization (SN). `-1` uses automatic heuristic (sqrt of tensor size).
* **`sn`** *(bool, default=False)*: Enable subset-based second-moment estimation (reduces memory by grouping elements).
* **`d0`** *(float, default=1e-6)*: Initial D-Adapt scalar when `DAdapt=True` (internal step-size scaler).
* **`growth_rate`** *(float, default=inf)*: Maximum allowed multiplicative growth for the D-Adapt scalar between updates.
* **`DAdapt`** *(bool, default=False)*: Enable D-Adapt automatic step-size controller.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional)*: Standard gradient clipping parameters forwarded to base optimizer.
* **`use_ema`** *(bool, default=False)*: Maintain exponential moving averages of parameters when enabled.
* **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA if used.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting model weights from EMA snapshot.
* **`loss_scale_factor`** *(float, optional)*: Loss scaling factor for mixed precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before applying updates.
* **`name`** *(str, default="ranger21_e")*: Optimizer name used by Keras.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.ranger2020_e import Ranger_e

optimizer = Ranger21_e(
    num_iterations=total_iterations,
    learning_rate=1e-3,
    betas=(0.9, 0.999),
    epsilon=1e-8,
    weight_decay=1e-4,
    use_softplus=True,
    beta_softplus=50.0,
    agc_clipping_value=1e-2,
    agc_eps=1e-3,
    centralize_gradients=True,
    normalize_gradients=True,
    lookahead_merge_time=5,
    lookahead_blending_alpha=0.5,
    sn=True,
    subset_size=-1,
    DAdapt=True,
    d0=1e-6,
    growth_rate=1e6
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Ranger25

**Overview**:

`Ranger25` is a feature-rich, experimental optimizer that mixes many modern optimization techniques into a single algorithm. It combines orthogonalized gradients (OrthoGrad), adaptive gradient clipping (AGC), stable AdamW-style updates (or atan2-based alternative), positive/negative and slow/fast momentum mixtures, Lookahead, subset-based second-moment estimation (subset normalization, SN) for memory efficiency, cautious masking, and optional D-Adapt automatic step-size adaptation. It is intended for advanced users who want to experiment with many stability and performance components together.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base learning rate (global multiplier).
* **`betas`** *(tuple of three floats, default=(0.9, 0.98, 0.9999))*: Momentum coefficients. Interpreted as `(beta1, beta2, beta3)` where `beta1`/`beta2` are used for first/second moment behavior and `beta3` is used for the slow (lookahead / slow-moving) momentum blending.
* **`epsilon`** *(float, default=1e-8)*: Small number added to denominators for numerical stability.
* **`weight_decay`** *(float, default=1e-3)*: L2 weight decay coefficient; applied either decoupled (AdamW-style) or in-place depending on other flags.
* **`alpha`** *(float, default=5.0)*: Lookahead / slow-momentum blending coefficient (or positive/negative momentum scale depending on internals).
* **`t_alpha_beta3`** *(int or None, default=None)*: Time horizon used to schedule `alpha` and `beta3`. When `None` scheduling is disabled and fixed `alpha`/`beta3` are used.
* **`lookahead_merge_time`** *(int, default=5)*: Number of steps between Lookahead slow/fast parameter merges.
* **`lookahead_blending_alpha`** *(float, default=0.5)*: Lookahead interpolation factor when merging slow and fast weights.
* **`cautious`** *(bool, default=False)*: Enable cautious masking that reduces updates that disagree in sign with gradients (helps stability).
* **`stable_adamw`** *(bool, default=True)*: Use the stable / normalized AdamW step-size scaling (instead of a plain AdamW update). When `False`, an alternate update (e.g. atan2-style) may be used where implemented.
* **`orthograd`** *(bool, default=True)*: Apply OrthoGrad (project gradient to be orthogonal to parameter direction) to reduce harmful alignment between weights and their gradients.
* **`weight_decouple`** *(bool, default=True)*: Use decoupled weight decay (AdamW-style) if `True`; otherwise apply standard L2-style decay.
* **`fixed_decay`** *(bool, default=False)*: When `True`, apply fixed (non-LR-scaled) decay; otherwise decay may be scaled by LR.
* **`subset_size`** *(int, default=-1)*: Hint for subset size used by subset-based second-moment estimation (SN). `-1` uses the automatic heuristic (sqrt of tensor size).
* **`sn`** *(bool, default=False)*: Enable subset-based second-moment estimation (Subset Normalization) to reduce memory for second-moment buffers.
* **`d0`** *(float, default=1e-6)*: Initial `d0` scalar for D-Adapt (initial adaptive multiplier).
* **`growth_rate`** *(float, default=inf)*: Maximum allowed multiplicative growth per D-Adapt update for the internal scale.
* **`DAdapt`** *(bool, default=False)*: Enable D-Adapt automatic step-size controller (adapts an internal scalar `d0_`).
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional)*: Standard gradient clipping parameters forwarded to the base optimizer.
* **`use_ema`** *(bool, default=False)*: Maintain exponential moving averages of parameters when enabled.
* **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA if `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency to overwrite model weights from EMA snapshot (if used).
* **`loss_scale_factor`** *(float, optional)*: Loss scaling factor for mixed precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Steps to accumulate gradients before applying an update.
* **`name`** *(str, default="ranger25")*: Optimizer name used by Keras.

**Notes & caveats**:

* Sparse gradients are **not** supported; the optimizer will raise an error if it encounters them.
* Many per-parameter statistics are stored in float32 for numerical stability. Mixed-precision training should be tested carefully and may require `loss_scale_factor`.
* When `sn=True`, per-variable subset sizes are selected automatically and stored in `optimizer.subset_size_` after `build()` — inspect this if you need deterministic grouping.
* `DAdapt=True` exposes an internal scalar `optimizer.d0_` which automatically adapts; monitoring it can be helpful to debug step-size behavior.
* `orthograd=True` alters gradients to be orthogonal to parameter vectors — this can help generalization in some cases but may change training dynamics.
* This optimizer integrates many experimental features; enable/disable components selectively to diagnose behavior for your model and dataset.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.ranger25 import Ranger25  # adjust import path as needed

# Create a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Instantiate optimizer
optimizer = Ranger25(
    learning_rate=1e-3,
    betas=(0.9, 0.98, 0.9999),
    epsilon=1e-8,
    weight_decay=1e-3,
    alpha=5.0,
    lookahead_merge_time=5,
    lookahead_blending_alpha=0.5,
    cautious=True,
    stable_adamw=True,
    orthograd=True,
    subset_size=-1,
    sn=True,
    DAdapt=True,
    d0=1e-6,
    growth_rate=1e6
)

# Compile and train
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# SophiaH_e

**Overview**:

The `SophiaH_e` optimizer implements a Sophia-style second-order-aware optimizer with Hutchinson Hessian estimation combined with several practical stabilization and performance features. It supports orthogonal gradient projection (Orthograd), projected-normal-momentum (PNM) first-moment variants, blockwise stochastic second-moment estimation (SN), Adaptive Gradient Clipping (AGC), cautious update masking, D-Adapt automatic global step-size adaptation, optional trust-ratio layer-wise scaling, lookahead slow/fast weight blending, and optional EMA integration. It is designed for stable, large-step training with approximate curvature information from Hutchinson probes.

**Parameters**:

* **`learning_rate`** *(float, default=6e-2)*: Base step size for parameter updates.
* **`beta1`** *(float, default=0.96)*: Exponential decay rate for the first moment (momentum) estimates.
* **`beta2`** *(float, default=0.99)*: Exponential decay rate for second moment / Hessian moment estimates.
* **`epsilon`** *(float, default=1e-12)*: Small constant for numerical stability when dividing by second-moment / Hessian terms.
* **`weight_decay`** *(float, default=0.0)*: Weight decay coefficient applied either decoupled (when `weight_decouple=True`) or added to gradients.
* **`weight_decouple`** *(bool, default=True)*: Use decoupled weight decay (AdamW-style) instead of adding parameter values to gradients.
* **`fixed_decay`** *(bool, default=False)*: When True use fixed decay factor (not scaled by learning rate).
* **`p`** *(float, default=1e-2)*: Sophia/Hutchinson damping / probe scaling hyperparameter (used by Hutchinson estimator logic).
* **`update_period`** *(int, default=10)*: Frequency (in steps) to update Sophia / Hessian moments from Hutchinson probes.
* **`num_samples`** *(int, default=1)*: Number of Hutchinson probe samples per Hessian update.
* **`hessian_distribution`** *(str, default='gaussian')*: Distribution of Hutchinson probes; either `'gaussian'` or `'rademacher'`.
* **`orthograd`** *(bool, default=True)*: Apply orthogonal gradient projection to remove components aligned with the parameter vector.
* **`lookahead_merge_time`** *(int, default=5)*: Number of steps between lookahead merges (slow/fast weight blending).
* **`lookahead_blending_alpha`** *(float, default=0.5)*: Blending coefficient for lookahead slow/fast weights.
* **`lookahead`** *(bool, default=False)*: Enable lookahead slow/fast parameter averaging.
* **`pnm`** *(bool, default=False)*: Use projected-normal-momentum (PNM) variant for first moment storage and updates.
* **`subset_size`** *(int, default=-1)*: Block size for stochastic/blockwise normalization (SN). If `-1` a heuristic is used.
* **`sn`** *(bool, default=False)*: Enable stochastic (blockwise) second moment / Hessian estimation to reduce memory.
* **`agc`** *(bool, default=False)*: Apply Adaptive Gradient Clipping (AGC) to gradients before updates.
* **`cautious`** *(bool, default=False)*: Use cautious masking that downscales updates whose sign disagrees with the raw gradient.
* **`d0`** *(float, default=1e-6)*: Initial base D-Adapt scalar used to compute adaptive global step-length.
* **`growth_rate`** *(float, default=float('inf'))*: Maximum allowed multiplicative growth for the D-Adapt scalar.
* **`DAdapt`** *(bool, default=False)*: Enable D-Adapt automatic global step-length adaptation logic.
* **`trust_ratio`** *(bool, default=False)*: Apply layer-wise trust-ratio scaling (parameter norm / update norm) to updates.
* **`trust_clip`** *(bool, default=False)*: If enabled clip the trust ratio to at most 1.0.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional)*: Gradient clipping options compatible with Keras optimizers.
* **`use_ema`** *(bool, default=False)*: Maintain an exponential moving average (EMA) of model weights.
* **`ema_momentum`** *(float, default=0.99)*: EMA momentum factor when `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights (if used).
* **`loss_scale_factor`** *(float, optional)*: Optional loss scaling factor for mixed precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
* **`name`** *(str, default="sophiah\_e")*: Name of the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.sophia import SophiaH_e

# Define model and loss
model = tf.keras.Sequential([...])
loss_fn = tf.keras.losses.MeanSquaredError()

# Instantiate optimizer
optimizer = SophiaH_e(
    learning_rate=6e-2,
    beta1=0.96,
    beta2=0.99,
    epsilon=1e-12,
    weight_decay=1e-3,
    weight_decouple=True,
    fixed_decay=False,
    p=1e-2,
    update_period=10,
    num_samples=2,
    hessian_distribution="gaussian",
    orthograd=True,
    lookahead=True,
    lookahead_merge_time=5,
    lookahead_blending_alpha=0.5,
    pnm=True,
    subset_size=256,
    sn=True,
    agc=True,
    cautious=True,
    DAdapt=True,
    d0=1e-6,
    trust_ratio=True,
    trust_clip=True
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

# SophiaG

**Overview**:

The `SophiaG` optimizer is a variant of second-moment / Hessian-aware optimizers that combines first-moment momentum with a per-parameter curvature estimate (here stored in `hessian`) to produce curvature-normalized updates. It supports decoupled weight decay, optional fixed decay scaling, and an option to maximize the objective (gradient ascent). It is a compact, efficient optimizer intended for scenarios where light curvature information improves stability and step sizing.

**Parameters**:

* **`learning_rate`** *(float, default=1e-4)*: The base step size for parameter updates.
* **`beta1`** *(float, default=0.965)*: Exponential decay rate for the first moment (momentum) estimates.
* **`beta2`** *(float, default=0.99)*: Exponential decay rate for the second moment / hessian moving average.
* **`rho`** *(float, default=0.04)*: Small scaling factor used to stabilize curvature normalization (appears in denominator `rho * hessian + eps`).
* **`weight_decay`** *(float, default=1e-1)*: Coefficient for weight decay. If `weight_decouple` is enabled, decay is applied as decoupled step (AdamW-style).
* **`weight_decouple`** *(bool, default=True)*: Whether to apply decoupled weight decay (preferred for adaptive optimizers).
* **`fixed_decay`** *(bool, default=False)*: If True, apply decay with a fixed factor instead of scaling by the learning rate.
* **`maximize`** *(bool, default=False)*: If True, the optimizer will maximize the objective (i.e., apply gradient ascent).
* **`clipnorm`** *(float, optional)*: Clip gradients by norm before applying updates.
* **`clipvalue`** *(float, optional)*: Clip gradients by value before applying updates.
* **`global_clipnorm`** *(float, optional)*: Clip gradients by a global norm across parameters.
* **`use_ema`** *(bool, default=False)*: Whether to maintain an exponential moving average (EMA) of model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum (decay) used by the EMA when `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency (steps) to overwrite model weights with EMA weights if desired.
* **`loss_scale_factor`** *(float, optional)*: Factor to scale the loss (useful with mixed precision).
* **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before applying an update.
* **`name`** *(str, default="sophiag")*: Name of the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.sophia import SophiaG

# Define model and loss
model = tf.keras.Sequential([...])
loss_fn = tf.keras.losses.MeanSquaredError()

# Instantiate optimizer
optimizer = SophiaG(
    learning_rate=1e-4,
    beta1=0.965,
    beta2=0.99,
    rho=0.04,
    weight_decay=1e-1,
    weight_decouple=True,
    fixed_decay=False
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

# SophiaG_e

**Overview**:

The `SophiaG_e` optimizer is a curvature-aware optimizer in the Sophia family that combines first-moment momentum variants with blockwise second-moment (Hessian) estimation and practical stabilization features. It supports orthogonal gradient projection (Orthograd), projected-normal-momentum (PNM), stochastic/blockwise second-moment estimation (SN), Adaptive Gradient Clipping (AGC), cautious update masking, optional D-Adapt automatic global step-size adaptation, optional layer-wise trust-ratio scaling, and lookahead slow/fast weight blending. It is intended for stable large-step training while leveraging approximate curvature (Hessian) information.

**Parameters**:

* **`learning_rate`** *(float, default=1e-4)*: The base step size for parameter updates.
* **`beta1`** *(float, default=0.965)*: Exponential decay rate for the first moment (momentum) estimates.
* **`beta2`** *(float, default=0.99)*: Exponential decay rate for second moment / Hessian tracking.
* **`rho`** *(float, default=0.04)*: Sophia/G-specific curvature scaling factor used with Hessian terms.
* **`weight_decay`** *(float, default=1e-1)*: Coefficient for weight decay. Applied decoupled (AdamW-style) when `weight_decouple=True` or added to gradients otherwise.
* **`weight_decouple`** *(bool, default=True)*: Use decoupled weight decay instead of adding parameter values to gradients.
* **`fixed_decay`** *(bool, default=False)*: When True, use a fixed decay factor (not scaled by the learning rate).
* **`maximize`** *(bool, default=False)*: If True, performs gradient ascent (useful for maximization objectives).
* **`orthograd`** *(bool, default=True)*: Apply orthogonal gradient projection to remove components aligned with the parameter vector.
* **`lookahead_merge_time`** *(int, default=5)*: Number of steps between lookahead slow/fast parameter merges.
* **`lookahead_blending_alpha`** *(float, default=0.5)*: Blending coefficient for lookahead slow/fast weights.
* **`lookahead`** *(bool, default=False)*: Enable lookahead slow/fast parameter averaging.
* **`pnm`** *(bool, default=False)*: Use projected-normal-momentum (PNM) variant for first-moment storage and updates (an alternative to a single momentum buffer).
* **`subset_size`** *(int, default=-1)*: Block size for stochastic/blockwise normalization (SN). If `-1` a heuristic is used to pick block size.
* **`sn`** *(bool, default=False)*: Enable stochastic (blockwise) second-moment / Hessian estimation to reduce memory and compute.
* **`agc`** *(bool, default=False)*: Apply Adaptive Gradient Clipping (AGC) to gradients before updates.
* **`cautious`** *(bool, default=False)*: Use cautious masking that downscales updates whose sign disagrees with the raw gradient.
* **`d0`** *(float, default=1e-6)*: Initial base D-Adapt scalar used as a starting point for adaptive global step-length (when `DAdapt=True`).
* **`growth_rate`** *(float, default=float('inf'))*: Maximum allowed multiplicative growth for the D-Adapt scalar.
* **`DAdapt`** *(bool, default=False)*: Enable D-Adapt automatic global step-length adaptation logic (tracks a global scale d0).
* **`trust_ratio`** *(bool, default=False)*: Apply layer-wise trust-ratio scaling (parameter norm / update norm) to the update (similar conceptually to LARS/LAMB).
* **`trust_clip`** *(bool, default=False)*: When True, clip the computed trust ratio to at most 1.0.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional)*: Gradient clipping options interoperable with Keras optimizers.
* **`use_ema`** *(bool, default=False)*: Maintain an exponential moving average (EMA) of model weights if enabled.
* **`ema_momentum`** *(float, default=0.99)*: EMA momentum factor when `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights (if used).
* **`loss_scale_factor`** *(float, optional)*: Optional loss scaling for mixed precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
* **`name`** *(str, default="sophiag\_e")*: Name of the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.sophia import SophiaG_e

# Define model and loss
model = tf.keras.Sequential([...])
loss_fn = tf.keras.losses.MeanSquaredError()

# Instantiate the enhanced optimizer
optimizer = SophiaG_e(
    learning_rate=1e-4,
    beta1=0.965,
    beta2=0.99,
    rho=0.04,
    weight_decay=1e-1,
    weight_decouple=True,
    fixed_decay=False,
    maximize=False,
    orthograd=True,
    lookahead=True,
    lookahead_merge_time=5,
    lookahead_blending_alpha=0.5,
    pnm=True,
    subset_size=256,
    sn=True,
    agc=True,
    cautious=True,
    DAdapt=True,
    d0=1e-6,
    trust_ratio=True,
    trust_clip=True,
    name="sophiag_e"
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

# SOAP_e

**Overview**:

The `SOAP_e` optimizer is a hybrid curvature-aware optimizer that combines Shampoo-style preconditioning, Sophia-like blockwise normalization, and practical stabilizers. It maintains blockwise preconditioners (Shampoo/GG and their eigen-bases), supports projection of gradients into preconditioned spaces, stochastic/blockwise second-moment estimation (SN), projected-normal-momentum (PNM), Adaptive Gradient Clipping (AGC), cautious masking, optional AEM-style slow averages, D-Adapt automatic global step-size adaptation, optional layer-wise trust-ratio scaling, and an optional lookahead slow/fast weight blending. `SOAP_e` is intended for stable large-step training while leveraging approximate curvature and block preconditioning to improve convergence on large models.

**Parameters**:

* **`learning_rate`** *(float, default=3e-3)*: Base step size for parameter updates.
* **`beta1`** *(float, default=0.95)*: Exponential decay rate for the first-moment (momentum) estimates.
* **`beta2`** *(float, default=0.95)*: Exponential decay rate used for second-moment / Shampoo accumulation.
* **`beta3`** *(float, default=0.9999)*: (AEM) decay for slow exponential averages when `aem=True`.
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in denominators.
* **`weight_decay`** *(float, default=1e-2)*: Weight decay coefficient. Applied decoupled (multiplicative) when used as shown in the implementation.
* **`shampoo_beta`** *(float or None, default=None)*: Beta for Shampoo preconditioner updates; if `None` falls back to `beta2`.
* **`precondition_frequency`** *(int, default=10)*: How often (in steps) to recompute eigen-bases / orthogonal matrices for preconditioning.
* **`max_precondition_dim`** *(int, default=10000)*: Maximum dimension for which full preconditioner matrices are maintained. Larger dims are skipped or simplified.
* **`merge_dims`** *(bool, default=False)*: Merge small dimensions to keep preconditioner matrix sizes manageable.
* **`precondition_1d`** *(bool, default=False)*: Enable 1D preconditioning for vector parameters when appropriate.
* **`correct_bias`** *(bool, default=True)*: Apply bias-correction factors to step sizes (similar to Adam/AdamW bias correction).
* **`normalize_gradient`** *(bool, default=False)*: Optionally normalize the final projected update before applying it.
* **`data_format`** *(str, default='channels\_last')*: Tensor layout used when merging / projecting convolutional parameter axes (`'channels_last'` or `'channels_first'`).
* **`lookahead_merge_time`** *(int, default=5)*: Number of steps between lookahead slow/fast parameter merges.
* **`lookahead_blending_alpha`** *(float, default=0.5)*: Blending coefficient for lookahead slow/fast weights.
* **`lookahead`** *(bool, default=False)*: Enable lookahead slow/fast parameter averaging.
* **`pnm`** *(bool, default=False)*: Use projected-normal-momentum (PNM) variant for first-moment representation (pos/neg momentum) instead of a single buffer.
* **`subset_size`** *(int, default=-1)*: Block size hint for stochastic/blockwise normalization. If negative, a heuristic is used.
* **`sn`** *(bool, default=False)*: Enable stochastic (blockwise) second-moment / block estimates to reduce memory and compute.
* **`agc`** *(bool, default=False)*: Apply Adaptive Gradient Clipping (AGC) to gradients before updates.
* **`cautious`** *(bool, default=False)*: Use cautious masking that reduces updates whose sign disagrees with the raw gradient.
* **`aem`** *(bool, default=True)*: Enable AEM-style extra slow exponential averages which can be added to the main first moment.
* **`alpha`** *(float, default=5.0)*: Blending coefficient used by the AEM slow-average when `aem=True`.
* **`t_alpha_beta3`** *(int or None, default=None)*: Steps over which to schedule `alpha`/`beta3` (when provided).
* **`d0`** *(float, default=1e-6)*: Initial D-Adapt scalar (base global step-size factor) used when `DAdapt=True`.
* **`growth_rate`** *(float, default=float('inf'))*: Maximum allowed multiplicative growth of the D-Adapt scalar between updates.
* **`DAdapt`** *(bool, default=False)*: Enable D-Adapt automatic global step-size adaptation (tracks a global `d0_` and updates it).
* **`trust_ratio`** *(bool, default=False)*: Apply layer-wise trust-ratio scaling (parameter norm / update norm) similar to LARS/LAMB.
* **`trust_clip`** *(bool, default=False)*: If True, clip the trust ratio to at most 1.0.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional)*: Standard gradient clipping options forwarded to the base `tf.keras` optimizer machinery.
* **`use_ema`** *(bool, default=False)*: Maintain an exponential moving average (EMA) of model weights if enabled.
* **`ema_momentum`** *(float, default=0.99)*: EMA momentum factor when `use_ema=True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights (if used).
* **`loss_scale_factor`** *(float, optional)*: Optional loss scaling factor for mixed precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before applying an update.
* **`name`** *(str, default="soap\_e")*: Name of the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.soap import SOAP_e

# Instantiate optimizer
optimizer = SOAP_e(
    learning_rate=3e-3,
    beta1=0.95,
    beta2=0.95,
    beta3=0.9999,
    epsilon=1e-8,
    weight_decay=1e-2,
    shampoo_beta=None,
    precondition_frequency=10,
    max_precondition_dim=10000,
    merge_dims=False,
    precondition_1d=False,
    correct_bias=True,
    normalize_gradient=False,
    data_format='channels_last',
    lookahead=True,
    lookahead_merge_time=5,
    lookahead_blending_alpha=0.5,
    pnm=True,
    subset_size=256,
    sn=True,
    agc=True,
    cautious=True,
    aem=True,
    alpha=5.0,
    DAdapt=True,
    d0=1e-6,
    trust_ratio=False,
    trust_clip=False,
    name="soap_e"
)

# Compile a model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(32, 32, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```

# Muon\_e

**Overview**:

The `Muon_e` optimizer is a hybrid, highly configurable optimizer that blends large-step momentum-based updates for matrix-like parameters ("Muon" mode) with AdamW / Sophia-style second-order-aware updates for small or non-matrix parameters. It supports projected-normal-momentum (PNM), stochastic blockwise second-moment estimates (SN), Adaptive Gradient Clipping (AGC), cautious masking, optional AEM slow averages, Hutchinson Hessian estimation for Sophia-style curvature, D-Adapt automatic global step-size adaptation, optional layer-wise trust-ratio scaling, and a lookahead slow/fast weight blending. Use `use_muon=True` to enable the Muon-style flattened-momentum pipeline; set it to `False` to use the AdamW/Sophia path.

**Parameters**:

* **`learning_rate`** *(float, default=2e-2)*: Base learning rate for Muon-style updates (or effective learning-rate scaling for the optimizer).
* **`beta1`** *(float, default=0.9)*: Exponential decay rate for first-moment / momentum estimates.
* **`beta2`** *(float, default=0.95)*: Exponential decay rate for second-moment / variance or related smoothing.
* **`beta3`** *(float, default=0.9999)*: (AEM) decay for the slow exponential average when `aem=True`.
* **`weight_decay`** *(float, default=1e-2)*: Weight decay coefficient; applied as decoupled multiplicative decay when enabled.
* **`momentum`** *(float, default=0.95)*: Momentum factor used by the Muon momentum buffer path.
* **`weight_decouple`** *(bool, default=True)*: When True, apply decoupled (AdamW-style) weight decay; otherwise apply standard L2-style decay to gradients.
* **`nesterov`** *(bool, default=True)*: Use Nesterov-style correction (lookahead on the momentum buffer) in Muon updates.
* **`ns_steps`** *(int, default=5)*: Number of Newton–Schulz iterations used by the "zero-power" normalization routine applied to momentum updates.
* **`use_adjusted_lr`** *(bool, default=False)*: If True, adjust learning rate per-parameter by shape-based ratio; otherwise use a simpler heuristic.
* **`adamw_lr`** *(float, default=3e-4)*: Learning rate used by the AdamW / Sophia path (when `use_muon=False`).
* **`adamw_wd`** *(float, default=0.0)*: Weight-decay coefficient applied in the AdamW path (separate from `weight_decay` for Muon branch).
* **`adamw_eps`** *(float, default=1e-8)*: Small epsilon used in denominators for AdamW/Sophia-style updates.
* **`use_muon`** *(bool, default=True)*: Enable Muon flattened-momentum pipeline for (typically) matrix-like parameters. When False, optimizer uses the AdamW / Sophia-style branch.
* **`subset_size`** *(int, default=-1)*: Block size hint for stochastic/blockwise normalization (SN). Negative means heuristic selection.
* **`sn`** *(bool, default=False)*: Enable stochastic (blockwise) second-moment estimation to reduce memory & compute for large tensors.
* **`lookahead_merge_time`** *(int, default=5)*: Steps between lookahead slow/fast parameter merges.
* **`lookahead_blending_alpha`** *(float, default=0.5)*: Blending factor for lookahead slow/fast weights.
* **`lookahead`** *(bool, default=False)*: Enable lookahead slow/fast parameter averaging.
* **`pnm`** *(bool, default=False)*: Use projected-normal-momentum (PNM) representation (pos/neg momentum) instead of a single first-moment buffer.
* **`agc`** *(bool, default=False)*: Apply Adaptive Gradient Clipping to gradients before computing updates.
* **`cautious`** *(bool, default=False)*: Use cautious masking that downscales updates whose sign disagrees with the raw gradient.
* **`aem`** *(bool, default=False)*: Enable AEM-style additional slow exponential averages (can be mixed into first-moment).
* **`alpha`** *(float, default=5.0)*: Blending coefficient for the AEM slow-average when `aem=True`.
* **`t_alpha_beta3`** *(int or None, default=None)*: Number of steps over which to schedule `alpha`/`beta3`; if `None` no schedule is used.
* **`sophia`** *(bool, default=False)*: Enable Sophia-style Hessian-moment and curvature-aware updates on the AdamW path (when `use_muon=False`).
* **`p`** *(float, default=1e-2)*: Sophia-style clipping / scaling parameter (algorithm-specific).
* **`update_period`** *(int, default=10)*: Frequency (in steps) to update Sophia / Hessian moments or Hutchinson estimates.
* **`num_samples`** *(int, default=1)*: Number of Hutchinson samples used to estimate Hessian-vector products when `sophia=True`.
* **`hessian_distribution`** *(str, default='gaussian')*: Distribution for Hutchinson estimation (`'gaussian'` or `'rademacher'`).
* **`d0`** *(float, default=1e-6)*: Initial D-Adapt scalar used by automatic step-size adaptation when `DAdapt=True`.
* **`growth_rate`** *(float, default=float('inf'))*: Maximum allowed growth factor for the D-Adapt scalar between updates.
* **`DAdapt`** *(bool, default=False)*: Enable D-Adapt automatic global step-size adaptation (maintains and updates a scalar `d0_`).
* **`trust_ratio`** *(bool, default=False)*: Apply layer-wise trust-ratio scaling (parameter norm / update norm) similar to LARS/LAMB.
* **`trust_clip`** *(bool, default=False)*: If True, clip computed trust ratio to a maximum of 1.0.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional)*: Standard gradient clipping options forwarded to the base `tf.keras` optimizer machinery.
* **`use_ema`** *(bool, default=False)*: Maintain exponential moving average (EMA) of model weights if enabled.
* **`ema_momentum`** *(float, default=0.99)*: EMA momentum factor (used when `use_ema=True`).
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights (if specified).
* **`loss_scale_factor`** *(float, optional)*: Optional loss scaling factor for mixed-precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before applying an update.
* **`name`** *(str, default="muon\_e")*: Name of the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.muon_e import Muon_e

# Instantiate optimizer
opt = Muon_e(
    learning_rate=2e-2,
    beta1=0.9,
    beta2=0.95,
    momentum=0.95,
    weight_decouple=True,
    nesterov=True,
    ns_steps=5,
    use_adjusted_lr=False,
    adamw_lr=3e-4,
    adamw_wd=1e-3,
    adamw_eps=1e-8,
    use_muon=True,
    subset_size=256,
    sn=True,
    lookahead=True,
    lookahead_merge_time=5,
    lookahead_blending_alpha=0.5,
    pnm=True,
    agc=True,
    cautious=True,
    aem=False,
    sophia=True,
    update_period=10,
    num_samples=1,
    hessian_distribution="gaussian",
    DAdapt=True,
    d0=1e-6,
    trust_ratio=False,
    trust_clip=False,
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

# DistributedMuon\_e

**Overview**:

The `DistributedMuon_e` optimizer is a distributed-aware, hybrid optimizer that extends the Muon/E-inspired family to multi-replica (tf.distribute) training. It blends flattened momentum updates for large / matrix-like parameters with AdamW / Sophia-style curvature-aware updates for smaller parameters, while providing distributed synchronization (all-gather / per-replica sharding), optional projected-normal-momentum (PNM), stochastic blockwise second-moment estimates (SN), Adaptive Gradient Clipping (AGC), Hutchinson Hessian estimation (Sophia), D-Adapt automatic global step-size adaptation, layer-wise trust-ratio scaling, lookahead slow/fast blending, and several practical controls for numerical stability. Use `use_muon=True` to enable the Muon flattened-momentum path for matrix-like parameters; set it to `False` to run the AdamW/Sophia path across replicas.

**Parameters**:

* **`learning_rate`** *(float, default=2e-2)*: Base learning rate for Muon-style updates (or overall learning-rate scale).
* **`weight_decay`** *(float, default=0.0)*: Weight decay coefficient (L2 / multiplicative depending on `weight_decouple` / branch).
* **`momentum`** *(float, default=0.95)*: Momentum factor for Muon flattened-momentum buffers.
* **`weight_decouple`** *(bool, default=True)*: Use decoupled (AdamW-style) multiplicative weight decay when True; otherwise apply standard L2-style decay onto gradients.
* **`nesterov`** *(bool, default=True)*: Apply Nesterov-style correction to the momentum buffer (in Muon path).
* **`ns_steps`** *(int, default=5)*: Number of Newton–Schulz iterations used by the zero-power normalization routine applied to Muon updates.
* **`use_adjusted_lr`** *(bool, default=False)*: Whether to apply the per-parameter shape-based adjusted LR heuristic.
* **`adamw_lr`** *(float, default=3e-4)*: Learning rate used by the AdamW / Sophia-style branch (when `use_muon=False`).
* **`adamw_betas`** *(tuple(float, float), default=(0.9, 0.95, 0.9999))*: Beta coefficients (beta1, beta2, beta3) used by the AdamW-style updates.
* **`adamw_wd`** *(float, default=0.0)*: AdamW weight decay coefficient used on the AdamW branch.
* **`adamw_eps`** *(float, default=1e-10)*: Small epsilon added to denominators for numerical stability in the AdamW/Sophia path.
* **`use_muon`** *(bool, default=True)*: Enable Muon flattened-momentum pipeline for (typically) matrix-like / large parameters. When False, optimizer runs the AdamW / Sophia branch across replicas.
* **`cautious`** *(bool, default=False)*: When True, apply cautious masking that scales updates whose sign disagrees with raw gradients.
* **`subset_size`** *(int, default=-1)*: Block size hint for stochastic / blockwise normalization (SN). Negative means the optimizer chooses a heuristic block size.
* **`sn`** *(bool, default=False)*: Use stochastic (blockwise) second-moment estimation to save memory & compute on large tensors.
* **`lookahead_merge_time`** *(int, default=5)*: Steps between lookahead slow/fast parameter merges.
* **`lookahead_blending_alpha`** *(float, default=0.5)*: Blending factor used when performing lookahead slow/fast weight updates.
* **`lookahead`** *(bool, default=False)*: Enable lookahead slow/fast parameter averaging.
* **`pnm`** *(bool, default=False)*: Use projected-normal-momentum (pos/neg momentum) representation instead of a single first-moment buffer.
* **`agc`** *(bool, default=False)*: Apply Adaptive Gradient Clipping to gradients prior to updates.
* **`aem`** *(bool, default=False)*: Enable AEM-style additional slow exponential averages (can be mixed into first-moment).
* **`alpha`** *(float, default=5.0)*: Blending coefficient for the AEM slow-average when `aem=True`.
* **`t_alpha_beta3`** *(int or None, default=None)*: Number of steps to schedule `alpha`/`beta3`; if `None` no schedule is used.
* **`sophia`** *(bool, default=False)*: Enable Sophia-style Hessian-moment / curvature-aware updates (Hutchinson estimates) on the AdamW branch.
* **`p`** *(float, default=1e-2)*: Sophia-style clipping/scaling parameter (algorithm specific).
* **`update_period`** *(int, default=10)*: Frequency (in steps) to update Sophia / Hessian moments or perform Hutchinson estimation.
* **`num_samples`** *(int, default=1)*: Number of Hutchinson samples used to estimate Hessian-vector products for Sophia.
* **`hessian_distribution`** *(str, default='gaussian')*: Distribution for Hutchinson estimation: `'gaussian'` or `'rademacher'`.
* **`d0`** *(float, default=1e-6)*: Initial D-Adapt scalar used by automatic step-size adaptation (D-Adapt).
* **`growth_rate`** *(float, default=float('inf'))*: Maximum growth factor allowed for the D-Adapt scalar between updates.
* **`DAdapt`** *(bool, default=False)*: Enable D-Adapt automatic global step-size adaptation (maintains and updates a scalar `d0_`).
* **`trust_ratio`** *(bool, default=False)*: Apply layer-wise trust-ratio scaling (parameter norm / update norm) like LARS/LAMB.
* **`trust_clip`** *(bool, default=False)*: If True, clip computed trust ratio to a maximum of 1.0.
* **`maximize`** *(bool, default=False)*: If True, the optimizer will maximize the objective (useful for e.g. adversarial inner loops).
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional)*: Standard TF gradient clipping options forwarded to base optimizer.
* **`use_ema`** *(bool, default=False)*: Maintain Exponential Moving Average (EMA) of model weights if enabled.
* **`ema_momentum`** *(float, default=0.99)*: EMA momentum factor (when `use_ema=True`).
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights (if specified).
* **`loss_scale_factor`** *(float, optional)*: Optional loss scaling factor for mixed-precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Steps to accumulate gradients before applying an update.
* **`name`** *(str, default="distributedmuon\_e")*: Name of the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.distributedmuon_e import DistributedMuon_e

# Create a distributed strategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    optimizer = DistributedMuon_e(
        learning_rate=2e-2,
        weight_decay=1e-3,
        momentum=0.95,
        use_muon=True,
        ns_steps=5,
        use_adjusted_lr=False,
        adamw_lr=3e-4,
        adamw_betas=(0.9, 0.95),
        adamw_wd=0.0,
        adamw_eps=1e-10,
        subset_size=256,
        sn=True,
        pnm=True,
        agc=True,
        sophia=True,
        update_period=10,
        num_samples=1,
        hessian_distribution='gaussian',
        DAdapt=True,
        d0=1e-6,
        lookahead=True,
        lookahead_merge_time=5,
        lookahead_blending_alpha=0.5,
        name="distributedmuon_e",
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

# Muon

**Overview**:

The `Muon` optimizer implements a matrix-aware "MuOn" style update for dense matrix-like parameters and a standard AdamW-style adaptive update for other tensors. For the Muon branch (when `use_muon=True`) it maintains a momentum buffer per-parameter, optionally applies Nesterov correction, and uses a Newton–Schulz based zero-power normalization to produce stable matrix-aware updates. For the non-Muon branch it applies an AdamW-like moment/variance normalization. `Muon` supports decoupled weight decay, optional per-parameter LR adjustment heuristics for matrix parameters, and simple distributed sharding of Muon updates via `WORLD_SIZE`/`RANK` environment variables.

**Parameters**:

* **`learning_rate`** *(float, default=2e-2)*: Base learning rate used by the Muon branch.
* **`beta1`** *(float, default=0.9)*: Exponential decay rate used for first-moment (momentum) estimation in Adam-like updates and used by some Muon momentum logic.
* **`beta2`** *(float, default=0.95)*: Exponential decay rate used for second-moment (variance) estimation in Adam-like updates.
* **`weight_decay`** *(float, default=1e-2)*: Weight decay coefficient. When `weight_decouple=True` this is applied as decoupled (AdamW-style) weight decay.
* **`momentum`** *(float, default=0.95)*: Momentum coefficient used by the Muon momentum buffer.
* **`weight_decouple`** *(bool, default=True)*: Use decoupled weight decay (AdamW-style) instead of adding decay to the gradients.
* **`nesterov`** *(bool, default=True)*: Apply Nesterov-style correction to the Muon momentum (fast-slow momentum correction).
* **`ns_steps`** *(int, default=5)*: Number of Newton–Schulz iterations executed by the zero-power normalization helper (`zero_power_via_newton_schulz_5`).
* **`use_adjusted_lr`** *(bool, default=False)*: If True apply a per-parameter LR adjustment heuristic for Muon branch (recommended for some matrix shapes).
* **`adamw_lr`** *(float, default=3e-4)*: Learning rate used by the AdamW-style branch (when `use_muon=False`).
* **`adamw_wd`** *(float, default=0.0)*: Weight decay applied to the AdamW branch parameters (subject to `weight_decouple`).
* **`adamw_eps`** *(float, default=1e-8)*: Epsilon for numerical stability in AdamW-style denominators.
* **`use_muon`** *(bool, default=True)*: If True run the Muon matrix-aware branch. If False the optimizer behaves like an AdamW-style optimizer.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional)*: Standard Keras gradient clipping options.
* **`use_ema`** *(bool, default=False)*: Maintain exponential moving average (EMA) of model weights.
* **`ema_momentum`** *(float, default=0.99)*: EMA momentum / decay.
* **`ema_overwrite_frequency`** *(int, optional)*: How often to overwrite model weights with EMA weights.
* **`loss_scale_factor`**, **`gradient_accumulation_steps`** *(optional)*: Support for loss scaling and gradient accumulation.
* **`name`** *(str, default="muon")*: Name of the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.muon import Muon

optimizer = Muon(
    learning_rate=2e-2,
    beta1=0.9,
    beta2=0.95,
    weight_decay=1e-2,
    momentum=0.95,
    weight_decouple=True,
    nesterov=True,
    ns_steps=5,
    use_adjusted_lr=False,
    adamw_lr=3e-4,
    adamw_wd=0.0,
    adamw_eps=1e-8,
    use_muon=True
)

model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AdaMuon

**Overview**:

`AdaMuon` is a hybrid optimizer that blends a Muon-style matrix-aware update with adaptive normalization inspired by Adam. For matrix-like parameters the optimizer computes Muon-style momentum buffers, performs Newton–Schulz zero-power normalization on the momentum, and uses a variance-normalized update built from a running `v` that mirrors Adam's second-moment estimate but flattened per-parameter. For non-Muon parameters `AdaMuon` falls back to an Adam-style update (exp moving average `m` and second moment `v`). `AdaMuon` offers decoupled weight decay, optional per-parameter LR adjustment, and distributed sharding for Muon updates.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base learning rate used by the Muon branch; non-Muon branch uses `adamw_lr`/`adamw_betas` semantics.
* **`beta1`** *(float, default=0.9)*: Decay rate for first-moment estimation (used in Muon momentum updates and Adam branch).
* **`beta2`** *(float, default=0.999)*: Decay rate for second-moment estimation (used in v updates).
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in denominators.
* **`weight_decouple`** *(bool, default=True)*: Apply decoupled weight decay (AdamW-style) if True.
* **`nesterov`** *(bool, default=True)*: Use Nesterov-like correction for Muon momentum updates.
* **`ns_steps`** *(int, default=5)*: Number of Newton–Schulz iterations for zero-power normalization.
* **`use_adjusted_lr`** *(bool, default=False)*: If True use per-parameter LR adjustment heuristic for Muon branch.
* **`adamw_betas`** *(tuple, default=(0.9, 0.999))*: Beta pair used for Adam-like branch when `use_muon=False`.
* **`adamw_lr`** *(float, default=3e-4)*: Learning rate for the Adam-like branch when `use_muon=False`.
* **`adamw_wd`** *(float, default=0.0)*: Weight decay for Adam-like branch.
* **`use_muon`** *(bool, default=True)*: Toggle to enable Muon branch; when False `AdaMuon` behaves like AdamW (with small differences).
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional)*: Keras gradient clipping options.
* **`use_ema`** *(bool, default=False)*: Maintain EMA of weights.
* **`ema_momentum`** *(float, default=0.99)*: EMA momentum.
* **`loss_scale_factor`**, **`gradient_accumulation_steps`** *(optional)*: Loss scaling and gradient accumulation support.
* **`name`** *(str, default="adamuon")*: Name of the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.muon import AdaMuon

optimizer = AdaMuon(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decouple=True,
    nesterov=True,
    ns_steps=5,
    use_adjusted_lr=False,
    adamw_betas=(0.9, 0.999),
    adamw_lr=3e-4,
    adamw_wd=0.0,
    use_muon=True
)

model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AdaMuon\_e

**Overview**:

The `AdaMuon_e` optimizer is a hybrid optimizer that combines Muon-style flattened-momentum updates (for large/matrix-like parameters) with AdamW-style adaptive updates and optional Sophia-style curvature-aware elements. It supports projected-normal-momentum (PNM), stochastic blockwise second-moment estimation (SN), Adaptive Gradient Clipping (AGC), lookahead slow/fast blending, per-parameter shape-based adjusted learning rates, D-Adapt automatic global step-size adaptation, Hutchinson Hessian estimation, and layer-wise trust-ratio scaling. Use `use_muon=True` to prefer the Muon flattened-momentum pipeline; set it to `False` to run the AdamW/Sophia-style branch.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base learning rate used by the optimizer (Muon branch uses this; AdamW branch uses `adamw_lr`).
* **`beta1`** *(float, default=0.9)*: Exponential decay rate (first-moment / momentum coefficient) used in moment updates.
* **`beta2`** *(float, default=0.999)*: Exponential decay rate for second-moment / variance estimates.
* **`beta3`** *(float, default=0.9999)*: (AEM) decay for the slow exponential average when `aem=True`.
* **`epsilon`** *(float, default=1e-8)*: Small constant added for numerical stability in denominators.
* **`weight_decay`** *(float, default=1e-2)*: L2 weight decay coefficient applied (decoupled or added to gradients depending on `weight_decouple`).
* **`weight_decouple`** *(bool, default=True)*: If `True`, apply decoupled multiplicative weight decay (AdamW-style); otherwise add L2 term to gradients.
* **`nesterov`** *(bool, default=True)*: Enable Nesterov-style correction for momentum updates in the Muon path.
* **`ns_steps`** *(int, default=5)*: Number of Newton–Schulz iterations used by the zero-power normalization routine for Muon updates.
* **`use_adjusted_lr`** *(bool, default=False)*: Use a per-parameter shape-based adjusted learning-rate heuristic for Muon updates.
* **`adamw_lr`** *(float, default=3e-4)*: Learning rate used by the AdamW / Sophia branch when `use_muon=False`.
* **`adamw_wd`** *(float, default=0.0)*: Weight decay coefficient used by the AdamW branch.
* **`use_muon`** *(bool, default=True)*: Choose Muon flattened-momentum pipeline (True) or AdamW/Sophia pipeline (False).
* **`subset_size`** *(int, default=-1)*: Block size hint for stochastic/blockwise normalization (SN). Negative lets the optimizer pick a heuristic.
* **`sn`** *(bool, default=False)*: Enable stochastic blockwise second-moment estimation to reduce memory for large tensors.
* **`lookahead_merge_time`** *(int, default=5)*: Frequency (in steps) to merge lookahead slow/fast weights.
* **`lookahead_blending_alpha`** *(float, default=0.5)*: Blending factor used by lookahead slow/fast updates.
* **`lookahead`** *(bool, default=False)*: Enable lookahead slow/fast parameter averaging.
* **`pnm`** *(bool, default=False)*: Use projected-normal-momentum (pos/neg momentum pair) instead of a single first-moment buffer.
* **`agc`** *(bool, default=False)*: Apply Adaptive Gradient Clipping to scale/clamp gradients before updates.
* **`cautious`** *(bool, default=False)*: Apply cautious masking that scales updates whose sign disagrees with raw gradients (reduces risky updates).
* **`aem`** *(bool, default=False)*: Maintain an auxiliary slow exponential moving average and optionally mix it into the first-moment.
* **`alpha`** *(float, default=5.0)*: Blending coefficient used by the auxiliary EMA when `aem=True`.
* **`t_alpha_beta3`** *(int or None, default=None)*: Scheduling length (steps) to warm up `alpha`/`beta3`. If `None`, no schedule is used.
* **`sophia`** *(bool, default=False)*: Enable Sophia-style curvature-aware components (Hessian-moment using Hutchinson estimation) on the AdamW branch.
* **`p`** *(float, default=1e-2)*: Sophia-style clipping/scaling parameter (algorithm-specific).
* **`update_period`** *(int, default=10)*: Frequency (in steps) to compute or integrate Sophia/Hessian updates (Hutchinson sampling frequency).
* **`num_samples`** *(int, default=1)*: Number of Hutchinson samples to estimate Hessian-vector products for Sophia.
* **`hessian_distribution`** *(str, default='gaussian')*: Distribution for Hutchinson estimation: `'gaussian'` or `'rademacher'`.
* **`d0`** *(float, default=1e-6)*: Initial scalar used by the D-Adapt automatic step-size adaptation routine.
* **`growth_rate`** *(float, default=float('inf'))*: Maximum allowed multiplicative growth of the D-Adapt scalar between updates.
* **`DAdapt`** *(bool, default=False)*: Enable D-Adapt automatic global step-size adaptation (maintains `d0_`, accumulators).
* **`trust_ratio`** *(bool, default=False)*: Apply layer-wise trust-ratio scaling (parameter norm / update norm) similar to LARS/LAMB.
* **`trust_clip`** *(bool, default=False)*: If True, clip the trust ratio to at most 1.0.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional)*: Standard TensorFlow gradient clipping options, forwarded to the base optimizer.
* **`use_ema`** *(bool, default=False)*: Maintain an Exponential Moving Average (EMA) of model weights if enabled.
* **`ema_momentum`** *(float, default=0.99)*: EMA momentum factor (when `use_ema=True`).
* **`ema_overwrite_frequency`** *(int or None, default=None)*: Frequency for overwriting EMA weights (if specified).
* **`loss_scale_factor`** *(float or None, default=None)*: Optional static loss scaling for mixed-precision training.
* **`gradient_accumulation_steps`** *(int or None, default=None)*: Number of steps to accumulate gradients before applying an update (if used).
* **`name`** *(str, default="adamuon\_e")*: Name of the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.adamuon_e import AdaMuon_e

# Instantiate optimizer
optimizer = AdaMuon_e(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=1e-2,
    weight_decouple=True,
    nesterov=True,
    ns_steps=5,
    use_adjusted_lr=False,
    adamw_betas=(0.9, 0.999),
    adamw_lr=3e-4,
    adamw_wd=0.0,
    use_muon=True,
    subset_size=256,
    sn=True,
    pnm=True,
    agc=True,
    sophia=True,
    update_period=10,
    num_samples=1,
    hessian_distribution='gaussian',
    DAdapt=True,
    d0=1e-6,
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

# AdaGO

**Overview**:

The `AdaGO` optimizer is a hybrid optimizer that blends Muon-style momentum normalization for large/matrix-like parameters with an AdamW-style adaptive update branch. It uses a normalized Newton–Schulz zero-power routine for stable direction normalization, supports optional Nesterov momentum, per-parameter shape-based adjusted learning rates, multiplicative (decoupled) weight decay, and a guarded adaptive step-size mechanism that rescales learning by gradient norms and a running curvature accumulator. Switch between the Muon pipeline (`use_muon=True`) and the AdamW-style adaptive branch (`use_muon=False`) depending on your parameter shapes and preferences.

**Parameters**:

* **`learning_rate`** *(float, default=5e-2)*: Base learning rate for Muon updates (AdamW branch uses `adamw_lr`).
* **`epsilon`** *(float, default=5e-4)*: Small constant used in denominator or as a lower bound to avoid division by zero in the Muon update rule.
* **`weight_decay`** *(float, default=0.0)*: L2 weight decay coefficient applied (when `weight_decouple=True` this is applied as decoupled multiplicative decay).
* **`momentum`** *(float, default=0.95)*: Momentum coefficient for the Muon momentum buffer.
* **`weight_decouple`** *(bool, default=True)*: If `True`, apply decoupled multiplicative weight decay; otherwise the L2 term is added to gradients.
* **`gamma`** *(float, default=10.0)*: Upper bound parameter used to clip or cap gradient norm contribution in the Muon v-accumulator (controls adaptive denominator growth).
* **`v`** *(float, default=1e-6)*: Initial value used for the running curvature accumulator (prevents tiny denominators).
* **`nesterov`** *(bool, default=True)*: Enable Nesterov-style correction for momentum updates in the Muon branch.
* **`ns_steps`** *(int, default=5)*: Number of Newton–Schulz iterations used by the zero-power normalization routine to compute normalized update directions.
* **`use_adjusted_lr`** *(bool, default=False)*: Use per-parameter shape-based adjusted learning rate heuristic (scales LR according to parameter shape).
* **`adamw_lr`** *(float, default=3e-4)*: Learning rate used by the AdamW-style adaptive branch.
* **`adamw_betas`** *(tuple, default=(0.9, 0.95))*: Beta coefficients `(beta1, beta2)` used by the AdamW-style updates.
* **`adamw_wd`** *(float, default=0.0)*: Weight decay coefficient used by the AdamW branch (applied decoupled if `weight_decouple=True`).
* **`adamw_eps`** *(float, default=1e-10)*: Epsilon for numerical stability in the AdamW denominator.
* **`maximize`** *(bool, default=False)*: If `True`, perform gradient ascent (negate gradients).
* **`cautious`** *(bool, default=False)*: When enabled, apply cautious masking to scale updates whose sign disagrees with raw gradients (reduces risky updates).
* **`use_muon`** *(bool, default=True)*: Choose Muon flattened-momentum pipeline (True) or AdamW-style adaptive pipeline (False).
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional)*: Standard TensorFlow gradient clipping options forwarded to the base optimizer.
* **`use_ema`** *(bool, default=False)*: Maintain an Exponential Moving Average (EMA) of model weights.
* **`ema_momentum`** *(float, default=0.99)*: EMA momentum factor (when `use_ema=True`).
* **`ema_overwrite_frequency`** *(int or None, default=None)*: Frequency to overwrite EMA weights if specified.
* **`loss_scale_factor`** *(float or None, default=None)*: Optional static loss scaling factor for mixed-precision training.
* **`gradient_accumulation_steps`** *(int or None, default=None)*: Number of steps to accumulate gradients before applying an update (if used).
* **`name`** *(str, default="adago")*: Name of the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.adago import AdaGO

# Instantiate optimizer
optimizer = AdaGO(
    learning_rate=5e-2,
    epsilon=5e-4,
    weight_decay=1e-3,
    momentum=0.95,
    weight_decouple=True,
    gamma=10.0,
    v=1e-6,
    nesterov=True,
    ns_steps=5,
    use_adjusted_lr=False,
    adamw_lr=3e-4,
    adamw_betas=(0.9, 0.95),
    adamw_wd=0.0,
    adamw_eps=1e-10,
    use_muon=True,
    cautious=False,
)

# Compile a model
model = ...  # build a tf.keras.Model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# AdaGO\_e

**Overview**:

The `AdaGO_e` optimizer is a hybrid optimizer that combines a Muon-style normalized momentum pipeline for large / matrix-like parameters with an AdamW-style adaptive branch. It includes features for normalized direction computation (Newton–Schulz zero-power normalization), optional Nesterov momentum, per-parameter shape-based adjusted learning rates, decoupled weight decay, stochastic-noise momentum (PNM) support, subset normalization (SN) for block-wise second-moment estimation, optional Sophia-style Hessian clipping, adaptive DAdapt scaling, lookahead blending, adaptive element-wise momentum (AEM), gradient clipping/AGC, and optional Hutchinson Hessian estimation. Use `use_muon=True` to prefer the Muon pipeline (normalized-momentum updates) or `use_muon=False` to use the AdamW-style adaptive updates.

**Parameters**:

* **`learning_rate`** *(float, default=5e-2)*: Base learning rate used by the Muon-style pipeline. The AdamW branch uses `adamw_lr`.
* **`epsilon`** *(float, default=5e-4)*: Small constant used to stabilize denominators and lower-bound adaptive scalars.
* **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 weight decay. When `weight_decouple=True`, decay is applied multiplicatively (decoupled).
* **`momentum`** *(float, default=0.95)*: Momentum coefficient for Muon momentum buffers.
* **`weight_decouple`** *(bool, default=True)*: If `True`, apply decoupled multiplicative weight decay; otherwise add L2 term to gradients.
* **`gamma`** *(float, default=10.0)*: Cap parameter (upper bound) used in the Muon branch to limit curvature accumulator growth.
* **`v`** *(float, default=1e-6)*: Initial value for the running curvature accumulator (prevents tiny denominators).
* **`nesterov`** *(bool, default=True)*: Enable Nesterov-style correction when computing momentum updates.
* **`ns_steps`** *(int, default=5)*: Number of Newton–Schulz iterations used by the zero-power normalization routine to form normalized directions.
* **`use_adjusted_lr`** *(bool, default=False)*: Use a per-parameter shape-based adjusted learning rate heuristic that scales LR by parameter shape.
* **`adamw_lr`** *(float, default=3e-4)*: Learning rate for the AdamW-style adaptive branch.
* **`adamw_betas`** *(tuple, default=(0.9, 0.95, 0.9999))*: Beta coefficients `(beta1, beta2, beta3)` used by the AdamW-style updates.
* **`adamw_wd`** *(float, default=0.0)*: Weight decay used by the AdamW branch (decoupled if `weight_decouple=True`).
* **`adamw_eps`** *(float, default=1e-10)*: Epsilon value for numerical stability in the AdamW denominator.
* **`maximize`** *(bool, default=False)*: If `True`, perform gradient ascent (negate gradients internally).
* **`use_muon`** *(bool, default=True)*: Choose Muon normalized-momentum pipeline (`True`) or AdamW-style adaptive pipeline (`False`).
* **`subset_size`** *(int, default=-1)*: Subset size used for subset normalization (SN). Negative means auto heuristic.
* **`sn`** *(bool, default=False)*: Enable subset normalization (block-wise second-moment aggregation) to reduce memory and stabilize large tensors.
* **`lookahead_merge_time`** *(int, default=5)*: Number of steps between lookahead synchronizations.
* **`lookahead_blending_alpha`** *(float, default=0.5)*: Blending factor for lookahead slow weights when merging.
* **`lookahead`** *(bool, default=False)*: Enable Lookahead-style slow/fast weight blending.
* **`pnm`** *(bool, default=False)*: Use stochastic-noise momentum (PNM) variant for momentum (alternating pos/neg buffers) for noise-robust momentum.
* **`agc`** *(bool, default=False)*: Apply Adaptive Gradient Clipping (AGC) to gradient updates.
* **`cautious`** *(bool, default=False)*: When enabled, apply cautious masking (scale updates whose sign disagrees with raw gradients) to reduce risky updates.
* **`aem`** *(bool, default=False)*: Apply Adaptive Element-wise Momentum (AEM) machinery (auxiliary slow momentum) to augment updates.
* **`alpha`** *(float, default=5.0)*: Scaling factor for the AEM contribution (when `aem=True`).
* **`t_alpha_beta3`** *(int or None, default=None)*: Timescale used to schedule `alpha` and `beta3` when AEM scheduling is desired.
* **`sophia`** *(bool, default=False)*: Enable Sophia-style Hessian moment tracking / clipping (uses stored Hutchinson estimates or internal Hessian moments to bound updates).
* **`p`** *(float, default=1e-2)*: Clip bound for Sophia-style clipping (when active).
* **`update_period`** *(int, default=10)*: Periodicity (in steps) for performing Hutchinson Hessian accumulation or Sophia hessian updates.
* **`num_samples`** *(int, default=1)*: Number of Hutchinson probe samples for stochastic Hessian estimation (when using Hutchinson).
* **`hessian_distribution`** *(str, default='gaussian')*: Distribution for Hutchinson probes: `'gaussian'` or `'rademacher'`.
* **`d0`** *(float, default=1e-6)*: Initial DAdapt base scale (used by the DAdapt automatic scaling mechanism).
* **`growth_rate`** *(float, default=inf)*: Maximum multiplicative growth factor for the DAdapt scale per update.
* **`DAdapt`** *(bool, default=False)*: Enable DAdapt — automatic global learning-rate scaling based on accumulated inner products.
* **`trust_ratio`** *(bool, default=False)*: Enable layer-wise trust ratio (ratio of parameter norm to update norm) scaling.
* **`trust_clip`** *(bool, default=False)*: Clip trust-ratio to at most 1.0 when `trust_ratio=True`.
* **`clipnorm`**, **`clipvalue`**, **`global_clipnorm`** *(optional)*: Standard TensorFlow gradient clipping options forwarded to the base optimizer.
* **`use_ema`** *(bool, default=False)*: Maintain an Exponential Moving Average (EMA) of model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for EMA weights (when `use_ema=True`).
* **`ema_overwrite_frequency`** *(int or None, default=None)*: Frequency to overwrite EMA weights if specified.
* **`loss_scale_factor`** *(float or None, default=None)*: Optional static loss-scaling factor for mixed-precision training.
* **`gradient_accumulation_steps`** *(int or None, default=None)*: Steps to accumulate gradients before applying an update (if used).
* **`name`** *(str, default="adago\_e")*: Name of the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.adago_e import AdaGO_e

# Instantiate optimizer
optimizer = AdaGO_e(
    learning_rate=5e-2,
    epsilon=5e-4,
    weight_decay=1e-3,
    momentum=0.95,
    use_muon=True,              # Muon-style normalized momentum pipeline
    use_adjusted_lr=False,      # per-parameter shape LR scaling
    adamw_lr=3e-4,              # LR for AdamW branch (if used)
    adamw_betas=(0.9, 0.95),
    ns_steps=5,
    pnm=True,                   # stochastic-noise momentum
    sn=True,                    # subset normalization for large params
    lookahead=True,
    aem=False,
    sophia=True,                # enable Sophia-style Hessian moments
    DAdapt=True,
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

# DAdaptLion_e

**Overview**:

`DAdaptLion_e` is a sign-based (Lion-style) optimizer extended with a data-dependent global scalar adaptation (DAdapt), optional orthogonal-gradient removal, optional low-rank projector support for 2-D parameters, Lookahead, stochastic-noise momentum (PNM) alternatives, Adaptive Gradient Clipping (AGC), cautious masking, trust-ratio layer-wise scaling, and an optional Muon-style orthogonalization step for projected updates. It is designed for stable large-scale training where a global adaptive step scalar and structured/orthogonal updates can improve convergence and robustness.

**Parameters**:

* **`learning_rate`** *(float, default=1.0)*: Base learning-rate multiplier. The effective step often combines this with the adaptive DAdapt scalar.
* **`beta1`** *(float, default=0.9)*: Decay rate for the first-moment / momentum estimate (used for EMA or PNM construction).
* **`beta2`** *(float, default=0.999)*: Decay rate used for second-stage accumulators and for weighting in DAdapt statistics.
* **`weight_decay`** *(float, default=0.0)*: L2 weight-decay coefficient. Applied either multiplicatively (decoupled) or added to gradients depending on `weight_decouple`.
* **`d0`** *(float, default=1e-6)*: Initial base DAdapt scalar; adaptive algorithm updates `d0_` from global statistics to scale effective step sizes.
* **`weight_decouple`** *(bool, default=True)*: If `True` apply decoupled multiplicative weight decay; if `False` weight decay is added to gradients.
* **`fixed_decay`** *(bool, default=False)*: When `True`, use a fixed decay multiplier (unaffected by the adaptive DAdapt scalar); otherwise decay may be scaled by DAdapt.
* **`orthograd`** *(bool, default=False)*: If `True`, remove the component of each gradient parallel to its parameter (orthogonal-gradient transform) before updates.
* **`lookahead_merge_time`** *(int, default=5)*: Frequency (in steps) at which the slow (Lookahead) weights are blended into the fast weights.
* **`lookahead_blending_alpha`** *(float, default=0.5)*: Blending factor for Lookahead slow/fast weight merging.
* **`lookahead`** *(bool, default=False)*: Enable Lookahead slow/fast blending when `True`.
* **`pnm`** *(bool, default=False)*: Use stochastic-noise momentum (PNM) variant instead of standard EMA momentum buffers when `True`.
* **`agc`** *(bool, default=False)*: Apply Adaptive Gradient Clipping (unit-wise gradient clipping) when enabled.
* **`cautious`** *(bool, default=False)*: Apply cautious masking that down-weights update components that disagree in sign with the instantaneous gradient (stabilizes updates).
* **`update_proj_gap`** *(int or None, default=None)*: If set and a variable is 2-D, build/use a low-rank projector every `update_proj_gap` steps; `None` disables projector behavior.
* **`scale`** *(float or None, default=None)*: Scale parameter passed to projector construction (projector-specific).
* **`projection_type`** *(str or None, default=None)*: Projection type passed to projector construction (projector-specific behaviour).
* **`trust_ratio`** *(bool, default=False)*: When `True`, apply a layer-wise trust-ratio scaling (scale updates by ‖w‖/‖g‖) before applying the sign update.
* **`trust_clip`** *(bool, default=False)*: If `True`, clip the computed trust ratio to a maximum of `1.0`.
* **`muon_ortho`** *(bool, default=False)*: If `True` (and a projected 2-D variable is present), apply a Muon-style orthogonalization (zero-power Newton–Schulz) to the projected update.
* **`muon_steps`** *(int, default=5)*: Number of Newton–Schulz steps to run for `muon_ortho` processing.
* **`clipnorm`** *(float, optional)*: Clip gradients by per-variable norm (forwarded to the base optimizer).
* **`clipvalue`** *(float, optional)*: Clip gradients by value (forwarded to the base optimizer).
* **`global_clipnorm`** *(float, optional)*: Clip gradients by a global norm across all variables (forwarded to the base optimizer).
* **`use_ema`** *(bool, default=False)*: Maintain an Exponential Moving Average (EMA) of model weights for evaluation/averaging.
* **`ema_momentum`** *(float, default=0.99)*: Momentum factor for EMA updates if `use_ema` is `True`.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) at which EMA weights may be overwritten (optional).
* **`loss_scale_factor`** *(float, optional)*: Static loss-scaling multiplier (useful for mixed-precision training).
* **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before performing an optimizer update.
* **`name`** *(str, default="dadaptlion_e")*: Name of the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.dadaptlion import DAdaptLion_e

# Instantiate optimizer with default sensible settings
optimizer = DAdaptLion_e(
    learning_rate=1.0,
    beta1=0.9,
    beta2=0.999,
    d0=1e-6,
    weight_decay=1e-4,
    weight_decouple=True,
    fixed_decay=False,
    orthograd=False,
    lookahead=False,
    pnm=False,
    agc=False,
    cautious=False,
    update_proj_gap=None,   # enable projector only for large 2-D matrices
    scale=0.1,
    projection_type="truncated",
    trust_ratio=False,
    muon_ortho=False,
    muon_steps=5,
)

# Compile and train a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Conda

**Overview**:

The `Conda` optimizer is a projector-augmented Adam-style optimizer that applies adaptive moment estimation together with optional low-rank projection for 2-D parameter matrices. For matrices, a `GaLoreProjector` can be used to project gradients and momentum into a low-rank subspace at configurable intervals (`update_proj_gap`) to reduce computation and enable structured updates. `Conda` uses bias-corrected first and second moments and supports standard options such as weight decay, maximization mode, and common gradient clipping hooks.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base step length used with Adam bias corrections.
* **`beta1`** *(float, default=0.9)*: Exponential decay rate for the first moment (momentum) estimate.
* **`beta2`** *(float, default=0.999)*: Exponential decay rate for the second moment estimate.
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability when dividing by second moment.
* **`weight_decay`** *(float, default=0.0)*: L2 weight decay coefficient (applied multiplicatively after the parameter update).
* **`update_proj_gap`** *(int or None, default=None)*: If set and the variable is 2-D, how frequently (in steps) to update/use the low-rank projector; `None` disables projection.
* **`scale`** *(float or None, default=None)*: Scale parameter passed to the projector (projector-specific).
* **`projection_type`** *(str or None, default=None)*: Projection type passed into `GaLoreProjector` (projector-specific).
* **`maximize`** *(bool, default=False)*: If `True`, the optimizer maximizes the objective (gradient is negated).
* **`clipnorm`** *(float, optional)*: Clip gradients by norm (forwarded to base optimizer).
* **`clipvalue`** *(float, optional)*: Clip gradients by value (forwarded to base optimizer).
* **`global_clipnorm`** *(float, optional)*: Clip gradients by a global norm (forwarded to base optimizer).
* **`use_ema`** *(bool, default=False)*: Maintain Exponential Moving Average (EMA) of model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum used by EMA when enabled.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights, if used.
* **`loss_scale_factor`** *(float, optional)*: Static loss scaling factor (useful for mixed precision).
* **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before applying an update.
* **`name`** *(str, default="conda")*: Name of the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.conda import Conda

optimizer = Conda(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=1e-4,
    update_proj_gap=100,        # enable GaLore projection for 2-D parameters every 100 steps
    scale=0.1,
    projection_type="some_type",
)

model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Conda_e

**Overview**:

`Conda_e` is an extended Conda variant that integrates additional features for robustness and adaptivity: stochastic-noise momentum (PNM), subset-normalization (SN) for block-wise second moment computation, Adaptive Gradient Clipping (AGC), cautious masking, an AEM-style slow momentum term, trust-ratio (layer-wise learning rate adaptation), and DAdapt — a global data-dependent adaptive scalar that rescales effective step sizes. It also supports the same low-rank `GaLoreProjector` mechanism for 2-D parameters. `Conda_e` aims to combine projector-based structured updates with advanced adaptive rescaling and stabilization methods for large-scale training.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: Base learning rate (referred to internally as `lr`).
* **`beta1`** *(float, default=0.9)*: Decay for first-moment / momentum estimates.
* **`beta2`** *(float, default=0.999)*: Decay for second-moment / variance estimates and DAdapt accumulators.
* **`beta3`** *(float, default=0.9999)*: Decay used by the AEM slow momentum (if `aem=True`).
* **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
* **`weight_decay`** *(float, default=0.0)*: L2 weight decay coefficient (applied multiplicatively or as configured).
* **`update_proj_gap`** *(int or None, default=None)*: Projection update interval for 2-D parameters; `None` disables projection.
* **`scale`** *(float or None, default=None)*: Scale passed to the projector.
* **`projection_type`** *(str or None, default=None)*: Projection mode for `GaLoreProjector`.
* **`maximize`** *(bool, default=False)*: If `True`, optimize to maximize the objective.
* **`lookahead_merge_time`** *(int, default=5)*: Steps between Lookahead slow/fast syncs.
* **`lookahead_blending_alpha`** *(float, default=0.5)*: Blending factor for Lookahead slow weights.
* **`lookahead`** *(bool, default=False)*: Enable Lookahead slow/fast blending.
* **`pnm`** *(bool, default=False)*: Use stochastic-noise momentum (PNM) instead of standard EMA first-moment.
* **`subset_size`** *(int, default=-1)*: Subset block size used by subset-normalization (SN). Negative means heuristic selection.
* **`sn`** *(bool, default=False)*: Enable subset-normalization for block-wise stats on large tensors.
* **`agc`** *(bool, default=False)*: Apply Adaptive Gradient Clipping.
* **`cautious`** *(bool, default=False)*: Scale updates using cautious masking to avoid destructive sign flips.
* **`aem`** *(bool, default=False)*: Enable AEM slow momentum (additional momentum term mixed into updates).
* **`alpha`** *(float, default=5.0)*: Mixing coefficient used by AEM if enabled.
* **`t_alpha_beta3`** *(int or None, default=None)*: Time horizon to schedule `alpha`/`beta3` (used by AEM scheduling).
* **`d0`** *(float, default=1e-6)*: Initial DAdapt base scale.
* **`growth_rate`** *(float, default=inf)*: Max growth factor for the DAdapt scaler per update (used to constrain updates).
* **`DAdapt`** *(bool, default=False)*: Enable the DAdapt global adaptive scalar mechanism.
* **`trust_ratio`** *(bool, default=False)*: Enable layer-wise trust-ratio adaptation (norm-based scaling).
* **`trust_clip`** *(bool, default=False)*: Clip trust ratio to ≤ 1.0 when enabled.
* **`clipnorm`** *(float, optional)*: Clip gradients by norm (forwarded).
* **`clipvalue`** *(float, optional)*: Clip gradients by value (forwarded).
* **`global_clipnorm`** *(float, optional)*: Clip gradients by global norm (forwarded).
* **`use_ema`** *(bool, default=False)*: Maintain EMA of model weights.
* **`ema_momentum`** *(float, default=0.99)*: EMA momentum.
* **`ema_overwrite_frequency`** *(int or None, default=None)*: EMA overwrite frequency.
* **`loss_scale_factor`** *(float or None, default=None)*: Static loss-scaling factor.
* **`gradient_accumulation_steps`** *(int or None, default=None)*: Number of steps to accumulate gradients.
* **`name`** *(str, default="conda_e")*: Name of the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.conda import Conda_e

optimizer = Conda_e(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    beta3=0.9999,
    epsilon=1e-8,
    weight_decay=1e-4,
    update_proj_gap=200,
    scale=0.1,
    projection_type="some_type",
    pnm=True,
    sn=True,
    agc=True,
    cautious=True,
    aem=True,
    alpha=5.0,
    d0=1e-6,
    DAdapt=True,
    trust_ratio=False,
    lookahead=True,
    lookahead_merge_time=5,
    lookahead_blending_alpha=0.5,
)

model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```

# BCOS

**Overview**:

The `BCOS` optimizer is a custom optimization algorithm that provides flexible control over momentum and variance updates through distinct modes. It features mechanisms for weight decoupling, simple conditional variance updates, and optional maximization strategies, making it adaptable to various training scenarios where standard optimizers may not suffice.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
* **`beta`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates (momentum) or variance, depending on the mode.
* **`beta2`** *(float, optional)*: Exponential decay rate for the second moment estimates. If `None`, it defaults based on `beta` or internal logic.
* **`mode`** *(str, default='c')*: Operation mode determining how momentum and variance are computed. Options include `'g'`, `'m'`, and `'c'`.
* **`simple_cond`** *(bool, default=False)*: Whether to use a simplified conditional update for variance computation.
* **`weight_decay`** *(float, default=0.1)*: Coefficient for weight decay.
* **`weight_decouple`** *(bool, default=True)*: If `True`, applies decoupled weight decay (similar to AdamW). If `False`, applies standard L2 regularization.
* **`epsilon`** *(float, default=1e-6)*: Small constant for numerical stability to avoid division by zero.
* **`maximize`** *(bool, default=False)*: If `True`, the optimizer maximizes the objective function (gradient ascent) instead of minimizing it.
* **`clipnorm`** *(float, optional)*: Clips gradients by norm.
* **`clipvalue`** *(float, optional)*: Clips gradients by value.
* **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
* **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
* **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients.
* **`name`** *(str, default="bcos")*: Name of the optimizer.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.bcos import BCOS

# Instantiate optimizer
optimizer = BCOS(
    learning_rate=1e-3,
    beta=0.9,
    mode='c',
    weight_decay=0.01,
    weight_decouple=True
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# BCOS_e

**Overview**:

The `BCOS_e` optimizer is an extended, highly configurable optimization algorithm designed for complex training scenarios. It builds upon the BCOS framework by integrating advanced techniques including Subspace Normalization (SN), Second-order optimization (Sophia), Trust Region methods, Cautious updates, and Lookahead mechanisms. This optimizer allows for fine-grained control over gradient reshaping, Hessian-based curvature estimation, and dynamic update smoothing.

**Parameters**:

* **`learning_rate`** *(float, default=1e-3)*: The step size for parameter updates.
* **`beta`** *(float, default=0.9)*: Exponential decay rate for the first moment estimates (momentum) or variance, depending on the mode.
* **`beta2`** *(float, optional)*: Exponential decay rate for the second moment estimates (or Hessian moment in Sophia mode).
* **`mode`** *(str, default='c')*: Operation mode determining how momentum and variance are computed (`'g'`, `'m'`, `'c'`).
* **`simple_cond`** *(bool, default=False)*: Whether to use a simplified conditional update for variance computation.
* **`weight_decay`** *(float, default=0.1)*: Coefficient for weight decay.
* **`weight_decouple`** *(bool, default=True)*: If `True`, applies decoupled weight decay (similar to AdamW). If `False`, applies standard L2 regularization.
* **`epsilon`** *(float, default=1e-6)*: Small constant for numerical stability.
* **`subset_size`** *(int, default=-1)*: Target size for gradient subsetting in Subspace Normalization. If `-1`, it is dynamically calculated based on variable size.
* **`sn`** *(bool, default=True)*: Enables Subspace Normalization. When `True`, gradients are reshaped and normalized over subsets.
* **`agc`** *(bool, default=False)*: Enables Adaptive Gradient Clipping to stabilize training.
* **`gc`** *(bool, default=False)*: Enables Gradient Centralization, which centers gradients to have zero mean.
* **`sophia`** *(bool, default=False)*: Enables Sophia (Second-order Clipping) optimization logic using Hutchinson's estimator for the Hessian.
* **`p`** *(float, default=1e-2)*: Clipping parameter used when `sophia` is enabled.
* **`update_period`** *(int, default=10)*: Frequency (in steps) of Hessian updates when `sophia` is enabled.
* **`num_samples`** *(int, default=1)*: Number of samples used for Hutchinson's Hessian estimation.
* **`hessian_distribution`** *(str, default='gaussian')*: Distribution used for Hessian estimation (`'gaussian'` or `'rademacher'`).
* **`trust_ratio`** *(bool, default=False)*: Enables layer-wise learning rate adaptation based on the ratio of weight norms to update norms.
* **`trust_clip`** *(bool, default=False)*: If `True`, clips the calculated trust ratio to be at most 1.0.
* **`cautious`** *(bool, default=False)*: Enables Cautious updates, masking steps where the update direction opposes the gradient.
* **`lookahead_merge_time`** *(int, default=5)*: Number of steps before merging the fast weights into the slow weights (Lookahead mechanism).
* **`lookahead_blending_alpha`** *(float, default=0.5)*: The interpolation factor for merging slow and fast weights.
* **`lookahead`** *(bool, default=False)*: Enables the Lookahead mechanism.
* **`maximize`** *(bool, default=False)*: If `True`, performs gradient ascent instead of descent.
* **`clipnorm`** *(float, optional)*: Clips gradients by norm.
* **`clipvalue`** *(float, optional)*: Clips gradients by value.
* **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
* **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
* **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
* **`gradient_accumulation_steps`** *(int, optional)*: Steps for accumulating gradients.
* **`name`** *(str, default="bcos_e")*: Name of the optimizer.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.bcos import BCOS_e

# Instantiate optimizer with Sophia and Lookahead enabled
optimizer = BCOS_e(
    learning_rate=1e-3,
    beta=0.965,
    beta2=0.99,
    mode='c',
    sophia=True,
    update_period=10,
    lookahead=True,
    weight_decay=0.05
)

# Compile a model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

# Ano

**Overview**:

The `Ano` optimizer is an Adam-style optimization algorithm that introduces a **sign-based second-moment adaptation mechanism**. Instead of directly tracking the exponential moving average of squared gradients, it updates the second moment using the **sign difference between the current squared gradient and the historical estimate**, enabling more responsive variance tracking under non-stationary or noisy gradients. This design emphasizes stability while preserving adaptive scaling behavior.

**Parameters**:

* **`learning_rate`** *(float, default=1e-4)*: The learning rate used to scale parameter updates.
* **`beta1`** *(float, default=0.92)*: Exponential decay rate for the first-moment estimates.
* **`beta2`** *(float, default=0.99)*: Exponential decay rate for the second-moment estimates.
* **`epsilon`** *(float, default=1e-8)*: Small constant added for numerical stability.
* **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay regularization.
* **`weight_decouple`** *(bool, default=True)*: Enables decoupled weight decay instead of applying it directly to gradients.
* **`fixed_decay`** *(bool, default=False)*: Applies fixed weight decay independent of the learning rate.
* **`logarithmic_schedule`** *(bool, default=False)*: Dynamically adjusts `beta1` using a logarithmic schedule based on the optimization step.
* **`maximize`** *(bool, default=False)*: Maximizes the objective instead of minimizing it.
* **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
* **`clipvalue`** *(float, optional)*: Clips gradients by their value.
* **`global_clipnorm`** *(float, optional)*: Clips gradients by the global norm.
* **`use_ema`** *(bool, default=False)*: Applies an exponential moving average to model weights.
* **`ema_momentum`** *(float, default=0.99)*: Momentum used for EMA updates.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency at which EMA weights overwrite model weights.
* **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for accumulating gradients.
* **`name`** *(str, default="ano")*: Name of the optimizer.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.ano import Ano

optimizer = Ano(
    learning_rate=1e-4,
    weight_decay=1e-2,
    logarithmic_schedule=True
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_dataset, epochs=10)
```

# Ano_e

**Overview**:

The `Ano_e` optimizer is an extended and highly modular variant of `Ano`, designed to support a wide range of advanced optimization techniques within a unified framework. It integrates optional components such as **subset-normalized second moments**, **adaptive gradient clipping**, **gradient centralization**, **Hessian-based updates (Sophia-style)**, **trust-ratio scaling**, **lookahead optimization**, and **D-Adaptation**. This optimizer is intended for research and experimentation with hybrid optimization strategies.

**Parameters**:

* **`learning_rate`** *(float, default=1e-4)*: Base learning rate for parameter updates.
* **`beta1`** *(float, default=0.92)*: Exponential decay rate for first-moment estimates.
* **`beta2`** *(float, default=0.99)*: Exponential decay rate for second-moment or Hessian-moment estimates.
* **`epsilon`** *(float, default=1e-8)*: Numerical stability constant.
* **`weight_decay`** *(float, default=0.0)*: Weight decay coefficient.
* **`weight_decouple`** *(bool, default=True)*: Enables decoupled weight decay.
* **`fixed_decay`** *(bool, default=False)*: Uses fixed decay instead of learning-rate-scaled decay.
* **`logarithmic_schedule`** *(bool, default=False)*: Applies logarithmic scheduling to `beta1`.
* **`subset_size`** *(int, default=-1)*: Size of subsets used for subset-normalized second-moment estimation.
* **`sn`** *(bool, default=True)*: Enables subset-normalized second-moment computation.
* **`agc`** *(bool, default=False)*: Enables adaptive gradient clipping based on unit-wise norms.
* **`gc`** *(bool, default=False)*: Enables gradient centralization.
* **`sophia`** *(bool, default=False)*: Enables Hessian-based updates using Hutchinson trace estimation.
* **`p`** *(float, default=1e-2)*: Clipping threshold for Hessian-based or sign-based updates.
* **`update_period`** *(int, default=10)*: Update frequency for Hessian estimation or delayed updates.
* **`num_samples`** *(int, default=1)*: Number of samples used in Hutchinson Hessian estimation.
* **`hessian_distribution`** *(str, default="gaussian")*: Distribution used for Hutchinson estimation.
* **`trust_ratio`** *(bool, default=False)*: Enables layer-wise trust ratio scaling.
* **`trust_clip`** *(bool, default=False)*: Clips the trust ratio to a maximum of 1.
* **`cautious`** *(bool, default=False)*: Applies sign-consistency masking to updates.
* **`lookahead_merge_time`** *(int, default=5)*: Number of steps between lookahead synchronization.
* **`lookahead_blending_alpha`** *(float, default=0.5)*: Interpolation factor for lookahead updates.
* **`lookahead`** *(bool, default=False)*: Enables lookahead optimization.
* **`DAdapt`** *(bool, default=False)*: Enables D-Adaptation for dynamic learning-rate scaling.
* **`d0`** *(float, default=1e-6)*: Initial D-Adaptation scaling factor.
* **`growth_rate`** *(float, default=inf)*: Maximum growth rate for D-Adaptation.
* **`maximize`** *(bool, default=False)*: Maximizes the objective function.
* **`clipnorm`** *(float, optional)*: Clips gradients by norm.
* **`clipvalue`** *(float, optional)*: Clips gradients by value.
* **`global_clipnorm`** *(float, optional)*: Clips gradients by global norm.
* **`use_ema`** *(bool, default=False)*: Applies exponential moving average to weights.
* **`ema_momentum`** *(float, default=0.99)*: EMA momentum.
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting model weights with EMA.
* **`loss_scale_factor`** *(float, optional)*: Loss scaling factor.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of gradient accumulation steps.
* **`name`** *(str, default="ano_e")*: Name of the optimizer.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.ano import Ano_e

optimizer = Ano_e(
    learning_rate=3e-4,
    sophia=True,
    sn=True,
    agc=True,
    lookahead=True,
    DAdapt=False
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_dataset, epochs=10)
```

# Optimizer

**Overview**:

The `Optimizer` class is the comprehensive base class for all Keras optimizers, providing an extensive framework for implementing state-of-the-art gradient-based optimization algorithms. It supports advanced features including exponential moving average (EMA), gradient accumulation, multiple gradient clipping strategies, weight decay, adaptive learning rates (D-Adapt), orthogonal gradients, positive-negative momentum (PNM), Sophia-style Hessian estimation, lookahead optimization, and subset normalization for memory-efficient training.

**Parameters**:

* **`learning_rate`** *(float or LearningRateSchedule)*: The step size for parameter updates. Can be a constant float value, a `LearningRateSchedule` instance, or a callable that returns the learning rate based on the current iteration.
* **`weight_decay`** *(float, optional)*: Coefficient for weight decay regularization. If set, applies weight decay to model parameters.
* **`clipnorm`** *(float, optional)*: If set, gradients are individually clipped so that their norm does not exceed this value.
* **`clipvalue`** *(float, optional)*: If set, gradients are clipped by value to be no higher than this threshold.
* **`global_clipnorm`** *(float, optional)*: If set, the global norm of all gradients is clipped to not exceed this value.
* **`use_ema`** *(bool, default=False)*: Whether to apply Exponential Moving Average to model weights. When enabled, maintains a moving average of model weights for improved inference performance.
* **`ema_momentum`** *(float, default=0.99)*: Momentum for EMA computation. Only used when `use_ema=True`. Must be in the range [0, 1].
* **`ema_overwrite_frequency`** *(int, optional)*: Frequency (in steps) for overwriting model variables with their moving average. Only used when `use_ema=True`. If `None`, variables are updated at the end of training.
* **`loss_scale_factor`** *(float, optional)*: Factor for scaling the loss during gradient computation. Useful for preventing underflow in mixed precision training.
* **`gradient_accumulation_steps`** *(int, optional)*: Number of steps to accumulate gradients before updating variables. Must be >= 2 if specified. Useful for simulating larger batch sizes with limited memory.
* **`name`** *(str, optional)*: Name of the optimizer instance. If not provided, auto-generates a name based on the class name.

**Advanced Optimizer Features**:

The base optimizer supports several advanced features that can be enabled in derived classes:

* **`sn`** *(bool)*: Enables subset normalization for memory-efficient second moment estimation in high-dimensional parameters.
* **`sophia`** *(bool)*: Enables Sophia-style Hessian estimation using Hutchinson's trace estimator for improved convergence.
* **`lookahead`** *(bool)*: Enables lookahead optimization, maintaining slow-moving weights that provide stability.
* **`DAdapt`** *(bool)*: Enables D-Adapt adaptive learning rate scheduling for automatic learning rate adjustment.
* **`pnm`** *(bool)*: Enables Positive-Negative Momentum for improved optimization dynamics.
* **`orthograd`** *(bool)*: Enables orthogonal gradient projection to maintain gradient orthogonality to parameters.

**Key Methods**:

* **`build(var_list)`**: Initialize optimizer variables for the given list of trainable variables. Automatically sets up state variables for enabled features (momentum, variance, Hessian, etc.).
* **`update_step(gradient, variable, learning_rate)`**: Implement the core update logic for a single variable. Must be overridden in subclasses.
* **`apply_gradients(grads_and_vars, tape=None)`**: Apply gradients to variables. Accepts a list of (gradient, variable) pairs and an optional GradientTape for Hessian computation.
* **`exclude_from_weight_decay(var_list, var_names)`**: Exclude specific variables or variables matching name patterns from weight decay.
* **`finalize_variable_values(var_list)`**: Finalize variable values, such as applying EMA averages. Called automatically at the end of training.
* **`agc(p, grad, agc_eps, agc_clip_val, eps)`**: Apply Adaptive Gradient Clipping to prevent gradient explosion.
* **`gc(grads, gradient, idx)`**: Apply gradient centralization to improve optimization stability.
* **`apply_orthogonal_gradients(params, grads, eps)`**: Project gradients to be orthogonal to parameters.
* **`apply_trust_ratio(variable, update)`**: Apply trust ratio scaling (as in LARS/LAMB optimizers).
* **`apply_cautious(update, gradient)`**: Apply cautious update masking to filter out conflicting updates.
* **`apply_pnm(gradient, step, idx)`**: Apply Positive-Negative Momentum update rule.
* **`get_config()`**: Returns the optimizer configuration as a serializable dictionary.
* **`set_weights(weights)`**: Set optimizer state from a list of weight arrays.
