### Optimizer Modules

### Namespace: SimpleNN.Optimizer

#### Optimizer (Abstract)

The abstract base class for all optimizers. Manages parameters and provides methods for gradient zeroing and parameter updates.

  * public Optimizer(params List\<TensorBox\>[] parameters)
  * public int NumParameters { get; }
  * public void ZeroGrad()
  * public void Step()
  * protected abstract Tensor StepGrad(int id, Tensor tensor, Tensor grad)

#### SGD

Implements the Stochastic Gradient Descent (SGD) optimization algorithm. Inherits from `Optimizer`.

  * public SGD(float learningRate, params List\<TensorBox\>[] parameters)

#### Adagrad

Implements the Adagrad (Adaptive Gradient Algorithm) optimization algorithm. Inherits from `Optimizer`.

  * public Adagrad(float learningRate, float epsilon, params List\<TensorBox\>[] parameters)
  * public Adagrad(float learningRate, params List\<TensorBox\>[] parameters)

#### Adadelta

Implements the Adadelta optimization algorithm, which adapts learning rates based on a moving window of gradient updates. Inherits from `Optimizer`.

  * public Adadelta(float rho, float epsilon, params List\<TensorBox\>[] parameters)
  * public Adadelta(params List\<TensorBox\>[] parameters)

#### Adam

Implements the Adam (Adaptive Moment Estimation) optimization algorithm. Inherits from `Optimizer`.

  * public Adam(float learningRate, float beta1, float beta2, float epsilon, params List\<TensorBox\>[] parameters)
  * public Adam(float learningRate, params List\<TensorBox\>[] parameters)

#### RAdam

Implements the RAdam (Rectified Adam) optimization algorithm, which introduces a term to rectify the variance of the adaptive learning rate. Inherits from `Optimizer`.

  * public RAdam(float learningRate, float beta1, float beta2, float epsilon, params List\<TensorBox\>[] parameters)
  * public RAdam(float learningRate, params List\<TensorBox\>[] parameters)