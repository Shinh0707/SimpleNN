## Module API Reference

### Namespace: SimpleNN.Module

### Base Classes

#### Module (Abstract)

The abstract base class for all neural network modules. Manages parameters and sub-modules.

  * public List\<TensorBox\> Parameters()
  * protected T AddModule\<T\>(T module) where T : Module
  * protected TensorBox AddParameter(TensorBox param)
  * protected TensorBox[] AddParameters(params TensorBox[] param)

#### SingleInSingleOutModule (Abstract)

An abstract module that takes a single `TensorBox` as input and returns a single `TensorBox` as output. Inherits from `Module`.

  * public abstract TensorBox Forward(TensorBox inputTensor)

#### Loss (Abstract)

The abstract base class for all loss functions. Inherits from `Module`.

  * public abstract TensorBox Forward(TensorBox input, TensorBox target)

-----

### Container Modules

#### Sequential

A sequential container of modules. Modules are executed in the order they are passed in the constructor. Inherits from `SingleInSingleOutModule`.

  * public Sequential(params SingleInSingleOutModule[] modules)
  * public override TensorBox Forward(TensorBox inputTensor)

#### ModuleList

Holds a list of modules. Can be indexed like a list. Inherits from `Module`.

  * public ModuleList(params Module[] modules)
  * public Module this[int i] { get; }

-----

### Layer Modules

#### Linear

Applies a linear transformation (y = xW + b) to the input data. Inherits from `SingleInSingleOutModule`.

  * public Linear(int inFeatures, int outFeatures, bool bias = true)
  * public override TensorBox Forward(TensorBox inputTensor)

#### Conv2d

Applies a 2D convolution operation. Inherits from `SingleInSingleOutModule`.

  * public Conv2d(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0, bool bias = true)
  * public Conv2d(int inChannels, int outChannels, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, bool bias = true)
  * public override TensorBox Forward(TensorBox inputTensor)

#### TransposedConv2d

Applies a 2D transposed convolution operation. Inherits from `SingleInSingleOutModule`.

  * public TransposedConv2d(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0, bool bias = true)
  * public TransposedConv2d(int inChannels, int outChannels, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, bool bias = true)
  * public override TensorBox Forward(TensorBox inputTensor)

-----

### Activation Functions

#### ReLU

Applies the Rectified Linear Unit function element-wise. Inherits from `SingleInSingleOutModule`.

  * public override TensorBox Forward(TensorBox inputTensor)

#### Sigmoid

Applies the Sigmoid function element-wise. Inherits from `SingleInSingleOutModule`.

  * public override TensorBox Forward(TensorBox inputTensor)

#### Tanh

Applies the Hyperbolic Tangent (Tanh) function element-wise. Inherits from `SingleInSingleOutModule`.

  * public override TensorBox Forward(TensorBox inputTensor)

#### Sin

Applies the Sine function element-wise. Inherits from `SingleInSingleOutModule`.

  * public override TensorBox Forward(TensorBox inputTensor)

#### Sech

Applies the Hyperbolic Secant (Sech) function element-wise. Inherits from `SingleInSingleOutModule`.

  * public override TensorBox Forward(TensorBox inputTensor)

-----

### Loss Functions

#### L1Loss

Calculates the Mean Absolute Error (L1 Loss). Inherits from `Loss`.

  * public override TensorBox Forward(TensorBox input, TensorBox target)

#### MSELoss

Calculates the Mean Squared Error (L2 Loss). Inherits from `Loss`.

  * public override TensorBox Forward(TensorBox input, TensorBox target)

-----

### Distribution Modules

#### BetaDistibution

A module representing a Beta distribution. Inherits from `Module`.

  * public BetaDistibution(TensorBox alpha, TensorBox beta)
  * public TensorBox LogProb(TensorBox x)
  * public TensorBox Exp { get; }
  * public TensorBox Mode { get; }
  * public TensorBox Entropy { get; }