## TensorBox API Reference

### Class Overview
The `TensorBox` class encapsulates a `Tensor` within a computation graph `Context`, providing automatic differentiation capabilities and a fluent API for tensor operations.

---

### Constructors
* public TensorBox(Tensor tensor, bool requireGrad = true)
* public TensorBox(Context ctx)

---

### Properties
* public bool RequireGrad { get; set; } # Within the `SimpleNN.Config.NoGrad()` scope, `RequireGrad` is automatically set to `false`.
* public int[] Size { get; }
* public int NDim { get; }

---

### Core Methods (Computation Graph)
* public void Backward()
* public Tensor GetTensor()
* public Tensor GetGrad()
* public TensorBox Detach()
* public void ZeroGrad()
* public void StepGrad(Context.StepFunction stepFunc)

---

### Shape Manipulation
* public TensorBox Reshape(params int[] shape)
* public TensorBox Permute(params int[] dims)
* public TensorBox Transpose(int dim0, int dim1)
* public TensorBox Unsqueeze(int dim)
* public TensorBox Squeeze(int dim)
* public TensorBox Squeeze()
* public static TensorBox Concat(int dim, params TensorBox[] tensors)
* public static TensorBox Stack(int dim, params TensorBox[] tensors)

* public static TensorBox Stack(int dim, params TensorBox[] tensors)
* public TensorBox Pad((int left, int right)[] paddingSizes, PaddingMode mode = PaddingMode.CONSTANT, float value = 0.0f)

---

### Broadcasting
* public static (TensorBox broadA, TensorBox broadB) BroadcastBoth(TensorBox a, TensorBox b)
* public TensorBox Broadcast(params int[] targetSize)
* public TensorBox Broadcast(Tensor tensor)
* public TensorBox Broadcast(TensorBox tensor)

---

### Arithmetic Operators
* public static TensorBox operator +(TensorBox a, TensorBox b)
* public static TensorBox operator +(TensorBox a, float value)
* public static TensorBox operator +(float value, TensorBox a)
* public static TensorBox operator -(TensorBox a, TensorBox b)
* public static TensorBox operator -(TensorBox a, float value)
* public static TensorBox operator -(float value, TensorBox a)
* public static TensorBox operator -(TensorBox a) (Unary)
* public static TensorBox operator *(TensorBox a, TensorBox b)
* public static TensorBox operator *(TensorBox a, float value)
* public static TensorBox operator *(float value, TensorBox a)
* public static TensorBox operator /(TensorBox a, TensorBox b)
* public static TensorBox operator /(TensorBox a, float value)
* public static TensorBox operator /(float value, TensorBox a)

---

### Mathematical Functions
* public static TensorBox MatMul(TensorBox a, TensorBox b)
* public static TensorBox Conv2d(TensorBox input, TensorBox weight, int stride = 1, int pad = 0, TensorBox bias = null)
* public static TensorBox Conv2d(TensorBox input, TensorBox weight, int strideH, int strideW, int padH, int padW, TensorBox bias = null)
* public static TensorBox TransposedConv2d(TensorBox input, TensorBox weight, int stride = 1, int pad = 0, TensorBox bias = null)
* public static TensorBox TransposedConv2d(TensorBox input, TensorBox weight, int strideH, int strideW, int padH, int padW, TensorBox bias = null)
* public static TensorBox Pow(TensorBox a, TensorBox b)
* public TensorBox Pow(TensorBox b)
* public static TensorBox Pow(TensorBox a, float power)
* public TensorBox Pow(float power)
* public static TensorBox Pow(float baseValue, TensorBox a)
* public TensorBox Sqrt()
* public TensorBox Square()
* public TensorBox Cube()
* public TensorBox Sign()
* public TensorBox Abs()
* public TensorBox Exp()
* public TensorBox MExp()
* public TensorBox Log()
* public TensorBox Sin()
* public TensorBox Cos()
* public TensorBox Tan()
* public TensorBox Tanh()
* public TensorBox Sech()
* public TensorBox Sigmoid()
* public TensorBox Softplus()

---

### Comparison & Activation Functions
* public static TensorBox Maximum(TensorBox a, TensorBox b)
* public TensorBox Maximum(float value)
* public static TensorBox Minimum(TensorBox a, TensorBox b)
* public TensorBox Minimum(float value)
* public TensorBox ReLU()
* public TensorBox LeakyReLU(float negativeSlope = 0.01f)
* public static TensorBox operator <(TensorBox a, TensorBox b)
* public static TensorBox operator <(TensorBox a, float b)
* public static TensorBox operator <(float a, TensorBox b)
* public static TensorBox operator >(TensorBox a, TensorBox b)
* public static TensorBox operator >(TensorBox a, float b)
* public static TensorBox operator >(float a, TensorBox b)
* public static TensorBox operator <=(TensorBox a, TensorBox b)
* public static TensorBox operator <=(TensorBox a, float b)
* public static TensorBox operator <=(float a, TensorBox b)
* public static TensorBox operator >=(TensorBox a, TensorBox b)
* public static TensorBox operator >=(TensorBox a, float b)
* public static TensorBox operator >=(float a, TensorBox b)
* public static TensorBox operator ==(TensorBox a, TensorBox b)
* public static TensorBox operator ==(TensorBox a, float b)
* public static TensorBox operator ==(float a, TensorBox b)
* public static TensorBox operator !=(TensorBox a, TensorBox b)
* public static TensorBox operator !=(TensorBox a, float b)
* public static TensorBox operator !=(float a, TensorBox b)
* public static bool operator ==(TensorBox a, float? b)
* public static bool operator ==(float? a, TensorBox b)
* public static bool operator !=(TensorBox a, float? b)
* public static bool operator !=(float? a, TensorBox b)

---

### Reduction Functions
* public TensorBox Max(int dim, bool keepDims = false)
* public TensorBox Max()
* public TensorBox Mean(int dim, bool keepDims = false)
* public TensorBox Mean()
* public TensorBox Sum(int dim, bool keepDims = false)
* public TensorBox Sum()

---

### Distribution & Statistics Functions
* public static TensorBox Softmax(TensorBox tensor, int dim)
* public TensorBox Softmax(int dim)
* public static TensorBox LBeta(TensorBox a, TensorBox b)
* public static TensorBox BetaExp(TensorBox a, TensorBox b)
* public static TensorBox BetaMode(TensorBox a, TensorBox b)
* public static TensorBox BetaEntropy(TensorBox a, TensorBox b)
* public static TensorBox BetaLogprob(TensorBox a, TensorBox b, TensorBox x)

---

### Utility & Conversions
* public static implicit operator TensorBox(Tensor tensor)
* public override string ToString()

---

### Serialization
* public static byte[] ToBinary(TensorBox box)
* public static TensorBox FromBinary(byte[] data)
* public static TensorBox FromBinary(BinaryReader reader)