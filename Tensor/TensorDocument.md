## Tensor API Reference

### Class Overview
The `Tensor` class is the fundamental data structure in SimpleNN, representing a multi-dimensional array. It stores raw `float[]` data and manages shape, strides, and contiguity.

---

### Constructors
* public Tensor(float[] data, int[] size, int[] strides)
* public Tensor(float[] data, int[] size)

---

### Properties
* public int[] Size { get; set; }
* public long TotalSize { get; }
* public int NDim { get; }
* public bool IsContiguous { get; }
* public bool IsScaler { get; }

---

### Core Methods
* public float Item()
* public float[] GetContiguousData()
* public override string ToString()
* public static bool IsSameSizeAndStrides(Tensor a, Tensor b)

---

### Implicit Conversions
* public static implicit operator Tensor(float value)
* public static implicit operator Tensor(int value)
* public static implicit operator Tensor(float[] values)
* public static implicit operator Tensor(int[] values)
* public static implicit operator Tensor(float[,] values)

---

### Tensor Generation
* public static Tensor Fill(int[] size, float value, int[] strides = null)
* public static Tensor FillLike(Tensor t, float value)
* public static Tensor Ones(int[] size, int[] strides = null)
* public static Tensor OnesLike(Tensor t)
* public static Tensor Zeros(int[] size, int[] strides = null)
* public static Tensor ZerosLike(Tensor t)
* public static Tensor Random(int[] size, float min = 0.0f, float max = 1.0f, int[] strides = null)
* public static Tensor RandomLike(Tensor t, float min = 0.0f, float max = 1.0f)
* public static Tensor Normal(int[] size, float loc = 0.0f, float std = 1.0f, int[] strides = null)
* public static Tensor NormalLike(Tensor t, float loc = 0.0f, float std = 1.0f)

---

### Shape Manipulation & Views
* public static Tensor Reshape(Tensor tensor, int[] newSize)
* public static Tensor Permute(Tensor tensor, params int[] dims)
* public static Tensor Transpose(Tensor tensor, int dim0, int dim1)
* public static Tensor Unsqueeze(Tensor tensor, int dim)
* public static Tensor Squeeze(Tensor tensor, int dim)
* public static Tensor Squeeze(Tensor tensor)
* public static Tensor Contiguous(Tensor tensor)
* public static Tensor Concat(int dim, params Tensor[] tensors)
* public static Tensor[] Split(int dim, Tensor tensor, int splitSize)
* public static Tensor[] Split(int dim, Tensor tensor, params int[] splitSizes)
* public static Tensor Stack(int dim, params Tensor[] tensors)
* public static Tensor[] Unstack(int dim, Tensor tensor)

* public static Tensor[] Unstack(int dim, Tensor tensor)

---

### Padding
* public static Tensor Pad(Tensor tensor, (int left, int right)[] paddingSizes, PaddingMode mode = PaddingMode.CONSTANT, float value = 0.0f)
* public static Tensor ConstantPad(Tensor tensor, (int left, int right)[] paddingSizes, float value = 0.0f)
* public static Tensor ReflectPad(Tensor tensor, (int left, int right)[] paddingSizes)
* public static Tensor ReplicatePad(Tensor tensor, (int left, int right)[] paddingSizes)
* public static Tensor CircularPad(Tensor tensor, (int left, int right)[] paddingSizes)

---

### Broadcasting
* public static Tensor Broadcast(Tensor tensor, int dim, int targetSize)
* public static Tensor Broadcast(Tensor tensor, int[] targetSize)
* public void Broadcast(int dim, int targetSize)
* public static bool CheckBroadcast(int size, int targetSize, bool invokeException=false)
* public static bool TryBroadcast(int size, int stride, int targetSize, out int newSize, out int newStride)

---

### Arithmetic Operators
* public static Tensor operator +(Tensor a, Tensor b)
* public static Tensor operator +(Tensor a, float value)
* public static Tensor operator +(float value, Tensor a)
* public static Tensor operator -(Tensor a, Tensor b)
* public static Tensor operator -(Tensor a, float value)
* public static Tensor operator -(float value, Tensor a)
* public static Tensor operator -(Tensor a) (Unary)
* public static Tensor operator *(Tensor a, Tensor b)
* public static Tensor operator *(Tensor a, float value)
* public static Tensor operator *(float value, Tensor a)
* public static Tensor operator /(Tensor a, Tensor b)
* public static Tensor operator /(Tensor a, float value)
* public static Tensor operator /(float value, Tensor a)

---

### Mathematical Functions
* public static Tensor MatMul(Tensor a, Tensor b)
* public static Tensor Conv2d(Tensor input, Tensor weight, Tensor bias = null, int stride = 1, int padding = 0)
* public static Tensor Conv2d(Tensor input, Tensor weight, Tensor bias, int strideH, int strideW, int padH, int padW)
* public static Tensor TransposedConv2d(Tensor input, Tensor weight, Tensor bias = null, int stride = 1, int padding = 0)
* public static Tensor TransposedConv2d(Tensor input, Tensor weight, Tensor bias, int strideH, int strideW, int padH, int padW)
* public static Tensor Pow(Tensor a, Tensor b)
* public static Tensor Pow(Tensor a, float power)
* public static Tensor Pow(float baseValue, Tensor a)
* public static Tensor Sqrt(Tensor a)
* public static Tensor Square(Tensor a)
* public static Tensor Cube(Tensor a)
* public static Tensor Abs(Tensor a)
* public static Tensor Reciprocal(Tensor a)
* public static Tensor Sign(Tensor a)
* public static Tensor Exp(Tensor a)
* public static Tensor MExp(Tensor a)
* public static Tensor Log(Tensor a)
* public static Tensor Tanh(Tensor a)
* public static Tensor Sech(Tensor a)
* public static Tensor Sigmoid(Tensor a)
* public static Tensor LGamma(Tensor a)
* public static Tensor Digamma(Tensor a)
* public static Tensor Trigamma(Tensor a)

---

### Trigonometric Functions
* public static Tensor Sin(Tensor x)
* public static Tensor Cos(Tensor x)
* public static Tensor Tan(Tensor x)
* public static Tensor Csc(Tensor a)
* public static Tensor Sec(Tensor a)
* public static Tensor Cot(Tensor a)

---

### Comparison & Activation Functions
* public static Tensor Maximum(Tensor a, Tensor b)
* public static Tensor Maximum(Tensor a, float b)
* public static Tensor Maximum(float b, Tensor a)
* public static Tensor Minimum(Tensor a, Tensor b)
* public static Tensor Minimum(Tensor a, float b)
* public static Tensor Minimum(float b, Tensor a)
* public static Tensor ReLU(Tensor a)
* public static Tensor LeakyReLU(Tensor a, float negativeSlope)
If you want to check `null`, you should use `tensor is null` / `tensor is not null`.  
* public static Tensor operator >=(Tensor a, Tensor b)
* public static Tensor operator >=(Tensor a, float b)
* public static Tensor operator >=(float a, Tensor b)
* public static Tensor operator >(Tensor a, Tensor b)
* public static Tensor operator >(Tensor a, float b)
* public static Tensor operator >(float a, Tensor b)
* public static Tensor operator <=(Tensor a, Tensor b)
* public static Tensor operator <=(Tensor a, float b)
* public static Tensor operator <=(float a, Tensor b)
* public static Tensor operator <(Tensor a, Tensor b)
* public static Tensor operator <(Tensor a, float b)
* public static Tensor operator <(float a, Tensor b)
* public static Tensor operator ==(Tensor a, Tensor b)
* public static Tensor operator ==(Tensor a, float b)
* public static Tensor operator ==(float a, Tensor b)
* public static Tensor operator !=(Tensor a, Tensor b)
* public static Tensor operator !=(Tensor a, float b)
* public static Tensor operator !=(float a, Tensor b)

---

### Reduction Functions
* public static Tensor Max(Tensor tensor, int dim, bool keepDims = false)
* public static Tensor Max(Tensor tensor)
* public static Tensor Mean(Tensor tensor, int dim, bool keepDims = false)
* public static Tensor Mean(Tensor tensor)
* public static Tensor Sum(Tensor tensor, int dim, bool keepDims = false)
* public static Tensor Sum(Tensor tensor)

---

### Texture Conversion
* public static Tensor AsRGB(Texture texture)
* public static Tensor AsRGBA(Texture texture)
* public static Tensor AsHSV(Texture texture)
* public static Tensor AsConicalHSV(Texture texture)  
If you read same texture and set to same tensor many time, recommend to use SetAs~.  
* public Tensor SetAsRGB(Texture texture)
* public Tensor SetAsRGBA(Texture texture)
* public Tensor SetAsHSV(Texture texture)
* public Tensor SetAsConicalHSV(Texture texture)