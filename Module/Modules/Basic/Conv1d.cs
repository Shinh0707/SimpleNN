namespace SimpleNN.Module
{
    using System;
    using SimpleNN.Graph;
    using SimpleNN.Tensor;

    /// <summary>
    /// 1D Convolution Layer Module (Conv1d).
    /// </summary>
    public class Conv1d : SingleInSingleOutModule
    {
        private TensorBox _weight;
        private TensorBox _bias;
        
        private int _stride, _padLeft, _padRight;
        private bool _useBias;

        /// <summary>
        /// Initializes a 1D convolution module (symmetric padding).
        /// </summary>
        public Conv1d(
            int inChannels, 
            int outChannels, 
            int kernelSize, 
            int stride = 1, 
            int padding = 0, 
            bool bias = true)
            : this(
                  inChannels, outChannels, 
                  kernelSize, 
                  stride, 
                  padding, padding, 
                  bias)
        {
        }

        /// <summary>
        /// Initializes a 1D convolution module (asymmetric padding).
        /// </summary>
        public Conv1d(
            int inChannels, 
            int outChannels, 
            int kernelSize, 
            int stride, 
            int padLeft, 
            int padRight, 
            bool bias = true)
        {
            _stride = stride;
            _padLeft = padLeft;
            _padRight = padRight;
            _useBias = bias;

            InitializeWeights(inChannels, outChannels, kernelSize);
            if (bias)
            {
                InitializeBias(outChannels);
            }
        }

        private void InitializeWeights(int inChannels, int outChannels, int kernelSize)
        {
            // fan-in based initialization
            int fanIn = inChannels * kernelSize;
            float limit = 1.0f / MathF.Sqrt(fanIn);
            
            var tensor = Tensor.Random(
                new int[] { outChannels, inChannels, kernelSize }, 
                -limit, 
                limit
            );
            
            _weight = AddParameter(new TensorBox(tensor));
        }

        private void InitializeBias(int outChannels)
        {
            var tensor = Tensor.Zeros(new int[] { outChannels });
            _bias = AddParameter(new TensorBox(tensor));
        }

        public int[] GetOutputSize(params int[] inputSize)
        {
            if (inputSize.Length != 3)
            {
                throw new ArgumentException("Input must be a 3D tensor (N, C, L)", nameof(inputSize));
            }
            
            int n = inputSize[0];
            int lIn = inputSize[2];

            int cOut = _weight.Size[0];
            int k = _weight.Size[2];

            // L_out = floor((L_in + padLeft + padRight - K) / S) + 1
            int lOut = (lIn + _padLeft + _padRight - k) / _stride + 1;
            
            return new int[] { n, cOut, lOut };
        }

        public override TensorBox Forward(TensorBox inputTensor)
        {
            return TensorBox.Conv1d(
                inputTensor, 
                _weight, 
                _stride, 
                _padLeft, _padRight,
                _useBias ? _bias : null
            );
        }
    }

    /// <summary>
    /// 1D Transposed Convolution Layer Module (TransposedConv1d).
    /// </summary>
    public class TransposedConv1d : SingleInSingleOutModule
    {
        private TensorBox _weight;
        private TensorBox _bias;
        
        private int _stride, _padLeft, _padRight;
        private bool _useBias;

        public TransposedConv1d(
            int inChannels, 
            int outChannels, 
            int kernelSize, 
            int stride = 1, 
            int padding = 0, 
            bool bias = true)
            : this(
                inChannels, outChannels, 
                kernelSize, 
                stride, 
                padding, padding, 
                bias)
        {
        }

        public TransposedConv1d(
            int inChannels, 
            int outChannels, 
            int kernelSize, 
            int stride, 
            int padLeft, 
            int padRight, 
            bool bias = true)
        {
            _stride = stride;
            _padLeft = padLeft;
            _padRight = padRight;
            _useBias = bias;

            InitializeWeights(inChannels, outChannels, kernelSize);
            if (bias)
            {
                InitializeBias(outChannels);
            }
        }

        private void InitializeWeights(int inChannels, int outChannels, int kernelSize)
        {
            int fanIn = inChannels; // fan-in based on input channels
            float limit = 1.0f / MathF.Sqrt(fanIn);
            
            // Weight shape for TransposedConv1d: (C_in, C_out, K)
            var tensor = Tensor.Random(
                new int[] { inChannels, outChannels, kernelSize }, 
                -limit, 
                limit
            );
            
            _weight = AddParameter(new TensorBox(tensor));
        }

        private void InitializeBias(int outChannels)
        {
            var tensor = Tensor.Zeros(new int[] { outChannels });
            _bias = AddParameter(new TensorBox(tensor));
        }

        public int[] GetOutputSize(params int[] inputSize)
        {
            if (inputSize.Length != 3)
            {
                throw new ArgumentException("Input must be a 3D tensor (N, C, L)", nameof(inputSize));
            }
            
            int n = inputSize[0];
            int lIn = inputSize[2];

            int cOut = _weight.Size[1];
            int k = _weight.Size[2];

            // L_out = (L_in - 1) * S - padLeft - padRight + K
            int lOut = (lIn - 1) * _stride - _padLeft - _padRight + k;
            
            return new int[] { n, cOut, lOut };
        }

        public override TensorBox Forward(TensorBox inputTensor)
        {
            return TensorBox.TransposedConv1d(
                inputTensor, 
                _weight, 
                _stride, 
                _padLeft, _padRight,
                _useBias ? _bias : null
            );
        }
    }

    /// <summary>
    /// 1D Max Pooling Layer Module (MaxPool1d).
    /// </summary>
    public class MaxPool1d : SingleInSingleOutModule
    {
        private int _kernelSize, _stride, _padLeft, _padRight;

        public MaxPool1d(int kernelSize, int stride = -1, int padding = 0)
        {
            int s = (stride <= 0) ? kernelSize : stride;
            Initialize(kernelSize, s, padding, padding);
        }
        
        public MaxPool1d(
            int kernelSize, 
            int stride, 
            int padLeft, int padRight)
        {
            Initialize(kernelSize, stride, padLeft, padRight);
        }

        private void Initialize(
            int kernelSize, 
            int stride, 
            int padLeft, int padRight)
        {
            _kernelSize = kernelSize;
            _stride = stride;
            _padLeft = padLeft;
            _padRight = padRight;
        }
        
        public int[] GetOutputSize(params int[] inputSize)
        {
            if (inputSize.Length != 3)
            {
                throw new ArgumentException("Input must be a 3D tensor", nameof(inputSize));
            }
            
            int n = inputSize[0];
            int cOut = inputSize[1]; 
            int lIn = inputSize[2];

            int lOut = (lIn + _padLeft + _padRight - _kernelSize) / _stride + 1;
            
            return new int[] { n, cOut, lOut };
        }

        public override TensorBox Forward(TensorBox inputTensor)
        {
            return TensorBox.MaxPool1d(
                inputTensor, 
                _kernelSize, 
                _stride, 
                _padLeft, _padRight
            );
        }
    }

    /// <summary>
    /// 1D Average Pooling Layer Module (AvgPool1d).
    /// </summary>
    public class AvgPool1d : SingleInSingleOutModule
    {
        private int _kernelSize, _stride, _padLeft, _padRight;

        public AvgPool1d(int kernelSize, int stride = -1, int padding = 0)
        {
            int s = (stride <= 0) ? kernelSize : stride;
            Initialize(kernelSize, s, padding, padding);
        }
        
        public AvgPool1d(
            int kernelSize, 
            int stride, 
            int padLeft, int padRight)
        {
            Initialize(kernelSize, stride, padLeft, padRight);
        }

        private void Initialize(
            int kernelSize, 
            int stride, 
            int padLeft, int padRight)
        {
            _kernelSize = kernelSize;
            _stride = stride;
            _padLeft = padLeft;
            _padRight = padRight;
        }

        public int[] GetOutputSize(params int[] inputSize)
        {
            if (inputSize.Length != 3)
            {
                throw new ArgumentException("Input must be a 3D tensor", nameof(inputSize));
            }
            
            int n = inputSize[0];
            int cOut = inputSize[1]; 
            int lIn = inputSize[2];

            int lOut = (lIn + _padLeft + _padRight - _kernelSize) / _stride + 1;
            
            return new int[] { n, cOut, lOut };
        }

        public override TensorBox Forward(TensorBox inputTensor)
        {
            return TensorBox.AvgPool1d(
                inputTensor, 
                _kernelSize, 
                _stride, 
                _padLeft, _padRight
            );
        }
    }
}
