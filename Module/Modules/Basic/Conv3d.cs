namespace SimpleNN.Module
{
    using System;
    using SimpleNN.Graph;
    using SimpleNN.Tensor;

    /// <summary>
    /// 3D Convolution Layer Module (Conv3d).
    /// </summary>
    public class Conv3d : SingleInSingleOutModule
    {
        private TensorBox _weight;
        private TensorBox _bias;
        
        private int _strideD, _strideH, _strideW;
        private int _padD, _padH, _padW;
        private bool _useBias;

        public Conv3d(
            int inChannels, 
            int outChannels, 
            int kernelSize, 
            int stride = 1, 
            int padding = 0, 
            bool bias = true)
            : this(
                  inChannels, outChannels, 
                  kernelSize, kernelSize, kernelSize,
                  stride, stride, stride,
                  padding, padding, padding,
                  bias)
        {
        }

        public Conv3d(
            int inChannels, 
            int outChannels, 
            int kernelD, int kernelH, int kernelW, 
            int strideD, int strideH, int strideW, 
            int padD, int padH, int padW, 
            bool bias = true)
        {
            _strideD = strideD;
            _strideH = strideH;
            _strideW = strideW;
            _padD = padD;
            _padH = padH;
            _padW = padW;
            _useBias = bias;

            InitializeWeights(inChannels, outChannels, kernelD, kernelH, kernelW);
            if (bias)
            {
                InitializeBias(outChannels);
            }
        }

        private void InitializeWeights(int inChannels, int outChannels, int kernelD, int kernelH, int kernelW)
        {
            int fanIn = inChannels * kernelD * kernelH * kernelW;
            float limit = 1.0f / MathF.Sqrt(fanIn);
            
            var tensor = Tensor.Random(
                new int[] { outChannels, inChannels, kernelD, kernelH, kernelW }, 
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
            if (inputSize.Length != 5)
            {
                throw new ArgumentException("Input must be a 5D tensor (N, C, D, H, W)", nameof(inputSize));
            }
            
            int n = inputSize[0];
            int dIn = inputSize[2];
            int hIn = inputSize[3];
            int wIn = inputSize[4];

            int cOut = _weight.Size[0];
            int kD = _weight.Size[2];
            int kH = _weight.Size[3];
            int kW = _weight.Size[4];

            int dOut = (dIn + 2 * _padD - kD) / _strideD + 1;
            int hOut = (hIn + 2 * _padH - kH) / _strideH + 1;
            int wOut = (wIn + 2 * _padW - kW) / _strideW + 1;
            
            return new int[] { n, cOut, dOut, hOut, wOut };
        }

        public override TensorBox Forward(TensorBox inputTensor)
        {
            return TensorBox.Conv3d(
                inputTensor, 
                _weight, 
                _strideD, _strideH, _strideW,
                _padD, _padH, _padW,
                _useBias ? _bias : null
            );
        }
    }

    /// <summary>
    /// 3D Transposed Convolution Layer Module (TransposedConv3d).
    /// </summary>
    public class TransposedConv3d : SingleInSingleOutModule
    {
        private TensorBox _weight;
        private TensorBox _bias;
        
        private int _strideD, _strideH, _strideW;
        private int _padD, _padH, _padW;
        private bool _useBias;

        public TransposedConv3d(
            int inChannels, 
            int outChannels, 
            int kernelSize, 
            int stride = 1, 
            int padding = 0, 
            bool bias = true)
            : this(
                inChannels, outChannels, 
                kernelSize, kernelSize, kernelSize,
                stride, stride, stride,
                padding, padding, padding,
                bias)
        {
        }

        public TransposedConv3d(
            int inChannels, 
            int outChannels, 
            int kernelD, int kernelH, int kernelW, 
            int strideD, int strideH, int strideW, 
            int padD, int padH, int padW, 
            bool bias = true)
        {
            _strideD = strideD;
            _strideH = strideH;
            _strideW = strideW;
            _padD = padD;
            _padH = padH;
            _padW = padW;
            _useBias = bias;

            InitializeWeights(inChannels, outChannels, kernelD, kernelH, kernelW);
            if (bias)
            {
                InitializeBias(outChannels);
            }
        }

        private void InitializeWeights(int inChannels, int outChannels, int kernelD, int kernelH, int kernelW)
        {
            int fanIn = inChannels;
            float limit = 1.0f / MathF.Sqrt(fanIn);
            
            var tensor = Tensor.Random(
                new int[] { inChannels, outChannels, kernelD, kernelH, kernelW }, 
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
            if (inputSize.Length != 5)
            {
                throw new ArgumentException("Input must be a 5D tensor (N, C, D, H, W)", nameof(inputSize));
            }
            
            int n = inputSize[0];
            int dIn = inputSize[2];
            int hIn = inputSize[3];
            int wIn = inputSize[4];

            int cOut = _weight.Size[1];
            int kD = _weight.Size[2];
            int kH = _weight.Size[3];
            int kW = _weight.Size[4];

            int dOut = (dIn - 1) * _strideD - 2 * _padD + kD;
            int hOut = (hIn - 1) * _strideH - 2 * _padH + kH;
            int wOut = (wIn - 1) * _strideW - 2 * _padW + kW;
            
            return new int[] { n, cOut, dOut, hOut, wOut };
        }

        public override TensorBox Forward(TensorBox inputTensor)
        {
            return TensorBox.TransposedConv3d(
                inputTensor, 
                _weight, 
                _strideD, _strideH, _strideW,
                _padD, _padH, _padW,
                _useBias ? _bias : null
            );
        }
    }

    /// <summary>
    /// 3D Max Pooling Layer Module (MaxPool3d).
    /// </summary>
    public class MaxPool3d : SingleInSingleOutModule
    {
        private int _kernelD, _kernelH, _kernelW;
        private int _strideD, _strideH, _strideW;
        private int _padD, _padH, _padW;

        public MaxPool3d(int kernelSize, int stride = -1, int padding = 0)
        {
            int s = (stride <= 0) ? kernelSize : stride;
            Initialize(
                kernelSize, kernelSize, kernelSize,
                s, s, s,
                padding, padding, padding
            );
        }
        
        public MaxPool3d(
            int kernelD, int kernelH, int kernelW, 
            int strideD, int strideH, int strideW, 
            int padD, int padH, int padW)
        {
            Initialize(
                kernelD, kernelH, kernelW, 
                strideD, strideH, strideW, 
                padD, padH, padW
            );
        }

        private void Initialize(
            int kernelD, int kernelH, int kernelW, 
            int strideD, int strideH, int strideW, 
            int padD, int padH, int padW)
        {
            _kernelD = kernelD;
            _kernelH = kernelH;
            _kernelW = kernelW;
            _strideD = strideD;
            _strideH = strideH;
            _strideW = strideW;
            _padD = padD;
            _padH = padH;
            _padW = padW;
        }
        
        public int[] GetOutputSize(params int[] inputSize)
        {
            if (inputSize.Length != 5)
            {
                throw new ArgumentException("Input must be a 5D tensor", nameof(inputSize));
            }
            
            int n = inputSize[0];
            int cOut = inputSize[1]; 
            int dIn = inputSize[2];
            int hIn = inputSize[3];
            int wIn = inputSize[4];

            int dOut = (dIn + 2 * _padD - _kernelD) / _strideD + 1;
            int hOut = (hIn + 2 * _padH - _kernelH) / _strideH + 1;
            int wOut = (wIn + 2 * _padW - _kernelW) / _strideW + 1;
            
            return new int[] { n, cOut, dOut, hOut, wOut };
        }

        public override TensorBox Forward(TensorBox inputTensor)
        {
            return TensorBox.MaxPool3d(
                inputTensor, 
                _kernelD, _kernelH, _kernelW, 
                _strideD, _strideH, _strideW, 
                _padD, _padH, _padW
            );
        }
    }

    /// <summary>
    /// 3D Average Pooling Layer Module (AvgPool3d).
    /// </summary>
    public class AvgPool3d : SingleInSingleOutModule
    {
        private int _kernelD, _kernelH, _kernelW;
        private int _strideD, _strideH, _strideW;
        private int _padD, _padH, _padW;

        public AvgPool3d(int kernelSize, int stride = -1, int padding = 0)
        {
            int s = (stride <= 0) ? kernelSize : stride;
            Initialize(
                kernelSize, kernelSize, kernelSize,
                s, s, s,
                padding, padding, padding
            );
        }
        
        public AvgPool3d(
            int kernelD, int kernelH, int kernelW, 
            int strideD, int strideH, int strideW, 
            int padD, int padH, int padW)
        {
            Initialize(
                kernelD, kernelH, kernelW, 
                strideD, strideH, strideW, 
                padD, padH, padW
            );
        }

        private void Initialize(
            int kernelD, int kernelH, int kernelW, 
            int strideD, int strideH, int strideW, 
            int padD, int padH, int padW)
        {
            _kernelD = kernelD;
            _kernelH = kernelH;
            _kernelW = kernelW;
            _strideD = strideD;
            _strideH = strideH;
            _strideW = strideW;
            _padD = padD;
            _padH = padH;
            _padW = padW;
        }

        public int[] GetOutputSize(params int[] inputSize)
        {
            if (inputSize.Length != 5)
            {
                throw new ArgumentException("Input must be a 5D tensor", nameof(inputSize));
            }
            
            int n = inputSize[0];
            int cOut = inputSize[1]; 
            int dIn = inputSize[2];
            int hIn = inputSize[3];
            int wIn = inputSize[4];

            int dOut = (dIn + 2 * _padD - _kernelD) / _strideD + 1;
            int hOut = (hIn + 2 * _padH - _kernelH) / _strideH + 1;
            int wOut = (wIn + 2 * _padW - _kernelW) / _strideW + 1;
            
            return new int[] { n, cOut, dOut, hOut, wOut };
        }

        public override TensorBox Forward(TensorBox inputTensor)
        {
            return TensorBox.AvgPool3d(
                inputTensor, 
                _kernelD, _kernelH, _kernelW, 
                _strideD, _strideH, _strideW, 
                _padD, _padH, _padW
            );
        }
    }
}
