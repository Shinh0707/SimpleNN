namespace SimpleNN.Graph
{
    using SimpleNN.Graph.Functions;
    public partial class TensorBox
    {
        public static TensorBox Conv3d(TensorBox input, TensorBox weight, int strideD, int strideH, int strideW, int padD, int padH, int padW, TensorBox bias = null)
        {
            if (bias is null)
            {
                return new(Conv3dFunction.Forward(new(
                    strideD, strideH, strideW, padD, padH, padW
                ), input._ctx, weight._ctx));
            }
            return new(Conv3dFunction.Forward(new(
                strideD, strideH, strideW, padD, padH, padW
            ), input._ctx, weight._ctx, bias._ctx));
        }
        public static TensorBox Conv3d(TensorBox input, TensorBox weight, int stride = 1, int pad = 0, TensorBox bias = null)
        {
            if (bias is null)
            {
                return new(Conv3dFunction.Forward(new(
                    stride, pad
                ), input._ctx, weight._ctx));
            }
            return new(Conv3dFunction.Forward(new(
                stride, pad
            ), input._ctx, weight._ctx, bias._ctx));
        }

        public static TensorBox TransposedConv3d(TensorBox input, TensorBox weight, int strideD, int strideH, int strideW, int padD, int padH, int padW, TensorBox bias = null)
        {
            if (bias is null)
            {
                return new(TransposedConv3dFunction.Forward(new(
                    strideD, strideH, strideW, padD, padH, padW
                ), input._ctx, weight._ctx));
            }
            return new(TransposedConv3dFunction.Forward(new(
                strideD, strideH, strideW, padD, padH, padW
            ), input._ctx, weight._ctx, bias._ctx));
        }
        public static TensorBox TransposedConv3d(TensorBox input, TensorBox weight, int stride = 1, int pad = 0, TensorBox bias = null)
        {
            if (bias is null)
            {
                return new(TransposedConv3dFunction.Forward(new(
                    stride, pad
                ), input._ctx, weight._ctx));
            }
            return new(TransposedConv3dFunction.Forward(new(
                stride, pad
            ), input._ctx, weight._ctx, bias._ctx));
        }

        public static TensorBox MaxPool3d(TensorBox input, int kernelD, int kernelH, int kernelW, int strideD, int strideH, int strideW, int padD, int padH, int padW)
        {
            return new(MaxPool3dFunction.Forward(new(
                kernelD, kernelH, kernelW, strideD, strideH, strideW, padD, padH, padW
            ), input._ctx));
        }
        public static TensorBox MaxPool3d(TensorBox input, int kernel, int stride = 1, int pad = 0)
        {
            return new(MaxPool3dFunction.Forward(new(
                kernel, stride, pad
            ), input._ctx));
        }

        public static TensorBox AvgPool3d(TensorBox input, int kernelD, int kernelH, int kernelW, int strideD, int strideH, int strideW, int padD, int padH, int padW)
        {
            return new(AvgPool3dFunction.Forward(new(
                kernelD, kernelH, kernelW, strideD, strideH, strideW, padD, padH, padW
            ), input._ctx));
        }
        public static TensorBox AvgPool3d(TensorBox input, int kernel, int stride = 1, int pad = 0)
        {
            return new(AvgPool3dFunction.Forward(new(
                kernel, stride, pad
            ), input._ctx));
        }
    }
}
