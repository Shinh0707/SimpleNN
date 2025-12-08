namespace SimpleNN.Graph
{
    using SimpleNN.Graph.Functions;
    public partial class TensorBox
    {
        public static TensorBox Conv2d(TensorBox input, TensorBox weight, int strideH, int strideW, int padH, int padW, TensorBox bias = null)
        {
            if (bias is null)
            {
                return new(Conv2dFunction.Forward(new(
                    strideH, strideW, padH, padW
                ), input._ctx, weight._ctx));
            }
            return new(Conv2dFunction.Forward(new(
                strideH, strideW, padH, padW
            ), input._ctx, weight._ctx, bias._ctx));
        }
        public static TensorBox Conv2d(TensorBox input, TensorBox weight, int stride = 1, int pad = 0, TensorBox bias = null)
        {
            if (bias is null)
            {
                return new(Conv2dFunction.Forward(new(
                    stride, pad
                ), input._ctx, weight._ctx));
            }
            return new(Conv2dFunction.Forward(new(
                stride, pad
            ), input._ctx, weight._ctx, bias._ctx));
        }
        public static TensorBox TransposedConv2d(TensorBox input, TensorBox weight, int strideH, int strideW, int padH, int padW, TensorBox bias = null)
        {
            if (bias is null)
            {
                return new(TransposedConv2dFunction.Forward(new(
                    strideH, strideW, padH, padW
                ), input._ctx, weight._ctx));
            }
            return new(TransposedConv2dFunction.Forward(new(
                strideH, strideW, padH, padW
            ), input._ctx, weight._ctx, bias._ctx));
        }
        public static TensorBox TransposedConv2d(TensorBox input, TensorBox weight, int stride = 1, int pad = 0, TensorBox bias = null)
        {
            if (bias is null)
            {
                return new(TransposedConv2dFunction.Forward(new(
                    stride, pad
                ), input._ctx, weight._ctx));
            }
            return new(TransposedConv2dFunction.Forward(new(
                stride, pad
            ), input._ctx, weight._ctx, bias._ctx));
        }
        public static TensorBox MaxPool2d(TensorBox input, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
        {
            return new(MaxPool2dFunction.Forward(new(
                kernelH, kernelW, strideH, strideW, padH, padW
            ), input._ctx));
        }
        public static TensorBox MaxPool2d(TensorBox input, int kernel, int stride = 1, int pad = 0)
        {
            return new(MaxPool2dFunction.Forward(new(
                kernel, stride, pad
            ), input._ctx));
        }
        public static TensorBox AvgPool2d(TensorBox input, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
        {
            return new(AvgPool2dFunction.Forward(new(
                kernelH, kernelW, strideH, strideW, padH, padW
            ), input._ctx));
        }
        public static TensorBox AvgPool2d(TensorBox input, int kernel, int stride = 1, int pad = 0)
        {
            return new(AvgPool2dFunction.Forward(new(
                kernel, stride, pad
            ), input._ctx));
        }
    }
}