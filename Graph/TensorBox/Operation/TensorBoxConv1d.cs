namespace SimpleNN.Graph
{
    using SimpleNN.Graph.Functions;
    public partial class TensorBox
    {
        public static TensorBox Conv1d(TensorBox input, TensorBox weight, int stride, int padLeft, int padRight, TensorBox bias = null)
        {
            if (bias is null)
            {
                return new(Conv1dFunction.Forward(new(
                    stride, padLeft, padRight
                ), input._ctx, weight._ctx));
            }
            return new(Conv1dFunction.Forward(new(
                stride, padLeft, padRight
            ), input._ctx, weight._ctx, bias._ctx));
        }
        public static TensorBox Conv1d(TensorBox input, TensorBox weight, int stride = 1, int pad = 0, TensorBox bias = null)
        {
            if (bias is null)
            {
                return new(Conv1dFunction.Forward(new(
                    stride, pad
                ), input._ctx, weight._ctx));
            }
            return new(Conv1dFunction.Forward(new(
                stride, pad
            ), input._ctx, weight._ctx, bias._ctx));
        }

        public static TensorBox TransposedConv1d(TensorBox input, TensorBox weight, int stride, int padLeft, int padRight, TensorBox bias = null)
        {
            if (bias is null)
            {
                return new(TransposedConv1dFunction.Forward(new(
                    stride, padLeft, padRight
                ), input._ctx, weight._ctx));
            }
            return new(TransposedConv1dFunction.Forward(new(
                stride, padLeft, padRight
            ), input._ctx, weight._ctx, bias._ctx));
        }
        public static TensorBox TransposedConv1d(TensorBox input, TensorBox weight, int stride = 1, int pad = 0, TensorBox bias = null)
        {
            if (bias is null)
            {
                return new(TransposedConv1dFunction.Forward(new(
                    stride, pad
                ), input._ctx, weight._ctx));
            }
            return new(TransposedConv1dFunction.Forward(new(
                stride, pad
            ), input._ctx, weight._ctx, bias._ctx));
        }

        public static TensorBox MaxPool1d(TensorBox input, int kernelSize, int stride, int padLeft, int padRight)
        {
            return new(MaxPool1dFunction.Forward(new(
                kernelSize, stride, padLeft, padRight
            ), input._ctx));
        }
        public static TensorBox MaxPool1d(TensorBox input, int kernelSize, int stride = 1, int pad = 0)
        {
            return new(MaxPool1dFunction.Forward(new(
                kernelSize, stride, pad
            ), input._ctx));
        }

        public static TensorBox AvgPool1d(TensorBox input, int kernelSize, int stride, int padLeft, int padRight)
        {
            return new(AvgPool1dFunction.Forward(new(
                kernelSize, stride, padLeft, padRight
            ), input._ctx));
        }
        public static TensorBox AvgPool1d(TensorBox input, int kernelSize, int stride = 1, int pad = 0)
        {
            return new(AvgPool1dFunction.Forward(new(
                kernelSize, stride, pad
            ), input._ctx));
        }
    }
}
