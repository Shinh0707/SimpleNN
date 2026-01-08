namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    /// <summary>
    /// Automatic differentiation function for 1D Max Pooling.
    /// </summary>
    public class MaxPool1dFunction : KwargsFunction<MaxPool1dFunction, MaxPool1dFunction.Pool1dKwargs>
    {
        public class Pool1dKwargs
        {
            public int KernelSize { get; set; }
            public int Stride { get; set; }
            public int PadLeft { get; set; }
            public int PadRight { get; set; }

            public Pool1dKwargs() : this(3, -1, 0)
            {
            }

            public Pool1dKwargs(int kernelSize, int stride = -1, int padding = 0)
            {
                KernelSize = kernelSize;
                
                int s = (stride <= 0) ? kernelSize : stride;
                Stride = s;
                PadLeft = padding;
                PadRight = padding;
            }

            public Pool1dKwargs(int kernelSize, int stride, int padLeft, int padRight)
            {
                KernelSize = kernelSize;
                Stride = stride;
                PadLeft = padLeft;
                PadRight = padRight;
            }
        }

        internal override Tensor Forward(Pool1dKwargs kwargs, Context ctx)
        {
            var input = ctx.GetInput(0);
            (Tensor output, int[] flatIndices) = Tensor.MaxPool1d(
                input,
                kwargs.KernelSize,
                kwargs.Stride,
                kwargs.PadLeft,
                kwargs.PadRight
            );
            ctx.RegisterDatas((object)flatIndices);

            return output;
        }

        internal override Tensor[] Backward(Pool1dKwargs kwargs, Context ctx, Tensor grad)
        {
            var input = ctx.GetInput(0);
            var flatIndices = ctx.GetRegisteredData<int[]>(0);
            var gradInputData = new float[input.TotalSize];
            float[] gradData = grad.GetContiguousData();
            for (int i = 0; i < flatIndices.Length; i++)
            {
                int inputIndex = flatIndices[i];
                if (inputIndex != -1) 
                {
                    gradInputData[inputIndex] += gradData[i];
                }
            }

            return new[] { new Tensor(gradInputData, input.Size, input.Strides) };
        }
    }
}
