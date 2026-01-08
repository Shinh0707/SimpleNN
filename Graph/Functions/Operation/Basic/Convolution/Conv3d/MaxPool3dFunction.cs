namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    /// <summary>
    /// Automatic differentiation function for 3D Max Pooling.
    /// </summary>
    public class MaxPool3dFunction : KwargsFunction<MaxPool3dFunction, MaxPool3dFunction.Pool3dKwargs>
    {
        public class Pool3dKwargs
        {
            public int KernelD { get; set; }
            public int KernelH { get; set; }
            public int KernelW { get; set; }
            public int StrideD { get; set; }
            public int StrideH { get; set; }
            public int StrideW { get; set; }
            public int PadD { get; set; }
            public int PadH { get; set; }
            public int PadW { get; set; }

            public Pool3dKwargs() : this(3, -1, 0)
            {
            }

            public Pool3dKwargs(int kernelSize, int stride = -1, int padding = 0)
            {
                KernelD = kernelSize;
                KernelH = kernelSize;
                KernelW = kernelSize;
                
                int s = (stride <= 0) ? kernelSize : stride;
                StrideD = s;
                StrideH = s;
                StrideW = s;
                PadD = padding;
                PadH = padding;
                PadW = padding;
            }

            public Pool3dKwargs(
                int kernelD, int kernelH, int kernelW,
                int strideD, int strideH, int strideW, 
                int padD, int padH, int padW)
            {
                KernelD = kernelD;
                KernelH = kernelH;
                KernelW = kernelW;
                StrideD = strideD;
                StrideH = strideH;
                StrideW = strideW;
                PadD = padD;
                PadH = padH;
                PadW = padW;
            }
        }

        internal override Tensor Forward(Pool3dKwargs kwargs, Context ctx)
        {
            var input = ctx.GetInput(0);
            (Tensor output, int[] flatIndices) = Tensor.MaxPool3d(
                input,
                kwargs.KernelD, kwargs.KernelH, kwargs.KernelW,
                kwargs.StrideD, kwargs.StrideH, kwargs.StrideW,
                kwargs.PadD, kwargs.PadH, kwargs.PadW
            );
            ctx.RegisterDatas((object)flatIndices);

            return output;
        }

        internal override Tensor[] Backward(Pool3dKwargs kwargs, Context ctx, Tensor grad)
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
