namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    /// <summary>
    /// Automatic differentiation function for 3D Average Pooling.
    /// </summary>
    public class AvgPool3dFunction : KwargsFunction<AvgPool3dFunction, MaxPool3dFunction.Pool3dKwargs>
    {
        internal override Tensor Forward(MaxPool3dFunction.Pool3dKwargs kwargs, Context ctx)
        {
            var input = ctx.GetInput(0);
            (Tensor output, int[] counts) = Tensor.AveragePool3d(
                input,
                kwargs.KernelD, kwargs.KernelH, kwargs.KernelW,
                kwargs.StrideD, kwargs.StrideH, kwargs.StrideW,
                kwargs.PadD, kwargs.PadH, kwargs.PadW
            );

            ctx.RegisterDatas((object)counts);

            return output;
        }

        internal override Tensor[] Backward(MaxPool3dFunction.Pool3dKwargs kwargs, Context ctx, Tensor grad)
        {
            var input = ctx.GetInput(0);
            var counts = ctx.GetRegisteredData<int[]>(0);
            
            var gradInputData = new float[input.TotalSize];
            float[] gradData = grad.GetContiguousData();

            int kernelD = kwargs.KernelD;
            int kernelH = kwargs.KernelH;
            int kernelW = kwargs.KernelW;
            int strideD = kwargs.StrideD;
            int strideH = kwargs.StrideH;
            int strideW = kwargs.StrideW;
            int padD = kwargs.PadD;
            int padH = kwargs.PadH;
            int padW = kwargs.PadW;

            int batchSize = input.Size[0];
            int channels = input.Size[1];
            int inDepth = input.Size[2];
            int inHeight = input.Size[3];
            int inWidth = input.Size[4];
            int[] inputStrides = input.Strides;

            int outDepth = grad.Size[2];
            int outHeight = grad.Size[3];
            int outWidth = grad.Size[4];

            int outputIndex = 0;

            for (int n = 0; n < batchSize; n++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int od = 0; od < outDepth; od++)
                    {
                        for (int oh = 0; oh < outHeight; oh++)
                        {
                            for (int ow = 0; ow < outWidth; ow++)
                            {
                                int count = counts[outputIndex];
                                float gradVal = gradData[outputIndex];
                                float gradPerInput = 0.0f;

                                if (count > 0)
                                {
                                    gradPerInput = gradVal / count;
                                }
                                
                                if (gradPerInput != 0.0f) 
                                {
                                    int dStart = od * strideD - padD;
                                    int hStart = oh * strideH - padH;
                                    int wStart = ow * strideW - padW;

                                    for (int kd = 0; kd < kernelD; kd++)
                                    {
                                        int id = dStart + kd;
                                        for (int kh = 0; kh < kernelH; kh++)
                                        {
                                            int ih = hStart + kh;
                                            for (int kw = 0; kw < kernelW; kw++)
                                            {
                                                int iw = wStart + kw;
                                                
                                                if (id >= 0 && id < inDepth &&
                                                    ih >= 0 && ih < inHeight &&
                                                    iw >= 0 && iw < inWidth)
                                                {
                                                    int inputIndex = 
                                                        n * inputStrides[0] +
                                                        c * inputStrides[1] +
                                                        id * inputStrides[2] +
                                                        ih * inputStrides[3] +
                                                        iw * inputStrides[4];
                                                    
                                                    gradInputData[inputIndex] += gradPerInput;
                                                }
                                            }
                                        }
                                    }
                                }

                                outputIndex++;
                            }   
                        }
                    }
                }
            }

            return new[] { 
                new Tensor(gradInputData, input.Size, input.Strides) 
            };
        }
    }
}
