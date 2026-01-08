namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    /// <summary>
    /// Automatic differentiation function for 1D Average Pooling.
    /// </summary>
    public class AvgPool1dFunction : KwargsFunction<AvgPool1dFunction, MaxPool1dFunction.Pool1dKwargs>
    {
        internal override Tensor Forward(MaxPool1dFunction.Pool1dKwargs kwargs, Context ctx)
        {
            var input = ctx.GetInput(0);
            (Tensor output, int[] counts) = Tensor.AveragePool1d(
                input,
                kwargs.KernelSize,
                kwargs.Stride,
                kwargs.PadLeft,
                kwargs.PadRight
            );

            ctx.RegisterDatas((object)counts);

            return output;
        }

        internal override Tensor[] Backward(MaxPool1dFunction.Pool1dKwargs kwargs, Context ctx, Tensor grad)
        {
            var input = ctx.GetInput(0);
            var counts = ctx.GetRegisteredData<int[]>(0);
            
            var gradInputData = new float[input.TotalSize];
            float[] gradData = grad.GetContiguousData();

            int kernelSize = kwargs.KernelSize;
            int stride = kwargs.Stride;
            int padLeft = kwargs.PadLeft;
            int padRight = kwargs.PadRight; // Not used directly in loop logic but good for completeness if needed later

            int batchSize = input.Size[0];
            int channels = input.Size[1];
            int inLength = input.Size[2];
            int[] inputStrides = input.Strides;

            int outLength = grad.Size[2];

            int outputIndex = 0;

            for (int n = 0; n < batchSize; n++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int ol = 0; ol < outLength; ol++)
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
                            int lStart = ol * stride - padLeft;

                            for (int k = 0; k < kernelSize; k++)
                            {
                                int il = lStart + k;

                                if (il >= 0 && il < inLength)
                                {
                                    int inputIndex = 
                                        n * inputStrides[0] +
                                        c * inputStrides[1] +
                                        il * inputStrides[2];
                                    
                                    gradInputData[inputIndex] += gradPerInput;
                                }
                            }
                        }

                        outputIndex++;
                    }
                }
            }

            return new[] { 
                new Tensor(gradInputData, input.Size, input.Strides) 
            };
        }
    }
}
