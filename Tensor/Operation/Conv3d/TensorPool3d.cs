namespace SimpleNN.Tensor
{
    using System;
    public partial class Tensor
    {
        // --- Pool3d (Overloads) ---

        private static (Tensor, int[]) Pool3d(
            Tensor input,
            PoolInitFunc initFunc,
            PoolAccumulateFunc accumulateFunc,
            PoolFinalizeFunc finalizeFunc,
            int kernelSize,
            int stride = -1,
            int padding = 0)
        {
            int s = (stride <= 0) ? kernelSize : stride;
            return Pool3d(
                input, initFunc, accumulateFunc, finalizeFunc,
                kernelSize, kernelSize, kernelSize,
                s, s, s,
                padding, padding, padding
            );
        }

        // --- Pool3d (Core) ---

        /// <summary>
        /// Executes a 3D pooling operation (generic core).
        /// </summary>
        private static (Tensor, int[]) Pool3d(
            Tensor input,
            PoolInitFunc initFunc,
            PoolAccumulateFunc accumulateFunc,
            PoolFinalizeFunc finalizeFunc,
            int kernelD, int kernelH, int kernelW,
            int strideD, int strideH, int strideW,
            int padD, int padH, int padW)
        {
            // 1. Get input shape
            if (input.NDim != 5)
                throw new ArgumentException("Input must be a 5D tensor");

            int batchSize = input.Size[0];
            int channels = input.Size[1];
            int inDepth = input.Size[2];
            int inHeight = input.Size[3];
            int inWidth = input.Size[4];

            float[] inputData = input._data;
            int[] inputStrides = input.Strides;

            // 2. Calculate output size
            int outDepth = (inDepth + 2 * padD - kernelD) / strideD + 1;
            int outHeight = (inHeight + 2 * padH - kernelH) / strideH + 1;
            int outWidth = (inWidth + 2 * padW - kernelW) / strideW + 1;

            if (outDepth <= 0 || outHeight <= 0 || outWidth <= 0)
                throw new ArgumentException("Invalid pooling parameters");

            // 3. Initialize output array
            int outTotalSize = batchSize * channels * outDepth * outHeight * outWidth;
            float[] outputData = new float[outTotalSize];
            int[] flatIndices = new int[outTotalSize];
            int outputIndex = 0;

            // 4. Execute pooling
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
                                // (1) Initialize
                                var currentState = initFunc();
                                int windowElementCount = 0;

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
                                                int inputIndex = n * inputStrides[0] +
                                                                 c * inputStrides[1] +
                                                                 id * inputStrides[2] +
                                                                 ih * inputStrides[3] +
                                                                 iw * inputStrides[4];
                                                float val = inputData[inputIndex];

                                                // (2) Accumulate
                                                currentState = accumulateFunc(
                                                    val, inputIndex, currentState
                                                );
                                                windowElementCount++;
                                            }
                                        }
                                    }
                                }

                                // (3) Finalize
                                var finalResult = finalizeFunc(
                                    currentState, windowElementCount
                                );

                                outputData[outputIndex] = finalResult.val;
                                flatIndices[outputIndex] = finalResult.ptr;
                                outputIndex++;
                            }
                        }
                    }
                }
            }

            var outputSize = new int[] {
                batchSize, channels, outDepth, outHeight, outWidth
            };
            var outputTensor = new Tensor(outputData, outputSize);

            return (outputTensor, flatIndices);
        }
    }
}
