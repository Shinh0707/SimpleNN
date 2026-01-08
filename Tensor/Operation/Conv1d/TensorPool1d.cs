namespace SimpleNN.Tensor
{
    using System;
    public partial class Tensor
    {
        // --- Pool1d (Overloads) ---

        private static (Tensor, int[]) Pool1d(
            Tensor input,
            PoolInitFunc initFunc,
            PoolAccumulateFunc accumulateFunc,
            PoolFinalizeFunc finalizeFunc,
            int kernelSize,
            int stride = -1,
            int padding = 0)
        {
            int s = (stride <= 0) ? kernelSize : stride;
            return Pool1d(
                input, initFunc, accumulateFunc, finalizeFunc,
                kernelSize, s, padding, padding
            );
        }

        // --- Pool1d (Core) ---

        /// <summary>
        /// Executes a 1D pooling operation (generic core).
        /// </summary>
        private static (Tensor, int[]) Pool1d(
            Tensor input,
            PoolInitFunc initFunc,
            PoolAccumulateFunc accumulateFunc,
            PoolFinalizeFunc finalizeFunc,
            int kernelSize,
            int stride,
            int padLeft,
            int padRight)
        {
            // 1. Get input shape
            if (input.NDim != 3)
                throw new ArgumentException("Input must be a 3D tensor");

            int batchSize = input.Size[0];
            int channels = input.Size[1];
            int inLength = input.Size[2];

            float[] inputData = input._data;
            int[] inputStrides = input.Strides;

            // 2. Calculate output size
            int outLength = (inLength + padLeft + padRight - kernelSize) / stride + 1;

            if (outLength <= 0)
                throw new ArgumentException("Invalid pooling parameters");

            // 3. Initialize output array
            int outTotalSize = batchSize * channels * outLength;
            float[] outputData = new float[outTotalSize];
            int[] flatIndices = new int[outTotalSize];
            int outputIndex = 0;

            // 4. Execute pooling
            for (int n = 0; n < batchSize; n++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int ol = 0; ol < outLength; ol++)
                    {
                        // (1) Initialize
                        var currentState = initFunc();
                        int windowElementCount = 0;

                        int lStart = ol * stride - padLeft;

                        for (int k = 0; k < kernelSize; k++)
                        {
                            int il = lStart + k;

                            if (il >= 0 && il < inLength)
                            {
                                int inputIndex = n * inputStrides[0] +
                                                 c * inputStrides[1] +
                                                 il * inputStrides[2];
                                float val = inputData[inputIndex];

                                // (2) Accumulate
                                currentState = accumulateFunc(
                                    val, inputIndex, currentState
                                );
                                windowElementCount++;
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

            var outputSize = new int[] {
                batchSize, channels, outLength
            };
            var outputTensor = new Tensor(outputData, outputSize);

            return (outputTensor, flatIndices);
        }
    }
}
