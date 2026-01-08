namespace SimpleNN.Tensor
{
    using System;

    /// <summary>
    /// This file defines convolution-related operations for the Tensor class (1D).
    /// </summary>
    public partial class Tensor
    {
        // --- public Conv1d (Overloads) ---

        /// <summary>
        /// Executes a 1D convolution operation.
        /// </summary>
        public static Tensor Conv1d(
            Tensor input, Tensor weight, Tensor bias = null,
            int stride = 1, int padding = 0)
        {
            return Conv1d(
                input, weight, bias,
                stride, padding, padding
            );
        }

        // --- public Conv1d (Core) ---

        /// <summary>
        /// Executes a 1D convolution operation.
        /// </summary>
        public static Tensor Conv1d(
            Tensor input, Tensor weight, Tensor bias,
            int stride, int padLeft, int padRight)
        {
            // 1. Get input and weight shapes
            if (input.NDim != 3)
                throw new ArgumentException("Input must be a 3D tensor (N, C, L)");
            if (weight.NDim != 3)
                throw new ArgumentException("Weight must be a 3D tensor (C_out, C_in, K)");

            int batchSize = input.Size[0];
            int inChannels = input.Size[1];
            int inLength = input.Size[2];

            int outChannels = weight.Size[0];
            int inChannelsW = weight.Size[1];
            int kernelSize = weight.Size[2];

            if (inChannels != inChannelsW)
            {
                throw new ArgumentException(
                    $"Input channels ({inChannels}) and weight " +
                    $"channels ({inChannelsW}) do not match."
                );
            }

            // 2. Calculate output size
            int outLength = (inLength + padLeft + padRight - kernelSize) / stride + 1;

            if (outLength <= 0)
            {
                throw new ArgumentException("Invalid convolution parameters");
            }

            // 3. Im2Col
            var inputCol = Im2Col1d(
                input, kernelSize, stride,
                padLeft, padRight, outLength
            );

            // 4. Flatten weights
            int weightFlatSize = inChannels * kernelSize;
            var kernelFlat = Reshape(weight, new int[] { outChannels, weightFlatSize });

            // 5. MatMul
            var output = MatMul(kernelFlat, inputCol);

            // 6. Reshape output
            output = Reshape(
                output,
                new int[] { outChannels, batchSize, outLength }
            );

            // 7. Transpose
            output = Transpose(output, 0, 1);

            // 8. Add bias
            if (bias is not null)
            {
                if (bias.NDim != 1 || bias.Size[0] != outChannels)
                {
                    throw new ArgumentException("Bias shape mismatch");
                }
                var biasView = Reshape(bias, new int[] { 1, outChannels, 1 });
                output += biasView;
            }

            return output;
        }

        // --- private Im2Col Helper ---

        public static Tensor Im2Col1d(
            Tensor input, int kernelSize,
            int stride, int padLeft, int padRight,
            int outL)
        {
            int batchSize = input.Size[0];
            int inChannels = input.Size[1];
            int inLength = input.Size[2];

            float[] inputData = input._data;
            int[] inputStrides = input.Strides;

            int colRows = inChannels * kernelSize;
            int colCols = batchSize * outL;

            float[] outputData = new float[colRows * colCols];

            int outputColIndex = 0;

            for (int n = 0; n < batchSize; n++)
            {
                for (int ol = 0; ol < outL; ol++)
                {
                    int outputRowIndex = 0;

                    for (int c = 0; c < inChannels; c++)
                    {
                        for (int k = 0; k < kernelSize; k++)
                        {
                            int currentInL = ol * stride + k - padLeft;

                            float value = 0.0f;

                            if (currentInL >= 0 && currentInL < inLength)
                            {
                                int inputIndex = n * inputStrides[0] +
                                                 c * inputStrides[1] +
                                                 currentInL * inputStrides[2];
                                value = inputData[inputIndex];
                            }

                            int outputIndex = outputRowIndex * colCols + outputColIndex;
                            outputData[outputIndex] = value;

                            outputRowIndex++;
                        }
                    }

                    outputColIndex++;
                }
            }

            var outputSize = new int[] { colRows, colCols };
            return new Tensor(outputData, outputSize);
        }

        // --- public Conv1dTranspose (Core) ---

        public static Tensor Conv1dTranspose(
            Tensor input, Tensor weight, Tensor bias,
            int stride, int padLeft, int padRight,
            int outLength
        )
        {
            if (input.NDim != 3)
                throw new ArgumentException("Input must be a 3D tensor");
            if (weight.NDim != 3)
                throw new ArgumentException("Weight must be a 3D tensor");

            int batchSize = input.Size[0];
            int inChannels = input.Size[1];
            int inLength = input.Size[2];

            int inChannelsW = weight.Size[0];
            int outChannels = weight.Size[1];
            int kernelSize = weight.Size[2];

            if (inChannels != inChannelsW)
            {
                throw new ArgumentException(
                    $"Input channels ({inChannels}) and weight " +
                    $"channels ({inChannelsW}) do not match."
                );
            }

            if (outLength <= 0)
            {
                throw new ArgumentException("Invalid transpose conv parameters");
            }

            int weightFlatSize = outChannels * kernelSize;
            var weightFlat = Reshape(weight, new int[] { inChannels, weightFlatSize });

            var weightT = Transpose(weightFlat, 0, 1);
            var inputT = Transpose(input, 0, 1);

            var inputFlat = Reshape(
                inputT, 
                new int[] { inChannels, batchSize * inLength }
            );

            var cols = MatMul(weightT, inputFlat);

            var output = Col2Im1d(
                cols,
                batchSize, outChannels, outLength,
                kernelSize, stride, padLeft, padRight,
                inLength
            );
            
            if (bias is not null)
            {
                if (bias.NDim != 1 || bias.Size[0] != outChannels)
                {
                    throw new ArgumentException("Bias shape mismatch");
                }
                var biasView = Reshape(bias, new int[] { 1, outChannels, 1 });
                output += biasView;
            }

            return output;
        }

        public static Tensor Conv1dTranspose(
            Tensor input, Tensor weight, Tensor bias,
            int stride, int padLeft, int padRight
        )
        {
            int inLength = input.Size[2];
            int kernelSize = weight.Size[2];

            int outLength = (inLength - 1) * stride - padLeft - padRight + kernelSize;

            return Conv1dTranspose(
                input, weight, bias,
                stride, padLeft, padRight,
                outLength
            );
        }

        // --- private Col2Im Helper ---

        public static Tensor Col2Im1d(
            Tensor cols,
            int N, int C, int L,
            int kernelSize,
            int stride,
            int padLeft, int padRight,
            int L_in)
        {
            float[] colsData = cols._data;
            int colCols = cols.Size[1];

            float[] outputData = new float[N * C * L];

            for (int n = 0; n < N; n++)
            {
                for (int il = 0; il < L_in; il++)
                {
                    for (int c = 0; c < C; c++)
                    {
                        for (int k = 0; k < kernelSize; k++)
                        {
                            int currentOutL = il * stride + k - padLeft;

                            if (currentOutL >= 0 && currentOutL < L)
                            {
                                int colRow = (c * kernelSize + k);
                                int colCol = (n * L_in + il);
                                int colsIndex = colRow * colCols + colCol;

                                int outputIndex =
                                    n * (C * L) +
                                    c * (L) +
                                    currentOutL;

                                outputData[outputIndex] += colsData[colsIndex];
                            }
                        }
                    }
                }
            }

            return new Tensor(outputData, new int[] { N, C, L });
        }
    }
}
