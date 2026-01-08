namespace SimpleNN.Tensor
{
    using System;

    /// <summary>
    /// This file defines convolution-related operations for the Tensor class (3D).
    /// </summary>
    public partial class Tensor
    {
        // --- public Conv3d (Overloads) ---

        /// <summary>
        /// Executes a 3D convolution operation.
        /// </summary>
        public static Tensor Conv3d(
            Tensor input, Tensor weight, Tensor bias = null,
            int stride = 1, int padding = 0)
        {
            return Conv3d(
                input, weight, bias,
                stride, stride, stride, 
                padding, padding, padding
            );
        }

        // --- public Conv3d (Core) ---

        /// <summary>
        /// Executes a 3D convolution operation (using im2col).
        /// </summary>
        public static Tensor Conv3d(
            Tensor input, Tensor weight, Tensor bias,
            int strideD, int strideH, int strideW, 
            int padD, int padH, int padW)
        {
            // 1. Get input and weight shapes
            if (input.NDim != 5)
                throw new ArgumentException("Input must be a 5D tensor (N, C, D, H, W)");
            if (weight.NDim != 5)
                throw new ArgumentException("Weight must be a 5D tensor (C_out, C_in, K_d, K_h, K_w)");

            int batchSize = input.Size[0];
            int inChannels = input.Size[1];
            int inDepth = input.Size[2];
            int inHeight = input.Size[3];
            int inWidth = input.Size[4];

            int outChannels = weight.Size[0];
            int inChannelsW = weight.Size[1];
            int kernelD = weight.Size[2];
            int kernelH = weight.Size[3];
            int kernelW = weight.Size[4];

            if (inChannels != inChannelsW)
            {
                throw new ArgumentException(
                    $"Input channels ({inChannels}) and weight " +
                    $"channels ({inChannelsW}) do not match."
                );
            }

            // 2. Calculate output size
            int outDepth = (inDepth + 2 * padD - kernelD) / strideD + 1;
            int outHeight = (inHeight + 2 * padH - kernelH) / strideH + 1;
            int outWidth = (inWidth + 2 * padW - kernelW) / strideW + 1;

            if (outDepth <= 0 || outHeight <= 0 || outWidth <= 0)
            {
                throw new ArgumentException("Invalid convolution parameters");
            }

            // 3. Im2Col: Reshape input patches into a matrix
            // (N, C_in, D_in, H_in, W_in) -> (C_in * K_d * K_h * K_w, N * D_out * H_out * W_out)
            var inputCol = Im2Col3d(
                input, kernelD, kernelH, kernelW, 
                strideD, strideH, strideW,
                padD, padH, padW, 
                outDepth, outHeight, outWidth
            );

            // 4. Flatten weights
            // (C_out, C_in, K_d, K_h, K_w) -> (C_out, C_in * K_d * K_h * K_w)
            int kernelSize = inChannels * kernelD * kernelH * kernelW;
            var kernelFlat = Reshape(weight, new int[] { outChannels, kernelSize });

            // 5. Compute convolution via matrix multiplication
            // MatMul( (C_out, K_size), (K_size, N * D * H * W) )
            // -> (C_out, N * D * H * W)
            var output = MatMul(kernelFlat, inputCol);

            // 6. Reshape output to (C_out, N, D_out, H_out, W_out)
            output = Reshape(
                output,
                new int[] { outChannels, batchSize, outDepth, outHeight, outWidth }
            );

            // 7. Transpose to (N, C_out, D_out, H_out, W_out)
            output = Transpose(output, 0, 1);

            // 8. Add bias (using broadcast)
            if (bias is not null)
            {
                if (bias.NDim != 1 || bias.Size[0] != outChannels)
                {
                    throw new ArgumentException("Bias shape mismatch");
                }
                // (C_out) -> (1, C_out, 1, 1, 1)
                var biasView = Reshape(bias, new int[] { 1, outChannels, 1, 1, 1 });
                output += biasView;
            }

            return output;
        }

        // --- private Im2Col Helper ---

        public static Tensor Im2Col3d(
            Tensor input, 
            int kernelD, int kernelH, int kernelW,
            int strideD, int strideH, int strideW,
            int padD, int padH, int padW,
            int outD, int outH, int outW)
        {
            int batchSize = input.Size[0];
            int inChannels = input.Size[1];
            int inDepth = input.Size[2];
            int inHeight = input.Size[3];
            int inWidth = input.Size[4];

            float[] inputData = input._data;
            int[] inputStrides = input.Strides;

            int colRows = inChannels * kernelD * kernelH * kernelW;
            int colCols = batchSize * outD * outH * outW;

            float[] outputData = new float[colRows * colCols];

            int outputColIndex = 0;

            for (int n = 0; n < batchSize; n++)
            {
                for (int od = 0; od < outD; od++)
                {
                    for (int oh = 0; oh < outH; oh++)
                    {
                        for (int ow = 0; ow < outW; ow++)
                        {
                            int outputRowIndex = 0;

                            for (int c = 0; c < inChannels; c++)
                            {
                                for (int kd = 0; kd < kernelD; kd++)
                                {
                                    for (int kh = 0; kh < kernelH; kh++)
                                    {
                                        for (int kw = 0; kw < kernelW; kw++)
                                        {
                                            int currentInD = od * strideD + kd - padD;
                                            int currentInH = oh * strideH + kh - padH;
                                            int currentInW = ow * strideW + kw - padW;

                                            float value = 0.0f;

                                            if (currentInD >= 0 && currentInD < inDepth &&
                                                currentInH >= 0 && currentInH < inHeight &&
                                                currentInW >= 0 && currentInW < inWidth)
                                            {
                                                int inputIndex = n * inputStrides[0] +
                                                                 c * inputStrides[1] +
                                                                 currentInD * inputStrides[2] +
                                                                 currentInH * inputStrides[3] +
                                                                 currentInW * inputStrides[4];
                                                value = inputData[inputIndex];
                                            }

                                            int outputIndex = outputRowIndex * colCols + outputColIndex;
                                            outputData[outputIndex] = value;

                                            outputRowIndex++;
                                        }
                                    }
                                }
                            }

                            outputColIndex++;
                        }
                    }
                }
            }

            var outputSize = new int[] { colRows, colCols };
            return new Tensor(outputData, outputSize);
        }

        // --- public Conv3dTranspose (Core) ---

        public static Tensor Conv3dTranspose(
            Tensor input, Tensor weight, Tensor bias,
            int strideD, int strideH, int strideW,
            int padD, int padH, int padW,
            int outDepth, int outHeight, int outWidth
        )
        {
            if (input.NDim != 5)
                throw new ArgumentException("Input must be a 5D tensor");
            if (weight.NDim != 5)
                throw new ArgumentException("Weight must be a 5D tensor");

            int batchSize = input.Size[0];
            int inChannels = input.Size[1];
            int inDepth = input.Size[2];
            int inHeight = input.Size[3];
            int inWidth = input.Size[4];

            int inChannelsW = weight.Size[0];
            int outChannels = weight.Size[1];
            int kernelD = weight.Size[2];
            int kernelH = weight.Size[3];
            int kernelW = weight.Size[4];

            if (inChannels != inChannelsW)
            {
                throw new ArgumentException(
                    $"Input channels ({inChannels}) and weight " +
                    $"channels ({inChannelsW}) do not match."
                );
            }

            if (outDepth <= 0 || outHeight <= 0 || outWidth <= 0)
            {
                throw new ArgumentException("Invalid transpose conv parameters");
            }

            int kernelSize = outChannels * kernelD * kernelH * kernelW;
            var weightFlat = Reshape(weight, new int[] { inChannels, kernelSize });

            var weightT = Transpose(weightFlat, 0, 1);
            var inputT = Transpose(input, 0, 1);

            var inputFlat = Reshape(
                inputT, 
                new int[] { inChannels, batchSize * inDepth * inHeight * inWidth }
            );

            var cols = MatMul(weightT, inputFlat);

            var output = Col2Im3d(
                cols,
                batchSize, outChannels, 
                outDepth, outHeight, outWidth,
                kernelD, kernelH, kernelW,
                strideD, strideH, strideW,
                padD, padH, padW,
                inDepth, inHeight, inWidth
            );
            
            if (bias is not null)
            {
                if (bias.NDim != 1 || bias.Size[0] != outChannels)
                {
                    throw new ArgumentException("Bias shape mismatch");
                }
                var biasView = Reshape(bias, new int[] { 1, outChannels, 1, 1, 1 });
                output += biasView;
            }

            return output;
        }

        public static Tensor Conv3dTranspose(
            Tensor input, Tensor weight, Tensor bias,
            int strideD, int strideH, int strideW, 
            int padD, int padH, int padW
        )
        {
            int inDepth = input.Size[2];
            int inHeight = input.Size[3];
            int inWidth = input.Size[4];
            int kernelD = weight.Size[2];
            int kernelH = weight.Size[3];
            int kernelW = weight.Size[4];

            int outDepth = (inDepth - 1) * strideD - 2 * padD + kernelD;
            int outHeight = (inHeight - 1) * strideH - 2 * padH + kernelH;
            int outWidth = (inWidth - 1) * strideW - 2 * padW + kernelW;

            return Conv3dTranspose(
                input, weight, bias,
                strideD, strideH, strideW, 
                padD, padH, padW,
                outDepth, outHeight, outWidth
            );
        }

        // --- private Col2Im Helper ---

        public static Tensor Col2Im3d(
            Tensor cols,
            int N, int C, 
            int D, int H, int W,
            int kernelD, int kernelH, int kernelW,
            int strideD, int strideH, int strideW,
            int padD, int padH, int padW,
            int D_in, int H_in, int W_in)
        {
            float[] colsData = cols._data;
            int colCols = cols.Size[1];

            float[] outputData = new float[N * C * D * H * W];

            for (int n = 0; n < N; n++)
            {
                for (int id = 0; id < D_in; id++)
                {
                    for (int ih = 0; ih < H_in; ih++)
                    {
                        for (int iw = 0; iw < W_in; iw++)
                        {
                            for (int c = 0; c < C; c++)
                            {
                                for (int kd = 0; kd < kernelD; kd++)
                                {
                                    for (int kh = 0; kh < kernelH; kh++)
                                    {
                                        for (int kw = 0; kw < kernelW; kw++)
                                        {
                                            int currentOutD = id * strideD + kd - padD;
                                            int currentOutH = ih * strideH + kh - padH;
                                            int currentOutW = iw * strideW + kw - padW;

                                            if (currentOutD >= 0 && currentOutD < D &&
                                                currentOutH >= 0 && currentOutH < H &&
                                                currentOutW >= 0 && currentOutW < W)
                                            {
                                                int colRow = ((c * kernelD + kd) * kernelH + kh) * kernelW + kw;
                                                int colCol = ((n * D_in + id) * H_in + ih) * W_in + iw;
                                                int colsIndex = colRow * colCols + colCol;

                                                int outputIndex =
                                                    n * (C * D * H * W) +
                                                    c * (D * H * W) +
                                                    currentOutD * (H * W) +
                                                    currentOutH * W +
                                                    currentOutW;

                                                outputData[outputIndex] += colsData[colsIndex];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return new Tensor(outputData, new int[] { N, C, D, H, W });
        }
    }
}
