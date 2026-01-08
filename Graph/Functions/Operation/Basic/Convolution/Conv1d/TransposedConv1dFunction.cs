namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;
    /// <summary>
    /// Automatic differentiation function for 1D transposed convolution.
    /// </summary>
    public class TransposedConv1dFunction : KwargsFunction<TransposedConv1dFunction, Conv1dFunction.Kwargs>
    {
        internal override Tensor Forward(Conv1dFunction.Kwargs kwargs, Context ctx)
        {
            var input = ctx.GetInput(0);
            var weight = ctx.GetInput(1); // (C_in, C_out, K_l)
            ctx.TryGetInput(2, out Tensor bias);

            return Tensor.Conv1dTranspose(
                input, weight, bias,
                kwargs.Stride, kwargs.PadLeft, kwargs.PadRight
            );
        }

        internal override Tensor[] Backward(Conv1dFunction.Kwargs kwargs, Context ctx, Tensor grad)
        {
            var input = ctx.GetInput(0);  // (N, C_in, L_in)
            var weight = ctx.GetInput(1); // (C_in, C_out, K_l)
            
            int n = input.Size[0];
            int cIn = input.Size[1];
            int lIn = input.Size[2];
            
            int cOut = weight.Size[1];
            int kernelSize = weight.Size[2];

            int lOut = grad.Size[2];
            int patchCount = n * lIn; // Input patches

            // --- 1. gradInput (dL/dInput) ---
            var weightT = Tensor.Transpose(weight, 0, 1);
            
            var gradInput = Tensor.Conv1d(
                grad, weightT, null,
                kwargs.Stride, kwargs.PadLeft, kwargs.PadRight
            );

            // --- 2. gradWeight (dL/dWeight) ---
            var gradCol = Tensor.Im2Col1d(
                grad, kernelSize,
                kwargs.Stride, kwargs.PadLeft, kwargs.PadRight, 
                lIn
            );

            var inputT = Tensor.Transpose(input, 1, 0); // (C_in, N, L_in)
            var inputFlat = Tensor.Reshape(
                inputT, 
                new int[] { cIn, patchCount }
            );

            var gradWeightFlat = Tensor.MatMul(
                inputFlat, 
                Tensor.Transpose(gradCol, 0, 1)
            );
            
            var gradWeight = Tensor.Reshape(gradWeightFlat, weight.Size);

            // --- 3. gradBias (dL/dBias) ---
            if (ctx.TryGetInput(2, out Tensor _))
            {
                var gradT_bias = Tensor.Transpose(grad, 1, 0); // (C_out, N, L)
                var gradFlat_bias = Tensor.Reshape(
                    gradT_bias, 
                    new int[] { cOut, n * lOut }
                );
                
                var ones = Tensor.Ones(new[] { n * lOut, 1 });
                var gradBias = Tensor.MatMul(gradFlat_bias, ones);
                gradBias = Tensor.Reshape(gradBias, new[] { cOut });

                return new[] { gradInput, gradWeight, gradBias };
            }
            else
            {
                return new[] { gradInput, gradWeight };
            }
        }
    }
}
