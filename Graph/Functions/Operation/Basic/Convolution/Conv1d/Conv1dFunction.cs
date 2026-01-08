namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;
    using System;

    /// <summary>
    /// Automatic differentiation function for 1D convolution.
    /// </summary>
    public class Conv1dFunction : KwargsFunction<Conv1dFunction, Conv1dFunction.Kwargs>
    {
        public class Kwargs
        {
            public int Stride { get; set; }
            public int PadLeft { get; set; }
            public int PadRight { get; set; }
            public Kwargs() : this(1, 0)
            {
            }
            public Kwargs(int stride = 1, int padding = 0)
            {
                Stride = stride;
                PadLeft = padding;
                PadRight = padding;
            }

            public Kwargs(int stride, int padLeft, int padRight)
            {
                Stride = stride;
                PadLeft = padLeft;
                PadRight = padRight;
            }
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            var input = ctx.GetInput(0);
            var weight = ctx.GetInput(1);
            ctx.TryGetInput(2, out Tensor bias);

            return Tensor.Conv1d(
                input, weight, bias,
                kwargs.Stride, kwargs.PadLeft, kwargs.PadRight
            );
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            var input = ctx.GetInput(0);  // (N, C_in, L_in)
            var weight = ctx.GetInput(1); // (C_out, C_in, K_l)
            
            int n = input.Size[0];
            int cIn = input.Size[1];
            int lIn = input.Size[2];
            
            int cOut = weight.Size[0];
            int kernelSize = weight.Size[2];

            int lOut = grad.Size[2];
            int patchCount = n * lOut;

            // --- 1. gradInput (dL/dInput) ---
            var gradInput = Tensor.Conv1dTranspose(
                grad, weight, null,
                kwargs.Stride, kwargs.PadLeft, kwargs.PadRight,
                lIn
            );

            // --- 2. gradWeight (dL/dWeight) ---
            var inputCol = Tensor.Im2Col1d(
                input, kernelSize,
                kwargs.Stride, kwargs.PadLeft, kwargs.PadRight, 
                lOut
            );

            var gradT = Tensor.Transpose(grad, 1, 0); // (C_out, N, L_out)
            var gradFlat = Tensor.Reshape(
                gradT, 
                new int[] { cOut, patchCount }
            ); // (C_out, N*L_out)

            var gradWeightFlat = Tensor.MatMul(
                gradFlat, 
                Tensor.Transpose(inputCol, 0, 1)
            );
            
            var gradWeight = Tensor.Reshape(gradWeightFlat, weight.Size);

            // --- 3. gradBias (dL/dBias) ---
            if (ctx.TryGetInput(2, out Tensor _))
            {
                var ones = Tensor.Ones(new[] { patchCount, 1 });
                var gradBias = Tensor.MatMul(gradFlat, ones);
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
