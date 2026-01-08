namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;
    /// <summary>
    /// Automatic differentiation function for 3D transposed convolution.
    /// </summary>
    public class TransposedConv3dFunction : KwargsFunction<TransposedConv3dFunction, Conv3dFunction.Kwargs>
    {
        internal override Tensor Forward(Conv3dFunction.Kwargs kwargs, Context ctx)
        {
            var input = ctx.GetInput(0);
            var weight = ctx.GetInput(1); 
            ctx.TryGetInput(2, out Tensor bias);

            return Tensor.Conv3dTranspose(
                input, weight, bias,
                kwargs.StrideD, kwargs.StrideH, kwargs.StrideW,
                kwargs.PadD, kwargs.PadH, kwargs.PadW
            );
        }

        internal override Tensor[] Backward(Conv3dFunction.Kwargs kwargs, Context ctx, Tensor grad)
        {
            var input = ctx.GetInput(0);  
            var weight = ctx.GetInput(1); 
            
            int n = input.Size[0];
            int cIn = input.Size[1];
            int dIn = input.Size[2];
            int hIn = input.Size[3];
            int wIn = input.Size[4];
            
            int cOut = weight.Size[1];
            int kD = weight.Size[2];
            int kH = weight.Size[3];
            int kW = weight.Size[4];

            int dOut = grad.Size[2];
            int hOut = grad.Size[3];
            int wOut = grad.Size[4];
            int patchCount = n * dIn * hIn * wIn; // Input patches

            // --- 1. gradInput (dL/dInput) ---
            var weightT = Tensor.Transpose(weight, 0, 1);
            
            var gradInput = Tensor.Conv3d(
                grad, weightT, null,
                kwargs.StrideD, kwargs.StrideH, kwargs.StrideW,
                kwargs.PadD, kwargs.PadH, kwargs.PadW
            );

            // --- 2. gradWeight (dL/dWeight) ---
            var gradCol = Tensor.Im2Col3d(
                grad, kD, kH, kW,
                kwargs.StrideD, kwargs.StrideH, kwargs.StrideW,
                kwargs.PadD, kwargs.PadH, kwargs.PadW,
                dIn, hIn, wIn
            );

            var inputT = Tensor.Transpose(input, 1, 0); 
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
                var gradT_bias = Tensor.Transpose(grad, 1, 0); 
                var gradFlat_bias = Tensor.Reshape(
                    gradT_bias, 
                    new int[] { cOut, n * dOut * hOut * wOut }
                );
                
                var ones = Tensor.Ones(new[] { n * dOut * hOut * wOut, 1 });
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
