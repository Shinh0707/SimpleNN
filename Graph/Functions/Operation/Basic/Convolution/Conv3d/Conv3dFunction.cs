namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;
    using System;

    /// <summary>
    /// Automatic differentiation function for 3D convolution.
    /// </summary>
    public class Conv3dFunction : KwargsFunction<Conv3dFunction, Conv3dFunction.Kwargs>
    {
        public class Kwargs
        {
            public int StrideD { get; set; }
            public int StrideH { get; set; }
            public int StrideW { get; set; }
            public int PadD { get; set; }
            public int PadH { get; set; }
            public int PadW { get; set; }

            public Kwargs() : this(1, 0)
            {
            }
            public Kwargs(int stride = 1, int padding = 0)
            {
                StrideD = stride;
                StrideH = stride;
                StrideW = stride;
                PadD = padding;
                PadH = padding;
                PadW = padding;
            }

            public Kwargs(int strideD, int strideH, int strideW, int padD, int padH, int padW)
            {
                StrideD = strideD;
                StrideH = strideH;
                StrideW = strideW;
                PadD = padD;
                PadH = padH;
                PadW = padW;
            }
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            var input = ctx.GetInput(0);
            var weight = ctx.GetInput(1);
            ctx.TryGetInput(2, out Tensor bias);

            return Tensor.Conv3d(
                input, weight, bias,
                kwargs.StrideD, kwargs.StrideH, kwargs.StrideW,
                kwargs.PadD, kwargs.PadH, kwargs.PadW
            );
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            var input = ctx.GetInput(0);  // (N, C_in, D_in, H_in, W_in)
            var weight = ctx.GetInput(1); // (C_out, C_in, K_d, K_h, K_w)
            
            int n = input.Size[0];
            int cIn = input.Size[1];
            int dIn = input.Size[2];
            int hIn = input.Size[3];
            int wIn = input.Size[4];
            
            int cOut = weight.Size[0];
            int kD = weight.Size[2];
            int kH = weight.Size[3];
            int kW = weight.Size[4];

            int dOut = grad.Size[2];
            int hOut = grad.Size[3];
            int wOut = grad.Size[4];
            int patchCount = n * dOut * hOut * wOut;

            // --- 1. gradInput (dL/dInput) ---
            var gradInput = Tensor.Conv3dTranspose(
                grad, weight, null,
                kwargs.StrideD, kwargs.StrideH, kwargs.StrideW,
                kwargs.PadD, kwargs.PadH, kwargs.PadW,
                dIn, hIn, wIn
            );

            // --- 2. gradWeight (dL/dWeight) ---
            var inputCol = Tensor.Im2Col3d(
                input, kD, kH, kW,
                kwargs.StrideD, kwargs.StrideH, kwargs.StrideW,
                kwargs.PadD, kwargs.PadH, kwargs.PadW,
                dOut, hOut, wOut
            );

            var gradT = Tensor.Transpose(grad, 1, 0); // (C_out, N, D_out, H_out, W_out)
            var gradFlat = Tensor.Reshape(
                gradT, 
                new int[] { cOut, patchCount }
            ); // (C_out, N*D*H*W)

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
