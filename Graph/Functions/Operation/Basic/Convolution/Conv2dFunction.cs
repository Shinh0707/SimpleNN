namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;
    using System;

    /// <summary>
    /// 2D 畳み込みの自動微分関数.
    /// </summary>
    public class Conv2dFunction : KwargsFunction<Conv2dFunction, Conv2dFunction.Kwargs>
    {
        /// <summary>
        /// 畳み込みのパラメータ
        /// </summary>
        public class Kwargs
        {
            public int StrideH { get; set; }
            public int StrideW { get; set; }
            public int PadH { get; set; }
            public int PadW { get; set; }
            public Kwargs() : this(1, 0)
            {
            }
            public Kwargs(int stride = 1, int padding = 0)
            {
                StrideH = stride;
                StrideW = stride;
                PadH = padding;
                PadW = padding;
            }

            public Kwargs(int strideH, int strideW, int padH, int padW)
            {
                StrideH = strideH;
                StrideW = strideW;
                PadH = padH;
                PadW = padW;
            }
        }

        /// <summary>
        /// フォワードパス (im2col -> MatMul)
        /// Inputs: 0=input, 1=weight, 2=bias (optional)
        /// </summary>
        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            var input = ctx.GetInput(0);
            var weight = ctx.GetInput(1);
            ctx.TryGetInput(2, out Tensor bias);

            return Tensor.Conv2d(
                input, weight, bias,
                kwargs.StrideH, kwargs.StrideW,
                kwargs.PadH, kwargs.PadW
            );
        }

        /// <summary>
        /// バックワードパス (勾配計算)
        /// </summary>
        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            // 保存したテンソルを取得
            var input = ctx.GetInput(0);  // (N, C_in, H_in, W_in)
            var weight = ctx.GetInput(1); // (C_out, C_in, K_h, K_w)
            
            // 形状情報を取得
            int n = input.Size[0];
            int cIn = input.Size[1];
            int hIn = input.Size[2];
            int wIn = input.Size[3];
            
            int cOut = weight.Size[0];
            int kH = weight.Size[2];
            int kW = weight.Size[3];

            int hOut = grad.Size[2];
            int wOut = grad.Size[3];
            int patchCount = n * hOut * wOut;

            // --- 1. gradInput (dL/dInput) の計算 ---
            // dL/dInput = Conv2dTranspose(dL/dOutput, Weight)
            // (N,C_out,H_out,W_out) @ (C_out,C_in,K,K) -> (N,C_in,H_in,W_in)
            var gradInput = Tensor.Conv2dTranspose(
                grad, weight, null, // バイアスなし
                kwargs.StrideH, kwargs.StrideW,
                kwargs.PadH, kwargs.PadW,
                hIn, wIn
            );

            // --- 2. gradWeight (dL/dWeight) の計算 ---
            // dL/dWeight = MatMul(dL/dOutput_flat, Input_col.T)
            
            // Input を Im2Col で展開
            // (N,C_in,H_in,W_in) -> (K_size, N*H_out*W_out)
            var inputCol = Tensor.Im2Col(
                input, kH, kW, 
                kwargs.StrideH, kwargs.StrideW, 
                kwargs.PadH, kwargs.PadW, 
                hOut, wOut
            );

            // grad を (C_out, N*H_out*W_out) に変形
            var gradT = Tensor.Transpose(grad, 1, 0); // (C_out, N, H_out, W_out)
            var gradFlat = Tensor.Reshape(
                gradT, 
                new int[] { cOut, patchCount }
            ); // (C_out, N*H_out*W_out)

            // dL/dWeight_flat = dL/dOutput_flat @ Input_col.T
            // (C_out, N*H*W) @ (N*H*W, K_size) -> (C_out, K_size)
            var gradWeightFlat = Tensor.MatMul(
                gradFlat, 
                Tensor.Transpose(inputCol, 0, 1)
            );
            
            // (C_out, K_size) -> (C_out, C_in, K_h, K_w)
            var gradWeight = Tensor.Reshape(gradWeightFlat, weight.Size);


            // --- 3. gradBias (dL/dBias) の計算 (存在する場合) ---
            if (ctx.TryGetInput(2, out Tensor _))
            {
                // dL/dBias = Sum(dL/dOutput, axis=[0, 2, 3])
                // gradFlat (C_out, N*H*W) を dim=1 で合計
                var ones = Tensor.Ones(new[] { patchCount, 1 });
                
                // (C_out, N*H*W) @ (N*H*W, 1) -> (C_out, 1)
                var gradBias = Tensor.MatMul(gradFlat, ones);
                
                // (C_out, 1) -> (C_out)
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