namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;
    /// <summary>
    /// 2D 転置畳み込みの自動微分関数.
    /// </summary>
    public class TransposedConv2dFunction : KwargsFunction<TransposedConv2dFunction,  Conv2dFunction.Kwargs>
    {
        /// <summary>
        /// フォワードパス (MatMul -> col2im)
        /// Inputs: 0=input, 1=weight, 2=bias (optional)
        /// </summary>
        internal override Tensor Forward(Conv2dFunction.Kwargs kwargs, Context ctx)
        {
            var input = ctx.GetInput(0);
            var weight = ctx.GetInput(1); // (C_in, C_out, K_h, K_w)
            ctx.TryGetInput(2, out Tensor bias);

            return Tensor.Conv2dTranspose(
                input, weight, bias,
                kwargs.StrideH, kwargs.StrideW,
                kwargs.PadH, kwargs.PadW
            );
        }

        /// <summary>
        /// バックワードパス (勾配計算)
        /// </summary>
        internal override Tensor[] Backward(Conv2dFunction.Kwargs kwargs, Context ctx, Tensor grad)
        {
            // 保存したテンソルを取得
            var input = ctx.GetInput(0);  // (N, C_in, H_in, W_in)
            var weight = ctx.GetInput(1); // (C_in, C_out, K_h, K_w)
            
            // 形状情報を取得
            int n = input.Size[0];
            int cIn = input.Size[1];
            int hIn = input.Size[2];
            int wIn = input.Size[3];
            
            int cOut = weight.Size[1];
            int kH = weight.Size[2];
            int kW = weight.Size[3];

            int hOut = grad.Size[2];
            int wOut = grad.Size[3];
            int patchCount = n * hIn * wIn; // Input パッチ数

            // --- 1. gradInput (dL/dInput) の計算 ---
            // dL/dInput = Conv2d(dL/dOutput, Weight)
            // Weight (C_in, C_out, K, K) -> (C_out, C_in, K, K) に転置
            var weightT = Tensor.Transpose(weight, 0, 1);
            
            // (N,C_out,H_out,W_out) @ (C_out,C_in,K,K) -> (N,C_in,H_in,W_in)
            var gradInput = Tensor.Conv2d(
                grad, weightT, null, // バイアスなし
                kwargs.StrideH, kwargs.StrideW,
                kwargs.PadH, kwargs.PadW
            );

            // --- 2. gradWeight (dL/dWeight) の計算 ---
            // dL/dWeight は Conv2d の dL/dWeight と同じ計算だが、
            // 'input' が 'grad' (im2col される側) に、
            // 'grad' が 'input' (flat になる側) になる.
            
            // dL/dOutput を Im2Col で展開
            // (N,C_out,H_out,W_out) -> (K_size_T, N*H_in*W_in)
            // K_size_T = C_out * K_h * K_w
            var gradCol = Tensor.Im2Col(
                grad, kH, kW, 
                kwargs.StrideH, kwargs.StrideW, 
                kwargs.PadH, kwargs.PadW, 
                hIn, wIn // 出力サイズが H_in, W_in
            );

            // Input を (C_in, N*H_in*W_in) に変形
            var inputT = Tensor.Transpose(input, 1, 0); // (C_in, N, H_in, W_in)
            var inputFlat = Tensor.Reshape(
                inputT, 
                new int[] { cIn, patchCount }
            ); // (C_in, N*H_in*W_in)

            // dL/dWeight_flat = Input_flat @ dL/dOutput_col.T
            // (C_in, N*H*W) @ (N*H*W, K_size_T) -> (C_in, K_size_T)
            var gradWeightFlat = Tensor.MatMul(
                inputFlat, 
                Tensor.Transpose(gradCol, 0, 1)
            );
            
            // (C_in, K_size_T) -> (C_in, C_out, K_h, K_w)
            var gradWeight = Tensor.Reshape(gradWeightFlat, weight.Size);


            // --- 3. gradBias (dL/dBias) の計算 (存在する場合) ---
            if (ctx.TryGetInput(2, out Tensor _))
            {
                // dL/dBias = Sum(dL/dOutput, axis=[0, 2, 3])
                // grad (N, C_out, H_out, W_out)
                
                var gradT_bias = Tensor.Transpose(grad, 1, 0); // (C_out, N, H, W)
                var gradFlat_bias = Tensor.Reshape(
                    gradT_bias, 
                    new int[] { cOut, n * hOut * wOut }
                );
                
                var ones = Tensor.Ones(new[] { n * hOut * wOut, 1 });
                
                // (C_out, N*H*W) @ (N*H*W, 1) -> (C_out, 1)
                var gradBias = Tensor.MatMul(gradFlat_bias, ones);
                
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