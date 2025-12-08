namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    /// <summary>
    /// 2D 最大プーリングの自動微分関数.
    /// </summary>
    public class MaxPool2dFunction : KwargsFunction<MaxPool2dFunction, MaxPool2dFunction.Pool2dKwargs>
    {
        /// <summary>
        /// Pool2d (MaxPool, AvgPool) 共通のパラメータ
        /// </summary>
        public class Pool2dKwargs
        {
            public int KernelH { get; set; }
            public int KernelW { get; set; }
            public int StrideH { get; set; }
            public int StrideW { get; set; }
            public int PadH { get; set; }
            public int PadW { get; set; }

            public Pool2dKwargs() : this(3, -1, 0)
            {
            }

            /// <summary>
            /// 共通のカーネルサイズ, ストライド, パディングで初期化する
            /// </summary>
            /// <param name="kernelSize">カーネルサイズ（縦横同じ値）</param>
            /// <param name="stride">
            /// ストライド. -1 以下の場合は kernelSize が使用される.
            /// </param>
            /// <param name="padding">パディング（縦横同じ値）</param>
            public Pool2dKwargs(int kernelSize, int stride = -1, int padding = 0)
            {
                KernelH = kernelSize;
                KernelW = kernelSize;
                
                // ストライドのデフォルト値をカーネルサイズにする
                int s = (stride <= 0) ? kernelSize : stride;
                StrideH = s;
                StrideW = s;
                PadH = padding;
                PadW = padding;
            }

            /// <summary>
            /// 個別のパラメータで初期化する
            /// </summary>
            public Pool2dKwargs(
                int kernelH, int kernelW,
                int strideH, int strideW,
                int padH, int padW)
            {
                KernelH = kernelH;
                KernelW = kernelW;
                StrideH = strideH;
                StrideW = strideW;
                PadH = padH;
                PadW = padW;
            }
        }

        /// <summary>
        /// フォワードパス (MaxPool2d).
        /// Inputs: 0=input.
        /// flatIndices を float テンソルとして Backward のために保存する.
        /// </summary>
        internal override Tensor Forward(Pool2dKwargs kwargs, Context ctx)
        {
            var input = ctx.GetInput(0);
            (Tensor output, int[] flatIndices) = Tensor.MaxPool2d(
                input,
                kwargs.KernelH, kwargs.KernelW,
                kwargs.StrideH, kwargs.StrideW,
                kwargs.PadH, kwargs.PadW
            );
            ctx.RegisterDatas((object)flatIndices);

            return output;
        }

        /// <summary>
        /// バックワードパス (勾配計算).
        /// dL/dInput = ScatterAdd(dL/dOutput, max_indices)
        /// </summary>
        internal override Tensor[] Backward(Pool2dKwargs kwargs, Context ctx, Tensor grad)
        {
            var input = ctx.GetInput(0);
            var flatIndices = ctx.GetRegisteredData<int[]>(0); // (N*C*H_out*W_out)
            var gradInputData = new float[input.TotalSize];
            float[] gradData = grad.GetContiguousData();
            for (int i = 0; i < flatIndices.Length; i++)
            {
                int inputIndex = flatIndices[i];
                if (inputIndex != -1) 
                {
                    gradInputData[inputIndex] += gradData[i];
                }
            }

            return new[] { new Tensor(gradInputData, input.Size, input.Strides) };
        }
    }
}