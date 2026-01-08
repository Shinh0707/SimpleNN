namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    /// <summary>
    /// 2D 平均プーリングの自動微分関数.
    /// </summary>
    public class AvgPool2dFunction : KwargsFunction<AvgPool2dFunction, MaxPool2dFunction.Pool2dKwargs>
    {
        /// <summary>
        /// フォワードパス (AveragePool2d).
        /// Inputs: 0=input.
        /// 各ウィンドウの要素数 (Count) を Backward のために保存する.
        /// </summary>
        internal override Tensor Forward(MaxPool2dFunction.Pool2dKwargs kwargs, Context ctx)
        {
            var input = ctx.GetInput(0);
            (Tensor output, int[] counts) = Tensor.AveragePool2d(
                input,
                kwargs.KernelH, kwargs.KernelW,
                kwargs.StrideH, kwargs.StrideW,
                kwargs.PadH, kwargs.PadW
            );

            ctx.RegisterDatas((object)counts);

            return output;
        }

        /// <summary>
        /// バックワードパス (勾配計算).
        /// dL/dInput = dL/dOutput * (1 / Count) をウィンドウ全体に分配する.
        /// </summary>
        internal override Tensor[] Backward(MaxPool2dFunction.Pool2dKwargs kwargs, Context ctx, Tensor grad)
        {
            var input = ctx.GetInput(0);
            var counts = ctx.GetRegisteredData<int[]>(0); // (N*C*H_out*W_out)
            
            // 2. 入力勾配配列を 0 で初期化
            var gradInputData = new float[input.TotalSize];
            
            // 3. 出力勾配 (grad) を Contiguous な配列として取得
            float[] gradData = grad.GetContiguousData();

            // 4. パラメータと形状情報を取得
            int kernelH = kwargs.KernelH;
            int kernelW = kwargs.KernelW;
            int strideH = kwargs.StrideH;
            int strideW = kwargs.StrideW;
            int padH = kwargs.PadH;
            int padW = kwargs.PadW;

            int batchSize = input.Size[0];
            int channels = input.Size[1];
            int inHeight = input.Size[2];
            int inWidth = input.Size[3];
            int[] inputStrides = input.Strides;

            int outHeight = grad.Size[2];
            int outWidth = grad.Size[3];

            int outputIndex = 0; // gradData と counts のインデックス

            // 5. Pool2d と同じループで勾配を分配する
            for (int n = 0; n < batchSize; n++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int oh = 0; oh < outHeight; oh++)
                    {
                        for (int ow = 0; ow < outWidth; ow++)
                        {
                            int count = counts[outputIndex];
                            float gradVal = gradData[outputIndex];
                            float gradPerInput = 0.0f;

                            // (dL/dOutput) / Count
                            if (count > 0)
                            {
                                gradPerInput = gradVal / count;
                            }
                            
                            // 0 でない勾配のみを分配する
                            if (gradPerInput != 0.0f) 
                            {
                                // ウィンドウの開始位置
                                int hStart = oh * strideH - padH;
                                int wStart = ow * strideW - padW;

                                // ウィンドウ内の入力に勾配を加算
                                for (int kh = 0; kh < kernelH; kh++)
                                {
                                    int ih = hStart + kh;
                                    for (int kw = 0; kw < kernelW; kw++)
                                    {
                                        int iw = wStart + kw;

                                        // パディング領域外かチェック
                                        if (ih >= 0 && ih < inHeight &&
                                            iw >= 0 && iw < inWidth)
                                        {
                                            // 1D インデックスを計算
                                            int inputIndex = 
                                                n * inputStrides[0] +
                                                c * inputStrides[1] +
                                                ih * inputStrides[2] +
                                                iw * inputStrides[3];
                                            
                                            // 勾配を加算
                                            gradInputData[inputIndex] += gradPerInput;
                                        }
                                    } // kw
                                } // kh
                            } // if gradPerInput != 0

                            outputIndex++;
                        } // ow
                    } // oh
                } // c
            } // n

            // 6. 勾配データを input と同じ形状/ストライドのテンソルにラップして返す
            return new[] { 
                new Tensor(gradInputData, input.Size, input.Strides) 
            };
        }
    }
}