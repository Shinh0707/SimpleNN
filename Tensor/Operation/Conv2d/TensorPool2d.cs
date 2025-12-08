namespace SimpleNN.Tensor
{
    using System;
    public partial class Tensor
    {
        // --- デリゲート定義 ---
        // float: 主な値 (Max値, Sum値など)
        // int:   補助的な値 (Maxインデックス, Count値など)
        

        private delegate (float val, int ptr) PoolInitFunc();
        private delegate (float val, int ptr) PoolAccumulateFunc(
            float inData, int inPtr, (float val, int ptr) currentState
        );
        // FinalizeFunc は、ウィンドウ内の要素数を引数に取るように変更
        // (AvgPool で Count > 0 のチェックに使うため)
        private delegate (float val, int ptr) PoolFinalizeFunc(
            (float val, int ptr) finalState, int windowElementCount
        );


        // --- Pool2d (Overloads) ---

        // （オーバーロードは kernelSize と stride の展開のみなので省略）
        private static (Tensor, int[]) Pool2d(
            Tensor input,
            PoolInitFunc initFunc,
            PoolAccumulateFunc accumulateFunc,
            PoolFinalizeFunc finalizeFunc,
            int kernelSize,
            int stride = -1,
            int padding = 0)
        {
            int s = (stride <= 0) ? kernelSize : stride;
            return Pool2d(
                input, initFunc, accumulateFunc, finalizeFunc,
                kernelSize, kernelSize, s, s, padding, padding
            );
        }

        // --- Pool2d (Core) ---

        /// <summary>
        /// 2D プーリング操作を実行する (汎用コア).
        /// </summary>
        private static (Tensor, int[]) Pool2d(
            Tensor input,
            PoolInitFunc initFunc,
            PoolAccumulateFunc accumulateFunc,
            PoolFinalizeFunc finalizeFunc,
            int kernelH, int kernelW,
            int strideH, int strideW,
            int padH, int padW)
        {
            // 1. 入力形状を取得 (変更なし)
            if (input.NDim != 4)
                throw new ArgumentException("Input must be a 4D tensor");

            int batchSize = input.Size[0];
            int channels = input.Size[1];
            int inHeight = input.Size[2];
            int inWidth = input.Size[3];

            float[] inputData = input._data;
            int[] inputStrides = input.Strides;

            // 2. 出力サイズを計算 (変更なし)
            int outHeight = (inHeight + 2 * padH - kernelH) / strideH + 1;
            int outWidth = (inWidth + 2 * padW - kernelW) / strideW + 1;

            if (outHeight <= 0 || outWidth <= 0)
                throw new ArgumentException("Invalid pooling parameters");

            // 3. 出力配列を初期化 (変更なし)
            int outTotalSize = batchSize * channels * outHeight * outWidth;
            float[] outputData = new float[outTotalSize];
            int[] flatIndices = new int[outTotalSize];
            int outputIndex = 0;

            // 4. プーリング実行
            for (int n = 0; n < batchSize; n++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int oh = 0; oh < outHeight; oh++)
                    {
                        for (int ow = 0; ow < outWidth; ow++)
                        {
                            // (1) 初期化
                            var currentState = initFunc();
                            int windowElementCount = 0; // パディング除く要素数

                            int hStart = oh * strideH - padH;
                            int wStart = ow * strideW - padW;

                            for (int kh = 0; kh < kernelH; kh++)
                            {
                                int ih = hStart + kh;
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    int iw = wStart + kw;

                                    if (ih >= 0 && ih < inHeight &&
                                        iw >= 0 && iw < inWidth)
                                    {
                                        int inputIndex = n * inputStrides[0] +
                                                         c * inputStrides[1] +
                                                         ih * inputStrides[2] +
                                                         iw * inputStrides[3];
                                        float val = inputData[inputIndex];

                                        // (2) 蓄積
                                        currentState = accumulateFunc(
                                            val, inputIndex, currentState
                                        );
                                        windowElementCount++;
                                    }
                                } // kw
                            } // kh

                            // (3) 最終化
                            var finalResult = finalizeFunc(
                                currentState, windowElementCount
                            );

                            outputData[outputIndex] = finalResult.val;
                            flatIndices[outputIndex] = finalResult.ptr;
                            outputIndex++;

                        } // ow
                    } // oh
                } // c
            } // n

            var outputSize = new int[] {
                batchSize, channels, outHeight, outWidth
            };
            var outputTensor = new Tensor(outputData, outputSize);

            return (outputTensor, flatIndices);
        }
    }
}