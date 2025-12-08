namespace SimpleNN.Tensor
{
    using System;

    /// <summary>
    /// このファイルは、Tensor クラスの畳み込み関連の操作を定義します.
    /// (TensorConvolution.cs)
    /// </summary>
    public partial class Tensor
    {
        // --- public Conv2d (Overloads) ---

        /// <summary>
        /// 2D 畳み込み操作を実行します.
        /// </summary>
        /// <param name="input">
        /// 入力テンソル. 形状 (N, C_in, H_in, W_in).
        /// </param>
        /// <param name="weight">
        /// 重み（カーネル）テンソル. 
        /// 形状 (C_out, C_in, K_h, K_w).
        /// </param>
        /// <param name="bias">
        /// バイアス テンソル. 形状 (C_out). null の場合はバイアスなし.
        /// </param>
        /// <param name="stride">ストライド（縦横同じ値）</param>
        /// <param name="padding">パディング（縦横同じ値）</param>
        /// <returns>畳み込み結果のテンソル. 形状 (N, C_out, H_out, W_out).</returns>
        public static Tensor Conv2d(
            Tensor input, Tensor weight, Tensor bias = null,
            int stride = 1, int padding = 0)
        {
            return Conv2d(
                input, weight, bias,
                stride, stride, padding, padding
            );
        }

        // --- public Conv2d (Core) ---

        /// <summary>
        /// 2D 畳み込み操作を実行します (im2col を使用).
        /// </summary>
        /// <param name="input">
        /// 入力テンソル. 形状 (N, C_in, H_in, W_in).
        /// </param>
        /// <param name="weight">
        /// 重み（カーネル）テンソル. 
        /// 形状 (C_out, C_in, K_h, K_w).
        /// </param>
        /// <param name="bias">
        /// バイアス テンソル. 形状 (C_out). null の場合はバイアスなし.
        /// </param>
        /// <param name="strideH">縦方向のストライド</param>
        /// <param name="strideW">横方向のストライド</param>
        /// <param name="padH">縦方向のパディング</param>
        /// <param name="padW">横方向のパディング</param>
        /// <returns>畳み込み結果のテンソル. 形状 (N, C_out, H_out, W_out).</returns>
        /// <exception cref="ArgumentException">
        /// 入力形状と重み形状に互換性がない場合にスローされます.
        /// </exception>
        public static Tensor Conv2d(
            Tensor input, Tensor weight, Tensor bias,
            int strideH, int strideW, int padH, int padW)
        {
            // 1. 入力と重みの形状を取得
            if (input.NDim != 4)
                throw new ArgumentException("Input must be a 4D tensor");
            if (weight.NDim != 4)
                throw new ArgumentException("Weight must be a 4D tensor");

            int batchSize = input.Size[0];
            int inChannels = input.Size[1];
            int inHeight = input.Size[2];
            int inWidth = input.Size[3];

            int outChannels = weight.Size[0];
            int inChannelsW = weight.Size[1];
            int kernelH = weight.Size[2];
            int kernelW = weight.Size[3];

            if (inChannels != inChannelsW)
            {
                throw new ArgumentException(
                    $"Input channels ({inChannels}) and weight " +
                    $"channels ({inChannelsW}) do not match."
                );
            }

            // 2. 出力サイズを計算
            int outHeight = (inHeight + 2 * padH - kernelH) / strideH + 1;
            int outWidth = (inWidth + 2 * padW - kernelW) / strideW + 1;

            if (outHeight <= 0 || outWidth <= 0)
            {
                throw new ArgumentException("Invalid convolution parameters");
            }

            // 3. Im2Col: 入力パッチを行列に展開
            // (N, C_in, H_in, W_in) -> (C_in * K_h * K_w, N * H_out * W_out)
            var inputCol = Im2Col(
                input, kernelH, kernelW, strideH, strideW,
                padH, padW, outHeight, outWidth
            );

            // 4. 重みをフラット化
            // (C_out, C_in, K_h, K_w) -> (C_out, C_in * K_h * K_w)
            int kernelSize = inChannels * kernelH * kernelW;
            var kernelFlat = Reshape(weight, new int[] { outChannels, kernelSize });

            // 5. 行列積で畳み込みを計算
            // MatMul( (C_out, K_size), (K_size, N * H_out * W_out) )
            // -> (C_out, N * H_out * W_out)
            var output = MatMul(kernelFlat, inputCol);

            // 6. 出力形状を (C_out, N, H_out, W_out) に変形
            output = Reshape(
                output,
                new int[] { outChannels, batchSize, outHeight, outWidth }
            );

            // 7. 軸を入れ替えて (N, C_out, H_out, W_out) にする
            output = Transpose(output, 0, 1);

            // 8. バイアスを加算 (ブロードキャスト利用)
            if (bias is not null)
            {
                if (bias.NDim != 1 || bias.Size[0] != outChannels)
                {
                    throw new ArgumentException("Bias shape mismatch");
                }
                // (C_out) -> (1, C_out, 1, 1)
                var biasView = Reshape(bias, new int[] { 1, outChannels, 1, 1 });
                // (N, C_out, H, W) + (1, C_out, 1, 1)
                output += biasView;
            }

            return output;
        }

        // --- private Im2Col Helper ---

        /// <summary>
        /// 畳み込みのための Im2Col (Image to Column) 変換を行う.
        /// 入力テンソルのスライディングウィンドウを抽出し, 
        /// 巨大な行列の列として配置する.
        /// </summary>
        /// <returns>
        /// 形状 (C_in * K_h * K_w, N * H_out * W_out) の 2D テンソル.
        /// </returns>
        public static Tensor Im2Col(
            Tensor input, int kernelH, int kernelW,
            int strideH, int strideW, int padH, int padW,
            int outH, int outW)
        {
            // 入力情報を取得
            int batchSize = input.Size[0];
            int inChannels = input.Size[1];
            int inHeight = input.Size[2];
            int inWidth = input.Size[3];

            float[] inputData = input._data;
            int[] inputStrides = input.Strides;

            // Im2Col 行列のサイズを計算
            // 行: 1パッチあたりの要素数 (C_in * K_h * K_w)
            // 列: パッチの総数 (N * H_out * W_out)
            int colRows = inChannels * kernelH * kernelW;
            int colCols = batchSize * outH * outW;

            float[] outputData = new float[colRows * colCols];

            int outputColIndex = 0; // 出力行列の現在の列インデックス

            // パッチ（出力）ごとにループ
            for (int n = 0; n < batchSize; n++)
            {
                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        int outputRowIndex = 0; // 出力行列の現在の行インデックス

                        // 1パッチ（カーネル）内の要素ごとにループ
                        for (int c = 0; c < inChannels; c++)
                        {
                            for (int kh = 0; kh < kernelH; kh++)
                            {
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    // 入力画像上の座標を計算
                                    int currentInH = oh * strideH + kh - padH;
                                    int currentInW = ow * strideW + kw - padW;

                                    float value = 0.0f;

                                    // パディング領域外かチェック
                                    if (currentInH >= 0 && currentInH < inHeight &&
                                        currentInW >= 0 && currentInW < inWidth)
                                    {
                                        // ストライドを考慮して 1D インデックスを計算
                                        int inputIndex = n * inputStrides[0] +
                                                         c * inputStrides[1] +
                                                         currentInH * inputStrides[2] +
                                                         currentInW * inputStrides[3];
                                        value = inputData[inputIndex];
                                    }

                                    // 出力行列 (row-major) に書き込む
                                    int outputIndex = outputRowIndex * colCols +
                                                      outputColIndex;
                                    outputData[outputIndex] = value;

                                    outputRowIndex++;
                                } // kw
                            } // kh
                        } // c

                        outputColIndex++;
                    } // ow
                } // oh
            } // n

            // 2D テンソルとして返す
            var outputSize = new int[] { colRows, colCols };
            return new Tensor(outputData, outputSize);
        }
        // --- public Conv2dTranspose (Core) ---

        /// <summary>
        /// 2D 転置畳み込み操作を実行します (col2im を使用).
        /// </summary>
        /// <param name="input">
        /// 入力テンソル. 形状 (N, C_in, H_in, W_in).
        /// </param>
        /// <param name="weight">
        /// 重み（カーネル）テンソル. 
        /// 形状 (C_in, C_out, K_h, K_w).
        /// (Conv2d と C_in/C_out の順序が逆なことに注意)
        /// </param>
        /// <param name="bias">
        /// バイアス テンソル. 形状 (C_out). null の場合はバイアスなし.
        /// </param>
        /// <param name="strideH">縦方向のストライド</param>
        /// <param name="strideW">横方向のストライド</param>
        /// <param name="padH">縦方向のパディング</param>
        /// <param name="padW">横方向のパディング</param>
        /// <returns>
        /// 転置畳み込み結果のテンソル. 形状 (N, C_out, H_out, W_out).
        /// </returns>
        /// <exception cref="ArgumentException">
        /// 形状に互換性がない場合にスローされます.
        /// </exception>
        public static Tensor Conv2dTranspose(
            Tensor input, Tensor weight, Tensor bias,
            int strideH, int strideW, int padH, int padW,
            int outHeight, int outWidth
        )
        {
            // 1. 入力と重みの形状を取得
            if (input.NDim != 4)
                throw new ArgumentException("Input must be a 4D tensor");
            if (weight.NDim != 4)
                throw new ArgumentException("Weight must be a 4D tensor");

            int batchSize = input.Size[0];
            int inChannels = input.Size[1]; // TransposeConv の C_in
            int inHeight = input.Size[2];
            int inWidth = input.Size[3];

            int inChannelsW = weight.Size[0]; // Weight の C_in
            int outChannels = weight.Size[1]; // TransposeConv の C_out
            int kernelH = weight.Size[2];
            int kernelW = weight.Size[3];

            if (inChannels != inChannelsW)
            {
                throw new ArgumentException(
                    $"Input channels ({inChannels}) and weight " +
                    $"channels ({inChannelsW}) do not match."
                );
            }

            if (outHeight <= 0 || outWidth <= 0)
            {
                throw new ArgumentException("Invalid transpose conv parameters");
            }
            // 3. MatMul の準備
            // Weight (C_in, C_out, K, K) -> (C_in, C_out*K*K)
            int kernelSize = outChannels * kernelH * kernelW;
            var weightFlat = Reshape(weight, new int[] { inChannels, kernelSize });

            // (C_in, C_out*K*K) -> (C_out*K*K, C_in)
            var weightT = Transpose(weightFlat, 0, 1);

            // Input (N, C_in, H_in, W_in) -> (C_in, N, H_in, W_in)
            var inputT = Transpose(input, 0, 1);

            // (C_in, N, H_in, W_in) -> (C_in, N * H_in * W_in)
            var inputFlat = Reshape(
                inputT, 
                new int[] { inChannels, batchSize * inHeight * inWidth }
            );

            // 4. 行列積で Col 配列を計算
            // MatMul( (C_out*K*K, C_in), (C_in, N*H_in*W_in) )
            // -> (C_out*K*K, N*H_in*W_in)
            var cols = MatMul(weightT, inputFlat);

            // 5. Col2Im: Col 配列を出力画像にマッピング
            var output = Col2Im(
                cols,
                batchSize, outChannels, outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW,
                padH, padW,
                inHeight, inWidth
            );
            
            // 6. バイアスを加算 (ブロードキャスト利用)
            if (bias is not null)
            {
                if (bias.NDim != 1 || bias.Size[0] != outChannels)
                {
                    throw new ArgumentException("Bias shape mismatch");
                }
                // (C_out) -> (1, C_out, 1, 1)
                var biasView = Reshape(bias, new int[] { 1, outChannels, 1, 1 });
                // (N, C_out, H, W) + (1, C_out, 1, 1)
                output += biasView;
            }

            return output;
        }

        public static Tensor Conv2dTranspose(
            Tensor input, Tensor weight, Tensor bias,
            int strideH, int strideW, int padH, int padW
        )
        {
            int inHeight = input.Size[2];
            int inWidth = input.Size[3];
            int kernelH = weight.Size[2];
            int kernelW = weight.Size[3];

            int outHeight = (inHeight - 1) * strideH - 2 * padH + kernelH;
            int outWidth = (inWidth - 1) * strideW - 2 * padW + kernelW;

            return Conv2dTranspose(
                input, weight, bias,
                strideH, strideW, padH, padW,
                outHeight, outWidth // 
            );
        }

        // --- private Col2Im Helper ---

        /// <summary>
        /// 畳み込みのための Col2Im (Column to Image) 変換を行う.
        /// Im2Col の逆操作.
        /// </summary>
        /// <param name="cols">
        /// 入力列行列. 形状 (C*K*K, N*H_in*W_in).
        /// </param>
        /// <param name="N">出力バッチサイズ</param>
        /// <param name="C">出力チャネル数</param>
        /// <param name="H">出力の高さ</param>
        /// <param name="W">出力の幅</param>
        /// <param name="kernelH">カーネルの高さ</param>
        /// <param name="kernelW">カーネルの幅</param>
        /// <param name="strideH">ストライド (縦)</param>
        /// <param name="strideW">ストライド (横)</param>
        /// <param name="padH">パディング (縦)</param>
        /// <param name="padW">パディング (横)</param>
        /// <param name="H_in">入力の高さ (Conv2d の出力高)</param>
        /// <param name="W_in">入力の幅 (Conv2d の出力幅)</param>
        /// <returns>形状 (N, C, H, W) のテンソル</returns>
        public static Tensor Col2Im(
            Tensor cols,
            int N, int C, int H, int W,
            int kernelH, int kernelW,
            int strideH, int strideW,
            int padH, int padW,
            int H_in, int W_in)
        {
            float[] colsData = cols._data;
            // int colRows = cols.Size[0]; // C * K * K
            int colCols = cols.Size[1]; // N * H_in * W_in

            // 出力配列 (0 で初期化される)
            float[] outputData = new float[N * C * H * W];

            // (N, H_in, W_in) のループ (入力パッチごと)
            for (int n = 0; n < N; n++)
            {
                for (int ih = 0; ih < H_in; ih++)
                {
                    for (int iw = 0; iw < W_in; iw++)
                    {
                        // (C, K_h, K_w) のループ (1パッチ内の要素ごと)
                        for (int c = 0; c < C; c++)
                        {
                            for (int kh = 0; kh < kernelH; kh++)
                            {
                                for (int kw = 0; kw < kernelW; kw++)
                                {
                                    // 出力画像上の座標を計算
                                    int currentOutH = ih * strideH + kh - padH;
                                    int currentOutW = iw * strideW + kw - padW;

                                    // パディング領域外かチェック
                                    if (currentOutH >= 0 && currentOutH < H &&
                                        currentOutW >= 0 && currentOutW < W)
                                    {
                                        // 1. colsData から読み取るインデックスを計算
                                        // colRow = (C, K_h, K_w)
                                        int colRow = (c * kernelH + kh) * kernelW + kw;
                                        // colCol = (N, H_in, W_in)
                                        int colCol = (n * H_in + ih) * W_in + iw;
                                        // 1D インデックス (Row-Major)
                                        int colsIndex = colRow * colCols + colCol;

                                        // 2. outputData に加算するインデックスを計算
                                        // (N, C, H, W)
                                        int outputIndex =
                                            n * (C * H * W) +
                                            c * (H * W) +
                                            currentOutH * W +
                                            currentOutW;

                                        // 3. 加算 (重複するピクセルは加算される)
                                        outputData[outputIndex] += colsData[colsIndex];
                                    }
                                } // kw
                            } // kh
                        } // c
                    } // iw
                } // ih
            } // n

            return new Tensor(outputData, new int[] { N, C, H, W });
        }
    }
}