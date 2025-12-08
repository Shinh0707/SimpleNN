namespace SimpleNN.Module
{
    using System;
    using SimpleNN.Graph;
    using SimpleNN.Tensor;

    /// <summary>
    /// 2D 畳み込みレイヤー (Conv2d) モジュール.
    /// </summary>
    public class Conv2d : SingleInSingleOutModule
    {
        private TensorBox _weight;
        private TensorBox _bias;
        
        private int _strideH, _strideW, _padH, _padW;
        private bool _useBias;

        /// <summary>
        /// 2D 畳み込みモジュールを初期化する (カーネル・ストライド・パディングが縦横共通).
        /// </summary>
        /// <param name="inChannels">入力チャネル数</param>
        /// <param name="outChannels">出力チャネル数 (フィルター数)</param>
        /// <param name="kernelSize">カーネルのサイズ (縦横共通)</param>
        /// <param name="stride">ストライド (縦横共通)</param>
        /// <param name="padding">パディング (縦横共通)</param>
        /// <param name="bias">バイアス項を含めるかどうか</param>
        public Conv2d(
            int inChannels, 
            int outChannels, 
            int kernelSize, 
            int stride = 1, 
            int padding = 0, 
            bool bias = true)
            : this(
                  inChannels, outChannels, 
                  kernelSize, kernelSize, // K_h, K_w
                  stride, stride,       // S_h, S_w
                  padding, padding,    // P_h, P_w
                  bias)
        {
        }

        /// <summary>
        /// 2D 畳み込みモジュールを初期化する (詳細設定).
        /// </summary>
        /// <param name="inChannels">入力チャネル数</param>
        /// <param name="outChannels">出力チャネル数</param>
        /// <param name="kernelH">カーネルの高さ</param>
        /// <param name="kernelW">カーネルの幅</param>
        /// <param name="strideH">縦方向のストライド</param>
        /// <param name="strideW">横方向のストライド</param>
        /// <param name="padH">縦方向のパディング</param>
        /// <param name="padW">横方向のパディング</param>
        /// <param name="bias">バイアス項を含めるかどうか</param>
        public Conv2d(
            int inChannels, 
            int outChannels, 
            int kernelH, 
            int kernelW, 
            int strideH, 
            int strideW, 
            int padH, 
            int padW, 
            bool bias = true)
        {
            _strideH = strideH;
            _strideW = strideW;
            _padH = padH;
            _padW = padW;
            _useBias = bias;

            // 重みとバイアスの初期化
            InitializeWeights(inChannels, outChannels, kernelH, kernelW);
            if (bias)
            {
                InitializeBias(outChannels);
            }
        }

        /// <summary>
        /// 重みパラメータ (W) を初期化し, 登録する.
        /// (C_out, C_in, K_h, K_w)
        /// </summary>
        private void InitializeWeights(int inChannels, int outChannels, int kernelH, int kernelW)
        {
            // Linear と同様に、入力の特徴量数 (fan-in) に基づいてスケールを決定
            int fanIn = inChannels * kernelH * kernelW;
            float limit = 1.0f / MathF.Sqrt(fanIn);
            
            var tensor = Tensor.Random(
                new int[] { outChannels, inChannels, kernelH, kernelW }, 
                -limit, 
                limit
            );
            
            // パラメータとして登録
            _weight = AddParameter(new TensorBox(tensor));
        }

        /// <summary>
        /// バイアスパラメータ (b) を初期化し, 登録する.
        /// (C_out)
        /// </summary>
        private void InitializeBias(int outChannels)
        {
            var tensor = Tensor.Zeros(new int[] { outChannels });
            
            // パラメータとして登録
            _bias = AddParameter(new TensorBox(tensor));
        }

        /// <summary>
        /// このモジュールの出力テンソルのサイズを計算する.
        /// </summary>
        /// <param name="inputSize">入力テンソルのサイズ (N, C_in, H_in, W_in)</param>
        /// <returns>出力テンソルのサイズ (N, C_out, H_out, W_out)</returns>
        /// <exception cref="ArgumentException">
        /// 入力サイズの次元が 4 (N, C, H, W) ではない場合にスローされる.
        /// </exception>
        public int[] GetOutputSize(params int[] inputSize)
        {
            // inputSize (N, C_in, H_in, W_in)
            if (inputSize.Length != 4)
            {
                throw new ArgumentException(
                    "Input must be a 4D", 
                    nameof(inputSize)
                );
            }
            
            int n = inputSize[0];
            int hIn = inputSize[2];
            int wIn = inputSize[3];

            // _weight.Shape (C_out, C_in, K_h, K_w)
            int cOut = _weight.Size[0];
            int kH = _weight.Size[2];
            int kW = _weight.Size[3];

            // H_out = floor((H_in + 2*P_h - K_h) / S_h) + 1
            // W_out = floor((W_in + 2*P_w - K_w) / S_w) + 1
            int hOut = (hIn + 2 * _padH - kH) / _strideH + 1;
            int wOut = (wIn + 2 * _padW - kW) / _strideW + 1;
            
            int[] outputSize = new int[4];
            outputSize[0] = n;
            outputSize[1] = cOut;
            outputSize[2] = hOut;
            outputSize[3] = wOut;
            
            return outputSize;
        }

        /// <summary>
        /// フォワードパスを実行する.
        /// </summary>
        public override TensorBox Forward(TensorBox inputTensor)
        {
            // TensorBox の畳み込み関数を呼び出す
            return TensorBox.Conv2d(
                inputTensor, 
                _weight, 
                _strideH, _strideW, 
                _padH, _padW,
                _useBias ? _bias : null //
            );
        }
    }

    /// <summary>
    /// 2D 転置畳み込みレイヤー (TransposedConv2d) モジュール.
    /// (逆畳み込み, Deconvolution)
    /// </summary>
    public class TransposedConv2d : SingleInSingleOutModule
    {
        private TensorBox _weight;
        private TensorBox _bias;
        
        private int _strideH, _strideW, _padH, _padW;
        private bool _useBias;

        /// <summary>
        /// 2D 転置畳み込みモジュールを初期化する (カーネル・ストライド・パディングが縦横共通).
        /// </summary>
        /// <param name="inChannels">入力チャネル数</param>
        /// <param name="outChannels">出力チャネル数</param>
        /// <param name="kernelSize">カーネルのサイズ (縦横共通)</param>
        /// <param name="stride">ストライド (縦横共通)</param>
        /// <param name="padding">パディング (縦横共通)</param>
        /// <param name="bias">バイアス項を含めるかどうか</param>
        public TransposedConv2d(
            int inChannels, 
            int outChannels, 
            int kernelSize, 
            int stride = 1, 
            int padding = 0, 
            bool bias = true)
            : this(
                inChannels, outChannels, 
                kernelSize, kernelSize, // K_h, K_w
                stride, stride,       // S_h, S_w
                padding, padding,    // P_h, P_w
                bias)
        {
        }

        /// <summary>
        /// 2D 転置畳み込みモジュールを初期化する (詳細設定).
        /// </summary>
        /// <param name="inChannels">入力チャネル数</param>
        /// <param name="outChannels">出力チャネル数</param>
        /// <param name="kernelH">カーネルの高さ</param>
        /// <param name="kernelW">カーネルの幅</param>
        /// <param name="strideH">縦方向のストライド</param>
        /// <param name="strideW">横方向のストライド</param>
        /// <param name="padH">縦方向のパディング</param>
        /// <param name="padW">横方向のパディング</param>
        /// <param name="bias">バイアス項を含めるかどうか</param>
        public TransposedConv2d(
            int inChannels, 
            int outChannels, 
            int kernelH, 
            int kernelW, 
            int strideH, 
            int strideW, 
            int padH, 
            int padW, 
            bool bias = true)
        {
            _strideH = strideH;
            _strideW = strideW;
            _padH = padH;
            _padW = padW;
            _useBias = bias;

            // 重みとバイアスの初期化
            InitializeWeights(inChannels, outChannels, kernelH, kernelW);
            if (bias)
            {
                InitializeBias(outChannels);
            }
        }

        /// <summary>
        /// 重みパラメータ (W) を初期化し, 登録する.
        /// (C_in, C_out, K_h, K_w)
        /// </summary>
        private void InitializeWeights(int inChannels, int outChannels, int kernelH, int kernelW)
        {
            // Conv2d と同様に fan-in (入力チャネル数) に基づいてスケールを決定
            int fanIn = inChannels;
            float limit = 1.0f / MathF.Sqrt(fanIn);
            
            var tensor = Tensor.Random(
                new int[] { inChannels, outChannels, kernelH, kernelW }, 
                -limit, 
                limit
            );
            
            // パラメータとして登録
            _weight = AddParameter(new TensorBox(tensor));
        }

        /// <summary>
        /// バイアスパラメータ (b) を初期化し, 登録する.
        /// (C_out)
        /// </summary>
        private void InitializeBias(int outChannels)
        {
            var tensor = Tensor.Zeros(new int[] { outChannels });
            
            // パラメータとして登録
            _bias = AddParameter(new TensorBox(tensor));
        }

        /// <summary>
        /// このモジュールの出力テンソルのサイズを計算する.
        /// </summary>
        /// <param name="inputSize">入力テンソルのサイズ (N, C_in, H_in, W_in)</param>
        /// <returns>出力テンソルのサイズ (N, C_out, H_out, W_out)</returns>
        /// <exception cref="ArgumentException">
        /// 入力サイズの次元が 4 (N, C, H, W) ではない場合にスローされる.
        /// </exception>
        public int[] GetOutputSize(params int[] inputSize)
        {
            // inputSize (N, C_in, H_in, W_in)
            if (inputSize.Length != 4)
            {
                throw new ArgumentException(
                    "Input must be a 4D", 
                    nameof(inputSize)
                );
            }
            
            int n = inputSize[0];
            int hIn = inputSize[2];
            int wIn = inputSize[3];

            // _weight.Shape (C_in, C_out, K_h, K_w)
            int cOut = _weight.Size[1];
            int kH = _weight.Size[2];
            int kW = _weight.Size[3];

            // H_out = (H_in - 1) * S_h - 2*P_h + K_h
            // W_out = (W_in - 1) * S_w - 2*P_w + K_w
            // (output_padding は 0 と仮定)
            
            int hOut = (hIn - 1) * _strideH - 2 * _padH + kH;
            int wOut = (wIn - 1) * _strideW - 2 * _padW + kW;
            
            int[] outputSize = new int[4];
            outputSize[0] = n;
            outputSize[1] = cOut;
            outputSize[2] = hOut;
            outputSize[3] = wOut;
            
            return outputSize;
        }

        /// <summary>
        /// フォワードパスを実行する.
        /// </summary>
        public override TensorBox Forward(TensorBox inputTensor)
        {
            // TensorBox の転置畳み込み関数を呼び出す
            return TensorBox.TransposedConv2d(
                inputTensor, 
                _weight, 
                _strideH, _strideW, 
                _padH, _padW,
                _useBias ? _bias : null
            );
        }
    }
    /// <summary>
    /// 2D 最大プーリングレイヤー (MaxPool2d) モジュール.
    /// </summary>
    public class MaxPool2d : SingleInSingleOutModule
    {
        private int _kernelH, _kernelW, _strideH, _strideW, _padH, _padW;

        /// <summary>
        /// 2D 最大プーリングモジュールを初期化する
        /// (カーネル・ストライド・パディングが縦横共通).
        /// </summary>
        /// <param name="kernelSize">カーネルのサイズ (縦横共通)</param>
        /// <param name="stride">
        /// ストライド (縦横共通).
        /// 0 以下を指定した場合, <paramref name="kernelSize"/> と同じ値が使われる.
        /// </param>
        /// <param name="padding">パディング (縦横共通)</param>
        public MaxPool2d(int kernelSize, int stride = -1, int padding = 0)
        {
            // ストライドのデフォルト値をカーネルサイズにする
            int s = (stride <= 0) ? kernelSize : stride;

            // 詳細設定コンストラクタを呼び出す
            Initialize(
                kernelSize, kernelSize, // K_h, K_w
                s, s,                   // S_h, S_w
                padding, padding        // P_h, P_w
            );
        }
        
        /// <summary>
        /// 2D 最大プーリングモジュールを初期化する (詳細設定).
        /// </summary>
        public MaxPool2d(
            int kernelH, int kernelW, 
            int strideH, int strideW, 
            int padH, int padW)
        {
            Initialize(kernelH, kernelW, strideH, strideW, padH, padW);
        }

        /// <summary>
        /// パラメータを初期化する（コンストラクタ共通処理）
        /// </summary>
        private void Initialize(
            int kernelH, int kernelW, 
            int strideH, int strideW, 
            int padH, int padW)
        {
            _kernelH = kernelH;
            _kernelW = kernelW;
            _strideH = strideH;
            _strideW = strideW;
            _padH = padH;
            _padW = padW;
        }
        
        /// <summary>
        /// このモジュールの出力テンソルのサイズを計算する.
        /// </summary>
        /// <param name="inputSize">入力テンソルのサイズ (N, C_in, H_in, W_in)</param>
        /// <returns>出力テンソルのサイズ (N, C_out, H_out, W_out)</returns>
        /// <exception cref="ArgumentException">
        /// 入力サイズの次元が 4 (N, C, H, W) ではない場合にスローされる.
        /// </exception>
        public int[] GetOutputSize(params int[] inputSize)
        {
            // inputSize (N, C_in, H_in, W_in)
            if (inputSize.Length != 4)
            {
                throw new ArgumentException(
                    "Input must be a 4D", 
                    nameof(inputSize)
                );
            }
            
            int n = inputSize[0];
            // プーリングではチャネル数は変わらない
            int cOut = inputSize[1]; 
            int hIn = inputSize[2];
            int wIn = inputSize[3];

            // H_out = floor((H_in + 2*P_h - K_h) / S_h) + 1
            // W_out = floor((W_in + 2*P_w - K_w) / S_w) + 1
            int hOut = (hIn + 2 * _padH - _kernelH) / _strideH + 1;
            int wOut = (wIn + 2 * _padW - _kernelW) / _strideW + 1;
            
            int[] outputSize = new int[4];
            outputSize[0] = n;
            outputSize[1] = cOut;
            outputSize[2] = hOut;
            outputSize[3] = wOut;
            
            return outputSize;
        }

        /// <summary>
        /// フォワードパスを実行する.
        /// </summary>
        public override TensorBox Forward(TensorBox inputTensor)
        {
            // TensorBox の MaxPool2d 関数を呼び出す
            return TensorBox.MaxPool2d(
                inputTensor, 
                _kernelH, _kernelW, 
                _strideH, _strideW, 
                _padH, _padW
            );
        }
    }

    /// <summary>
    /// 2D 平均プーリングレイヤー (AvgPool2d) モジュール.
    /// </summary>
    public class AvgPool2d : SingleInSingleOutModule
    {
        private int _kernelH, _kernelW, _strideH, _strideW, _padH, _padW;

        /// <summary>
        /// 2D 平均プーリングモジュールを初期化する
        /// (カーネル・ストライド・パディングが縦横共通).
        /// </summary>
        /// <param name="kernelSize">カーネルのサイズ (縦横共通)</param>
        /// <param name="stride">
        /// ストライド (縦横共通).
        /// 0 以下を指定した場合, <paramref name="kernelSize"/> と同じ値が使われる.
        /// </param>
        /// <param name="padding">パディング (縦横共通)</param>
        public AvgPool2d(int kernelSize, int stride = -1, int padding = 0)
        {
            // ストライドのデフォルト値をカーネルサイズにする
            int s = (stride <= 0) ? kernelSize : stride;
            
            Initialize(
                kernelSize, kernelSize, 
                s, s, 
                padding, padding
            );
        }
        
        /// <summary>
        /// 2D 平均プーリングモジュールを初期化する (詳細設定).
        /// </summary>
        public AvgPool2d(
            int kernelH, int kernelW, 
            int strideH, int strideW, 
            int padH, int padW)
        {
            Initialize(kernelH, kernelW, strideH, strideW, padH, padW);
        }

        /// <summary>
        /// パラメータを初期化する（コンストラクタ共通処理）
        /// </summary>
        private void Initialize(
            int kernelH, int kernelW, 
            int strideH, int strideW, 
            int padH, int padW)
        {
            _kernelH = kernelH;
            _kernelW = kernelW;
            _strideH = strideH;
            _strideW = strideW;
            _padH = padH;
            _padW = padW;
        }

        /// <summary>
        /// このモジュールの出力テンソルのサイズを計算する.
        /// </summary>
        /// <param name="inputSize">入力テンソルのサイズ (N, C_in, H_in, W_in)</param>
        /// <returns>出力テンソルのサイズ (N, C_out, H_out, W_out)</returns>
        /// <exception cref="ArgumentException">
        /// 入力サイズの次元が 4 (N, C, H, W) ではない場合にスローされる.
        /// </exception>
        public int[] GetOutputSize(params int[] inputSize)
        {
            // inputSize (N, C_in, H_in, W_in)
            if (inputSize.Length != 4)
            {
                throw new ArgumentException(
                    "Input must be a 4D", 
                    nameof(inputSize)
                );
            }
            
            int n = inputSize[0];
            int cOut = inputSize[1]; // チャネル数は不変
            int hIn = inputSize[2];
            int wIn = inputSize[3];

            // H_out = floor((H_in + 2*P_h - K_h) / S_h) + 1
            // W_out = floor((W_in + 2*P_w - K_w) / S_w) + 1
            int hOut = (hIn + 2 * _padH - _kernelH) / _strideH + 1;
            int wOut = (wIn + 2 * _padW - _kernelW) / _strideW + 1;
            
            int[] outputSize = new int[4];
            outputSize[0] = n;
            outputSize[1] = cOut;
            outputSize[2] = hOut;
            outputSize[3] = wOut;
            
            return outputSize;
        }

        /// <summary>
        /// フォワードパスを実行する.
        /// </summary>
        public override TensorBox Forward(TensorBox inputTensor)
        {
            // TensorBox の AvgPool2d 関数を呼び出す
            return TensorBox.AvgPool2d(
                inputTensor, 
                _kernelH, _kernelW, 
                _strideH, _strideW, 
                _padH, _padW
            );
        }
    }
}