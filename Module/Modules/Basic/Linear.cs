namespace SimpleNN.Module
{
    using System;
    using SimpleNN.Graph;
    using SimpleNN.Tensor;

    /// <summary>
    /// 入力データに対して線形変換 (y = xW + b) を適用する全結合層モジュール.
    /// </summary>
    public class Linear : SingleInSingleOutModule
    {
        private TensorBox _weight;
        private TensorBox _bias;
        private bool _useBias;

        /// <summary>
        /// 入力サイズと出力サイズを指定して Linear モジュールを初期化する.
        /// </summary>
        /// <param name="inFeatures">入力テンソルのサイズ（各サンプルの特徴量数）</param>
        /// <param name="outFeatures">出力テンソルのサイズ</param>
        /// <param name="bias">バイアス項を含めるかどうか</param>
        public Linear(int inFeatures, int outFeatures, bool bias = true)
        {
            InitializeWeights(inFeatures, outFeatures);
            _useBias = bias;
            if (bias)
            {
                InitializeBias(outFeatures);
            }
        }

        /// <summary>
        /// 重みパラメータ (W) を初期化し, 登録する.
        /// 初期値は -1/sqrt(in) ~ 1/sqrt(in) の一様乱数を使用する.
        /// </summary>
        private void InitializeWeights(int inFeatures, int outFeatures)
        {
            float limit = 1.0f / MathF.Sqrt(inFeatures);
            var tensor = Tensor.Random(new int[] { inFeatures, outFeatures }, -limit, limit);
            _weight = AddParameter(new TensorBox(tensor));
        }

        /// <summary>
        /// バイアスパラメータ (b) を初期化し, 登録する.
        /// 初期値は 0 とする.
        /// </summary>
        private void InitializeBias(int outFeatures)
        {
            var tensor = Tensor.Zeros(new int[] { outFeatures });
            _bias = AddParameter(new TensorBox(tensor));
        }
        public override TensorBox Forward(TensorBox inputTensor)
        {
            var output = TensorBox.MatMul(inputTensor, _weight);
            if (_useBias)
            {
                output += _bias;
            }
            return output;
        }
    }
}