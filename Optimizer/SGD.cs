namespace SimpleNN.Optimizer
{
    using System.Collections.Generic;
    using SimpleNN.Graph;
    using SimpleNN.Tensor;

    /// <summary>
    /// 確率的勾配降下法 (Stochastic Gradient Descent) オプティマイザ.
    /// パラメータ更新式: W = W - learning_rate * grad
    /// </summary>
    public class SGD : Optimizer
    {
        private readonly float _learningRate;

        /// <summary>
        /// 学習率と更新対象のパラメータを指定して SGD を初期化する.
        /// </summary>
        /// <param name="learningRate">学習率 (例: 0.01). 各更新ステップでの移動量を調整する.</param>
        /// <param name="parameters">最適化を行う対象のパラメータリスト (TensorBoxのリスト)</param>
        public SGD(float learningRate, params List<TensorBox>[] parameters) : base(parameters)
        {
            _learningRate = learningRate;
        }

        /// <summary>
        /// 個別のパラメータに対する更新計算を行う.
        /// OptimizerBase.Step() から呼び出される.
        /// </summary>
        /// <param name="tensor">現在のパラメータの値 (重み)</param>
        /// <param name="grad">パラメータに対応する勾配</param>
        /// <returns>更新後のパラメータの値</returns>
        protected override Tensor StepGrad(int _, Tensor tensor, Tensor grad)
        {
            return tensor - (grad * _learningRate);
        }
    }
}