namespace SimpleNN.Optimizer
{
    using System.Collections.Generic;
    using SimpleNN.Graph;
    using SimpleNN.Tensor;

    /// <summary>
    /// Adagrad (Adaptive Gradient Algorithm) オプティマイザ.
    /// 過去の勾配の二乗和を蓄積し, 学習率を調整する.
    /// </summary>
    public class Adagrad : Optimizer
    {
        private readonly float _learningRate;
        private readonly float _epsilon;
        private readonly Tensor[] _sumSquaredGrads;

        /// <summary>
        /// Adagrad を初期化する.
        /// </summary>
        /// <param name="learningRate">学習率</param>
        /// <param name="epsilon">ゼロ除算を防ぐための微小値</param>
        /// <param name="parameters">最適化対象のパラメータ</param>
        public Adagrad(float learningRate, float epsilon, params List<TensorBox>[] parameters) 
            : base(parameters)
        {
            _learningRate = learningRate;
            _epsilon = epsilon;
            _sumSquaredGrads = new Tensor[NumParameters];
        }
        
        public Adagrad(float learningRate, params List<TensorBox>[] parameters) 
            : this(learningRate, 1e-7f, parameters)
        {
        }
        protected override Tensor StepGrad(int id, Tensor tensor, Tensor grad)
        {
            Tensor sumSq = _sumSquaredGrads[id];
            sumSq ??= grad * 0.0f;
            sumSq += grad * grad;
            _sumSquaredGrads[id] = sumSq;
            var std = Tensor.Sqrt(sumSq) + _epsilon;
            return tensor - grad * _learningRate / std;
        }
    }
}