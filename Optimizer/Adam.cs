namespace SimpleNN.Optimizer
{
    using System;
    using System.Collections.Generic;
    using SimpleNN.Graph;
    using SimpleNN.Tensor;

    /// <summary>
    /// Adam (Adaptive Moment Estimation) オプティマイザ.
    /// </summary>
    public class Adam : Optimizer
    {
        private readonly float _learningRate;
        private readonly float _beta1;
        private readonly float _beta2;
        private readonly float _epsilon;
        
        private int _step = 0;
        
        // (m: 1次モーメント, v: 2次モーメント) の配列
        private readonly (Tensor m, Tensor v)[] _moments;

        public Adam(
            float learningRate, 
            float beta1, 
            float beta2, 
            float epsilon, 
            params List<TensorBox>[] parameters) : base(parameters)
        {
            _learningRate = learningRate;
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;
            _moments = new (Tensor m, Tensor v)[NumParameters];
        }

        public Adam(float learningRate, params List<TensorBox>[] parameters)
            : this(learningRate, 0.9f, 0.999f, 1e-8f, parameters)
        {
        }

        /// <summary>
        /// ステップ数を更新し, 全パラメータの更新を実行する.
        /// </summary>
        public override void PreStep()
        {
            _step++;
        }

        protected override Tensor StepGrad(int id, Tensor tensor, Tensor grad)
        {
            // 状態取得
            var state = _moments[id];
            if (state.m is null)
            {
                state.m = grad * 0.0f;
                state.v = grad * 0.0f;
            }

            var m = state.m;
            var v = state.v;
            m = (m * _beta1) + (grad * (1.0f - _beta1));
            v = (v * _beta2) + (Tensor.Square(grad) * (1.0f - _beta2));
            _moments[id] = (m, v);

            // バイアス補正
            float biasCorrection1 = 1.0f - (float)Math.Pow(_beta1, _step);
            float biasCorrection2 = 1.0f - (float)Math.Pow(_beta2, _step);

            var mHat = m / biasCorrection1;
            var vHat = v / biasCorrection2;
            var vHatSqrt = Tensor.Sqrt(vHat);
            return tensor - (mHat * _learningRate) / (vHatSqrt + _epsilon);
        }
    }
}