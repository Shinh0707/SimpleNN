namespace SimpleNN.Optimizer
{
    using System;
    using System.Collections.Generic;
    using SimpleNN.Graph;
    using SimpleNN.Tensor;

    /// <summary>
    /// RAdam (Rectified Adam) オプティマイザ.
    /// </summary>
    public class RAdam : Optimizer
    {
        private readonly float _learningRate;
        private readonly float _beta1;
        private readonly float _beta2;
        private readonly float _epsilon;
        
        private int _step = 0;
        private readonly (Tensor m, Tensor v)[] _moments;

        public RAdam(
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

        public RAdam(float learningRate, params List<TensorBox>[] parameters)
            : this(learningRate, 0.9f, 0.999f, 1e-8f, parameters)
        {
        }

        public override void PreStep()
        {
            _step++;
        }

        protected override Tensor StepGrad(int id, Tensor tensor, Tensor grad)
        {
            var state = _moments[id];
            
            if (state.m is null)
            {
                state.m = grad * 0.0f;
                state.v = grad * 0.0f;
            }

            var m = state.m;
            var v = state.v;

            // モーメントの更新
            m = (m * _beta1) + (grad * (1.0f - _beta1));
            v = (v * _beta2) + (Tensor.Square(grad) * (1.0f - _beta2));
            
            _moments[id] = (m, v);

            float rhoInf = 2.0f / (1.0f - _beta2) - 1.0f;
            float beta2Pow = (float)Math.Pow(_beta2, _step);
            float rhoT = rhoInf - 2.0f * _step * beta2Pow / (1.0f - beta2Pow);

            float biasCorrection1 = 1.0f - (float)Math.Pow(_beta1, _step);
            var mHat = m / biasCorrection1;
            if (rhoT > 4.0f)
            {
                float rectNumerator = (rhoT - 4.0f) * (rhoT - 2.0f) * rhoInf;
                float rectDenominator = (rhoInf - 4.0f) * (rhoInf - 2.0f) * rhoT;
                float rectification = (float)Math.Sqrt(rectNumerator / rectDenominator);

                float biasCorrection2 = 1.0f - beta2Pow;
                var vHat = v / biasCorrection2;
                var vHatSqrt = Tensor.Sqrt(vHat);
                return tensor - (mHat * (_learningRate * rectification)) / (vHatSqrt + _epsilon);
            }
            else
            {
                return tensor - (mHat * _learningRate);
            }
        }
    }
}