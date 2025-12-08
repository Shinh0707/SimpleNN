namespace SimpleNN.Module
{
    using SimpleNN.Graph;
    public class ReLU : SingleInSingleOutModule
    {
        public override TensorBox Forward(TensorBox inputTensor)
        {
            return inputTensor.ReLU();
        }
    }
     public class LeakyReLU : SingleInSingleOutModule
    {
        private float _negativeSlope = 0.01f;
        public LeakyReLU(float negativeSlope = 0.01f)
        {
            _negativeSlope = negativeSlope;
        }
        public override TensorBox Forward(TensorBox inputTensor)
        {
            return inputTensor.LeakyReLU(_negativeSlope);
        }
    }
    public class Sigmoid : SingleInSingleOutModule
    {
        public override TensorBox Forward(TensorBox inputTensor)
        {
            return inputTensor.Sigmoid();
        }
    }
    public class Tanh : SingleInSingleOutModule
    {
        public override TensorBox Forward(TensorBox inputTensor)
        {
            return inputTensor.Tanh();
        }
    }
    public class Sin : SingleInSingleOutModule
    {
        public override TensorBox Forward(TensorBox inputTensor)
        {
            return inputTensor.Sin();
        }
    }
    public class ShiftedSin : SingleInSingleOutModule
    {
        private float _amp;
        private float _bias;
        public ShiftedSin(float amp = 1.0f, float bias = 0.0f)
        {
            _amp = amp;
            _bias = bias;
        }
        public override TensorBox Forward(TensorBox inputTensor)
        {
            return inputTensor.Sin() * _amp + _bias;
        }
    }
    public class Sech : SingleInSingleOutModule
    {
        public override TensorBox Forward(TensorBox inputTensor)
        {
            return inputTensor.Sech();
        }
    }
}