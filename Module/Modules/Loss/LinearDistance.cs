namespace SimpleNN.Module
{
    using SimpleNN.Graph;
    /// <summary>
    /// 平均絶対誤差 (L1 Loss) 損失関数.
    /// Loss = Mean(|Input - Target|)
    /// </summary>
    public class L1Loss : Loss
    {
        public override TensorBox Forward(TensorBox input, TensorBox target)
        {
            var diff = input - target;
            return diff.Abs().Mean();
        }
    }
    /// <summary>
    /// 平均二乗誤差 (Mean Squared Error) 損失関数.
    /// Loss = Mean((Input - Target)^2)
    /// </summary>
    public class MSELoss : Loss
    {
        public override TensorBox Forward(TensorBox input, TensorBox target)
        {
            var diff = input - target;
            return diff.Square().Mean();
        }
    }
    /// <summary>
    /// 平均二乗誤差 (Mean Squared Error) のLog損失関数.
    /// Loss = Log(Mean((Input - Target)^2))
    /// </summary>
    public class LogMSELoss : Loss
    {
        private float _eps = 1e-4f;
        public LogMSELoss(float eps = 1e-4f) : base()
        {
            _eps = eps;
        }
        public override TensorBox Forward(TensorBox input, TensorBox target)
        {
            var diff = input - target;
            return (diff.Square().Mean() + _eps).Log();
        }
    }
}