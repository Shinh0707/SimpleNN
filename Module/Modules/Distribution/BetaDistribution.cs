namespace SimpleNN.Module
{
    using SimpleNN.Graph;
    public class BetaDistibution : Module
    {
        private TensorBox _alpha;
        private TensorBox _beta;
        public BetaDistibution(TensorBox alpha, TensorBox beta)
        {
            (_alpha, _beta) = TensorBox.BroadcastBoth(alpha, beta);
            var param = AddParameters(_alpha, _beta);
            _alpha = param[0];
            _beta = param[1];
        }
        public TensorBox LogProb(TensorBox x) => TensorBox.BetaLogprob(_alpha, _beta, x);
        public TensorBox Exp => TensorBox.BetaExp(_alpha, _beta);
        public TensorBox Mode => TensorBox.BetaMode(_alpha, _beta);
        public TensorBox Entropy => TensorBox.BetaEntropy(_alpha, _beta);
    }
}