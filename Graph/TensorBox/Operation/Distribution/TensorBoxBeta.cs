namespace SimpleNN.Graph
{
    using SimpleNN.Graph.Functions;
    public partial class TensorBox
    {
        public static TensorBox LBeta(TensorBox a, TensorBox b)
        {
            return Apply(a, b, LBetaFunction.Forward);
        }
        public static TensorBox BetaExp(TensorBox a, TensorBox b)
        {
            return Apply(a, b, BetaExpFunction.Forward);
        }
        public static TensorBox BetaMode(TensorBox a, TensorBox b)
        {
            return Apply(a, b, BetaModeFunction.Forward);
        }
        public static TensorBox BetaEntropy(TensorBox a, TensorBox b)
        {
            return Apply(a, b, BetaEntropyFunction.Forward);
        }
        public static TensorBox BetaLogprob(TensorBox a, TensorBox b, TensorBox x)
        {
            return Apply(a, b, x, BetaLogprobFunction.Forward);
        }
    }
}