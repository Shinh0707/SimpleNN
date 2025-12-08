namespace SimpleNN.Graph
{
    using SimpleNN.Graph.Functions;
    using SimpleNN.Tensor;
    public partial class TensorBox
    {
        public static TensorBox GammaSample(TensorBox a, TensorBox b)
        {
            var eps = Tensor.Normal(a.Size);
            var a9 = 9f * a;
            var x = (a / b) * (((1f - 1f/a9) + eps/(a9.Sqrt())).Cube());
            return Apply(a, b, LBetaFunction.Forward);
        }
    }
}