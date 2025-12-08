namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;
    public class SigmoidFunction : SingleFunction<SigmoidFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Sigmoid(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var y = ctx.Tensor;
            var oneMinusY = 1.0f - y;
            return new[] { grad * y * oneMinusY };
        }
    }
    public class SoftplusFunction : SingleFunction<SoftplusFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Softplus(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var x = ctx.Tensor;
            return new[] { grad * Tensor.Sigmoid(x) };
        }
    }
}