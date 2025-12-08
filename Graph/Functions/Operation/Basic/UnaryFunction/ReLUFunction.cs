namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;
    public class ReLUFunction : SingleFunction<ReLUFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.ReLU(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var x = ctx.GetInput(0);
            var mask = Tensor.GtEq(x, 0.0f);
            return new[] { grad * mask };
        }
    }
    public class LeakyReLUFunction : KwargsFunction<LeakyReLUFunction, LeakyReLUFunction.Kwargs>
    {
        public class Kwargs
        {
            public float NegativeSlope;
        }
        internal override Tensor Forward(Kwargs kwargs,Context ctx)
        {
            return Tensor.LeakyReLU(ctx.GetInput(0), kwargs.NegativeSlope);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            var x = ctx.GetInput(0);
            var negativeMask = Tensor.Lt(x, 0.0f);
            return new[] { grad * (negativeMask * kwargs.NegativeSlope + (1f - negativeMask)) };
        }
    }
}