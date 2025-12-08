namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;
    public class AbsFunction : SingleFunction<AbsFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Abs(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var x = ctx.GetInput(0);
            var sign = Tensor.Sign(x);
            return new[] { grad * sign };
        }
    }
    public class NegFunction : SingleFunction<NegFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Neg(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            return new[] { Tensor.Neg(grad) };
        }
    }
}