namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;
    public class ExpFunction : SingleFunction<ExpFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Exp(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            return new[] { Tensor.Mul_(grad, ctx.Tensor) };
        }
    }
    public class MExpFunction : SingleFunction<MExpFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.MExp(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            return new[] { -Tensor.Mul_(grad, ctx.Tensor) };
        }
    }
}