namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;
    public class MulFunction : SingleFunction<MulFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Mul_(ctx.GetInput(0), ctx.GetInput(1));
        }
        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            return new[] { grad * ctx.GetInput(1), grad * ctx.GetInput(0) };
        }
    }
    public class MulSclFunction : KwargsFunction<MulSclFunction, MulSclFunction.Kwargs>
    {
        public class Kwargs
        {
            public float Value;
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.Mul_(ctx.GetInput(0), kwargs.Value);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            return new[] { grad * kwargs.Value};
        }
    }
}