namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    public class SubFunction : SingleFunction<SubFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Sub_(ctx.GetInput(0), ctx.GetInput(1));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            return new[] { grad, Tensor.Neg(grad) };
        }
    }
    public class SubRSclFunction : KwargsFunction<SubRSclFunction, SubRSclFunction.Kwargs>
    {
        public class Kwargs
        {
            public float Value;
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.Sub_(ctx.GetInput(0), kwargs.Value);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            return new[] { grad };
        }
    }
    public class SubLSclFunction : KwargsFunction<SubLSclFunction, SubLSclFunction.Kwargs>
    {
        public class Kwargs
        {
            public float Value;
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.Sub_(kwargs.Value, ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            return new[] { Tensor.Neg(grad)};
        }
    }
}