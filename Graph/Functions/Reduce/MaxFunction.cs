namespace SimpleNN.Graph
{
    using SimpleNN.Graph.Functions;
    using SimpleNN.Tensor;
    /// <summary>
    /// Max (Maximum) Reduce 関数.
    /// </summary>
    public class MaxFunction : ReduceFunction<MaxFunction, MaxFunction.Kwargs>
    {
        public class Kwargs : ReduceKwargs { }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.Max(ctx.GetInput(0), kwargs.dim, kwargs.keepDims);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            var input = ctx.GetInput(0);
            var mask = Tensor.Eq(ctx.Tensor, input);
            return new[] { grad * mask };
        }
    }
    public class ReduceMaxFunction : ReduceFunction<ReduceMaxFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Max(ctx.GetInput(0));
        }
        internal override Tensor[] ReduceBackward(Context ctx, Tensor grad)
        {
            var input = ctx.GetInput(0);
            var mask = Tensor.Eq(ctx.Tensor, input);
            return new[] { grad * mask };
        }
    }
}