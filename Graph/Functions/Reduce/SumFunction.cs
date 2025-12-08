namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;
    public class SumFunction : ReduceFunction<SumFunction, SumFunction.Kwargs>
    {
        public class Kwargs : ReduceKwargs{}
        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.Sum(ctx.GetInput(0), kwargs.dim, kwargs.keepDims);
        }
        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            return new[]{ grad };
        }
    }
    public class ReduceSumFunction : ReduceFunction<ReduceSumFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Sum(ctx.GetInput(0));
        }
        internal override Tensor[] ReduceBackward(Context _, Tensor grad)
        {
            return new[]{ grad };
        }
    }
}