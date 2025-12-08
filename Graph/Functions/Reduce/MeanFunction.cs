namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    /// <summary>
    /// Mean (Average) Reduce 関数.
    /// </summary>
    public class MeanFunction : ReduceFunction<MeanFunction, MeanFunction.Kwargs>
    {
        public class Kwargs : ReduceKwargs { }
        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            var input = ctx.GetInput(0);
            int n = input.Size[kwargs.Dim(input.NDim)];
            ctx.RegisterDatas(n < 1 ? 1 : n);
            return Tensor.Mean(input, kwargs.dim, kwargs.keepDims);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            var scale = ctx.GetRegisteredData<int>(0);
            return new[] { grad / scale };
        }
    }
    public class ReduceMeanFunction : ReduceFunction<ReduceMeanFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            var n = ctx.GetInput(0).TotalSize;
            ctx.RegisterDatas(n < 1 ? 1 : n);
            return Tensor.Mean(ctx.GetInput(0));
        }
        internal override Tensor[] ReduceBackward(Context ctx, Tensor grad)
        {
            var scale = ctx.GetRegisteredData<ulong>(0);
            return new[] { grad / scale };
        }
    }
}