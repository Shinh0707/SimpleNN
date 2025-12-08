namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    /// <summary>
    /// 指定した次元にサイズ1の軸を挿入する関数.
    /// </summary>
    public class UnsqueezeFunction : KwargsFunction<UnsqueezeFunction, UnsqueezeFunction.Kwargs>
    {
        public class Kwargs
        {
            public int dim;
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.Unsqueeze(ctx.GetInput(0), kwargs.dim);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            var inputSize = ctx.GetInput(0).Size;
            return new[] { Tensor.Reshape(grad, inputSize) };
        }
    }
}