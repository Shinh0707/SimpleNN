namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    /// <summary>
    /// サイズが1であるすべての次元を削除する関数.
    /// </summary>
    public class AllSqueezeFunction : SingleFunction<AllSqueezeFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Squeeze(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var inputSize = ctx.GetInput(0).Size;
            return new[] { Tensor.Reshape(grad, inputSize) };
        }
    }
}