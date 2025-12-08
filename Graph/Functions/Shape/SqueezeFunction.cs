namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    /// <summary>
    /// 指定した次元のサイズが1の場合にその次元を削除する関数.
    /// </summary>
    public class SqueezeFunction : KwargsFunction<SqueezeFunction, SqueezeFunction.Kwargs>
    {
        public class Kwargs
        {
            public int dim;
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.Squeeze(ctx.GetInput(0), kwargs.dim);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            // 逆伝播では、勾配を入力テンソルの形状に戻す
            var inputSize = ctx.GetInput(0).Size;
            return new[] { Tensor.Reshape(grad, inputSize) };
        }
    }
}