namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    /// <summary>
    /// テンソルの2つの次元を入れ替える関数 (転置).
    /// </summary>
    public class TransposeFunction : KwargsFunction<TransposeFunction, TransposeFunction.Kwargs>
    {
        public class Kwargs
        {
            public int dim0;
            public int dim1;
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.Transpose(ctx.GetInput(0), kwargs.dim0, kwargs.dim1);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            // 転置の逆伝播は、同じ次元を再度転置することで元の形状に戻る
            // (A^T)^T = A
            return new[] { Tensor.Transpose(grad, kwargs.dim0, kwargs.dim1) };
        }
    }
}