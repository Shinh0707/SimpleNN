namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    /// <summary>
    /// テンソルの次元を入れ替える関数.
    /// </summary>
    public class PermuteFunction : KwargsFunction<PermuteFunction, PermuteFunction.Kwargs>
    {
        public class Kwargs
        {
            public int[] dims;
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.Permute(ctx.GetInput(0), kwargs.dims);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            // 逆伝播は逆置換を行う
            // dims[i] = k のとき、元のi番目の次元は新しいk番目の次元に移動している
            // 戻すには、新しいk番目の次元をi番目に戻す必要がある
            // つまり、inv[k] = i となる inv を作成して Permute する
            
            int ndim = kwargs.dims.Length;
            int[] inv = new int[ndim];
            for (int i = 0; i < ndim; i++)
            {
                inv[kwargs.dims[i]] = i;
            }

            return new[] { Tensor.Permute(grad, inv) };
        }
    }
}
