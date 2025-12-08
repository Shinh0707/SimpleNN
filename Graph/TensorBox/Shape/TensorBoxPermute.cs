namespace SimpleNN.Graph
{
    using SimpleNN.Tensor;
    using SimpleNN.Graph.Functions;

    public partial class TensorBox
    {
        /// <summary>
        /// テンソルの次元を入れ替えた新しいTensorBoxを返す.
        /// </summary>
        /// <param name="dims">新しい次元の順序</param>
        /// <returns>次元が入れ替えられたTensorBox</returns>
        public TensorBox Permute(params int[] dims)
        {
            var kwargs = new PermuteFunction.Kwargs { dims = dims };
            return new TensorBox(PermuteFunction.Forward(kwargs, _ctx));
        }
    }
}
