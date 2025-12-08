namespace SimpleNN.Graph
{
    using SimpleNN.Graph.Functions;
    public partial class TensorBox
    {
        public static TensorBox Softmax(TensorBox tensor, int dim)
        {
            var reducedMax = tensor - tensor.Max(dim, true);
            var exprm = reducedMax.MExp();
            var summed = exprm.Sum(dim, true);
            return exprm / summed;
        }
        public TensorBox Softmax(int dim) => Softmax(this, dim);
    }
}