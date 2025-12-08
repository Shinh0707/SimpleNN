namespace SimpleNN.Tensor
{
    public partial class Tensor
    {
        /// <summary>
        /// 指定された次元に沿って要素の平均値を計算する.
        /// </summary>
        public static Tensor Mean(Tensor tensor, int dim, bool keepDims = false)
        {
            var sum = Sum(tensor, dim, keepDims);
            var n = tensor.Size[dim];
            return sum / (n < 1 ? 1 : n);
        }
        public static Tensor Mean(Tensor tensor)
        {
            var sum = Sum(tensor);
            var n = tensor.TotalSize;
            return sum / (n < 1 ? 1 : n);
        }
    }
}