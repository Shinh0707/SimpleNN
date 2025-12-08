namespace SimpleNN.Graph
{
    using SimpleNN.Graph.Functions;

    public partial class TensorBox
    {
        /// <summary>
        /// 指定された次元の最大値を計算する.
        /// </summary>
        /// <param name="dim">計算する次元</param>
        /// <param name="keepDims">次元を保持するかどうか</param>
        /// <returns>計算結果のTensorBox</returns>
        public TensorBox Max(int dim, bool keepDims = false)
        {
            return new TensorBox(MaxFunction.Forward(
                new()
                {
                    dim = dim,
                    keepDims = keepDims
                },
                _ctx
            ));
        }
        public TensorBox Max()
        {
            return new TensorBox(ReduceMaxFunction.Forward(
                _ctx
            ));
        }
        /// <summary>
        /// 指定された次元の要素の平均値を計算する.
        /// </summary>
        public TensorBox Mean(int dim, bool keepDims = false)
        {
            return new TensorBox(MeanFunction.Forward(
                new()
                {
                    dim = dim,
                    keepDims = keepDims
                },
                _ctx
            ));
        }
        public TensorBox Mean()
        {
            return new TensorBox(ReduceMeanFunction.Forward(
                _ctx
            ));
        }
        /// <summary>
        /// 指定された次元の要素を合計する.
        /// </summary>
        /// <param name="dim">合計する次元</param>
        /// <param name="keepDims">次元を保持するかどうか</param>
        /// <returns>計算結果のTensorBox</returns>
        public TensorBox Sum(int dim, bool keepDims = false) {
            return new TensorBox(SumFunction.Forward(
                new()
                {
                    dim = dim,
                    keepDims = keepDims
                },
                _ctx
            ));
        }
        public TensorBox Sum()
        {
            return new TensorBox(ReduceSumFunction.Forward(
                _ctx
            ));
        }
    }
}