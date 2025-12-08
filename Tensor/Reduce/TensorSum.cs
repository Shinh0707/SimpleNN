using System;

namespace SimpleNN.Tensor
{
    public partial class Tensor
    {
        /// <summary>
        /// 指定された次元に沿って要素を合計する.
        /// </summary>
        /// <param name="tensor">入力テンソル</param>
        /// <param name="dim">合計する次元</param>
        /// <param name="keepDims">次元を保持するかどうか</param>
        /// <returns>合計されたテンソル</returns>
        public static Tensor Sum(Tensor tensor, int dim, bool keepDims = false)
        {
            return ReduceFunction(tensor, dim, keepDims, (float inData, float currentValue) =>
            {
                return currentValue + inData;
            });
        }
        public static Tensor Sum(Tensor tensor)
        {
            return ReduceFunction(tensor, (float inData, float currentValue) =>
            {
                return currentValue + inData;
            });
        }
    }
}