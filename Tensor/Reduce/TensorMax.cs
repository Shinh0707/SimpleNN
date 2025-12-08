using System;

namespace SimpleNN.Tensor
{
    public partial class Tensor
    {
        /// <summary>
        /// 指定された次元に沿って要素の最大値を求める.
        /// </summary>
        /// <param name="tensor">入力テンソル</param>
        /// <param name="dim">計算する次元</param>
        /// <param name="keepDims">次元を保持するかどうか</param>
        /// <returns>最大値のテンソル</returns>
        public static Tensor Max(Tensor tensor, int dim, bool keepDims = false)
        {
            return ReduceFunction(tensor, dim, keepDims, (float inData, float currentValue) =>
            {
                if (inData > currentValue)
                {
                    return inData;
                }
                return currentValue;
            }, float.NegativeInfinity);
        }
        public static Tensor Max(Tensor tensor)
        {
            return ReduceFunction(tensor, (float inData, float currentValue) =>
            {
                if (inData > currentValue)
                {
                    return inData;
                }
                return currentValue;
            }, float.NegativeInfinity);
        }
    }
}