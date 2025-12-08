using System;

namespace SimpleNN.Tensor
{
    public partial class Tensor
    {
        /// <summary>
        /// テンソルの次元を入れ替えた新しいテンソルを返す.
        /// データはコピーされず、ストライドの変更のみで実現される(View).
        /// </summary>
        /// <param name="tensor">入力テンソル</param>
        /// <param name="dims">新しい次元の順序 (例: {2, 0, 1})</param>
        /// <returns>次元が入れ替えられたテンソル</returns>
        /// <exception cref="ArgumentException">dimsが不正な場合</exception>
        public static Tensor Permute(Tensor tensor, params int[] dims)
        {
            int ndim = tensor.NDim;

            // 1. 引数チェック
            if (dims.Length != ndim)
            {
                throw new ArgumentException($"Permute dimensions must match tensor dimension. Tensor: {ndim}, Permute: {dims.Length}");
            }

            // dimsが 0 ~ ndim-1 の順列になっているか確認
            // 重複チェックと範囲チェック
            var check = new bool[ndim];
            for (int i = 0; i < ndim; i++)
            {
                int d = dims[i];
                if (d < 0 || d >= ndim)
                {
                    throw new ArgumentException($"Dimension out of range: {d}");
                }
                if (check[d])
                {
                    throw new ArgumentException($"Duplicate dimension: {d}");
                }
                check[d] = true;
            }

            // 2. 新しいサイズとストライドを作成
            int[] newSize = new int[ndim];
            int[] newStrides = new int[ndim];

            for (int i = 0; i < ndim; i++)
            {
                int originalDim = dims[i];
                newSize[i] = tensor.Size[originalDim];
                newStrides[i] = tensor.Strides[originalDim];
            }

            // 3. 新しいTensorを作成して返す (データ共有)
            return new Tensor(tensor._data, newSize, newStrides);
        }
    }
}
