using System;

namespace SimpleNN.Tensor
{
    public partial class Tensor
    {
        /// <summary>
        /// テンソルの形状を変更した新しいテンソルを返す.
        /// 可能であればデータを共有するViewを返し、メモリ上で不連続な場合はデータを整列させたコピーを作成する.
        /// </summary>
        /// <param name="tensor">入力テンソル</param>
        /// <param name="newSize">新しい形状. -1を指定すると、他の次元からサイズを自動推論する.</param>
        /// <returns>新しい形状を持つテンソル</returns>
        /// <exception cref="ArgumentException">要素数が一致しない場合</exception>
        public static Tensor Reshape(Tensor tensor, int[] newSize)
        {
            uint targetLen = 1;
            int inferIndex = -1;

            // サイズ推論とターゲット要素数の計算
            // 配列をコピーして作業用変数を作成
            int[] finalSize = new int[newSize.Length];
            
            for (int i = 0; i < newSize.Length; i++)
            {
                int s = newSize[i];
                if (s == -1)
                {
                    if (inferIndex != -1) throw new ArgumentException("Only one dimension can be -1");
                    inferIndex = i;
                }
                else
                {
                    if (s < 0) throw new ArgumentException($"Invalid dimension size: {s}");
                    targetLen *= (uint)s;
                    finalSize[i] = s;
                }
            }

            var totalElements = tensor.TotalSize;

            // -1 の次元サイズを解決
            if (inferIndex != -1)
            {
                if (totalElements % targetLen != 0)
                    throw new ArgumentException($"Cannot reshape tensor of size {totalElements} into shape with multiple {targetLen}");
                
                var inferredSize = totalElements / targetLen;
                finalSize[inferIndex] = (int)inferredSize;
                targetLen *= (uint)inferredSize;
            }

            if (targetLen != totalElements)
            {
                throw new ArgumentException($"Cannot reshape tensor of size {totalElements} (={tensor}) into shape {string.Join(",", finalSize)}");
            }

            // ケース1: 入力がContiguousなら、データを共有して新しいView（ストライド再計算）を作る
            if (tensor.IsContiguous)
            {
                // C-Contiguousなストライドを計算して新しいTensorを返す
                // データ配列(_data)は共有される
                return new Tensor(tensor._data, finalSize);
            }

            // ケース2: 入力が不連続（Broadcast後など）なら、データをコピー・整列（Contiguous化）して返す
            // GetContiguousData() は内部で物理的なコピーを作成する
            return new Tensor(tensor.GetContiguousData(), finalSize);
        }

        /// <summary>
        /// 指定した位置にサイズ1の次元を挿入した新しいテンソルを返す.
        /// </summary>
        /// <param name="tensor">入力テンソル</param>
        /// <param name="dim">挿入する次元のインデックス</param>
        /// <returns>形状変更後のテンソル</returns>
        public static Tensor Unsqueeze(Tensor tensor, int dim)
        {
            int ndim = tensor.NDim;
            if (dim < 0) dim += ndim + 1;
            if (dim < 0 || dim > ndim) throw new ArgumentOutOfRangeException(nameof(dim));

            var newSize = new int[ndim + 1];

            // 配列コピーによるサイズ構築
            if (dim > 0) Array.Copy(tensor.Size, 0, newSize, 0, dim);
            newSize[dim] = 1;
            if (dim < ndim) Array.Copy(tensor.Size, dim, newSize, dim + 1, ndim - dim);

            // Reshapeに委譲
            return Reshape(tensor, newSize);
        }

        /// <summary>
        /// 指定した次元のサイズが1であれば削除した新しいテンソルを返す.
        /// サイズが1でない場合は元のテンソルをそのまま返す.
        /// </summary>
        /// <param name="tensor">入力テンソル</param>
        /// <param name="dim">削除対象の次元インデックス</param>
        /// <returns>形状変更後のテンソル</returns>
        public static Tensor Squeeze(Tensor tensor, int dim)
        {
            int ndim = tensor.NDim;
            if (dim < 0) dim += ndim;

            if (dim < 0 || dim >= ndim)
            {
                throw new ArgumentOutOfRangeException(nameof(dim));
            }

            // 指定された次元が1でない場合は、何もしない
            if (tensor.Size[dim] != 1)
            {
                return tensor;
            }

            var newSize = new int[ndim - 1];

            // dimより前の部分をコピー
            if (dim > 0)
            {
                Array.Copy(tensor.Size, 0, newSize, 0, dim);
            }

            // dimより後の部分をコピー
            if (dim < ndim - 1)
            {
                Array.Copy(tensor.Size, dim + 1, newSize, dim, ndim - dim - 1);
            }

            // Reshapeに委譲
            return Reshape(tensor, newSize);
        }

        /// <summary>
        /// サイズが1であるすべての次元を削除した新しいテンソルを返す.
        /// </summary>
        /// <param name="tensor">入力テンソル</param>
        /// <returns>形状変更後のテンソル</returns>
        public static Tensor Squeeze(Tensor tensor)
        {
            int ndim = tensor.NDim;
            int newCount = 0;

            // サイズが1でない次元数をカウント
            for (int i = 0; i < ndim; i++)
            {
                if (tensor.Size[i] != 1)
                {
                    newCount++;
                }
            }

            // 変更がない（サイズ1の次元がない）場合は元のテンソルを返す
            if (newCount == ndim)
            {
                return tensor;
            }

            var newSize = new int[newCount];
            int index = 0;

            // サイズが1でない次元のみを抽出して新しいサイズ配列を作成
            for (int i = 0; i < ndim; i++)
            {
                if (tensor.Size[i] != 1)
                {
                    newSize[index] = tensor.Size[i];
                    index++;
                }
            }

            // Reshapeに委譲
            return Reshape(tensor, newSize);
        }

        /// <summary>
        /// テンソルを1次元のベクトルに平坦化する.
        /// (startIndex=0, endIndex=-1 と等価)
        /// 可能であればデータを共有するViewを返し、メモリ上で不連続な場合はデータを整列させたコピーを作成する.
        /// </summary>
        /// <param name="tensor">入力テンソル</param>
        /// <returns>1次元のテンソル</returns>
        public static Tensor Flatten(Tensor tensor)
        {
            // 0から最後まで平坦化する (全体平坦化)
            return Flatten(tensor, 0, -1);
        }

        /// <summary>
        /// テンソルの指定した範囲の次元を平坦化する.
        /// 可能であればデータを共有するViewを返し、メモリ上で不連続な場合はデータを整列させたコピーを作成する.
        /// </summary>
        /// <param name="tensor">入力テンソル</param>
        /// <param name="startIndex">平坦化を開始する次元インデックス (0から始まる). 負の値は末尾から数える.</param>
        /// <param name="endIndex">平坦化を終了する次元インデックス (0から始まる). 負の値は末尾から数える. -1は最後の次元を意味する.</param>
        /// <returns>平坦化されたテンソル</returns>
        /// <exception cref="ArgumentOutOfRangeException">インデックスが範囲外の場合</exception>
        /// <exception cref="ArgumentException">startIndexがendIndexより大きい場合</exception>
        public static Tensor Flatten(Tensor tensor, int startIndex, int endIndex = -1)
        {
            int ndim = tensor.NDim;

            // 1. 引数の正規化 (負のインデックスを解決)
            int start = startIndex;
            int end = endIndex;

            if (start < 0) start += ndim;
            if (end < 0) end += ndim;

            if (start < 0 || start >= ndim)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(startIndex), 
                    $"Invalid startIndex {startIndex} for tensor with {ndim} dims."
                );
            }
            if (end < 0 || end >= ndim)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(endIndex), 
                    $"Invalid endIndex {endIndex} for tensor with {ndim} dims."
                );
            }
            
            if (start > end)
            {
                throw new ArgumentException(
                    $"startIndex {startIndex} must be <= endIndex {endIndex}"
                );
            }

            // 2. 新しい形状の計算
            
            // ケース1: テンソル全体を平坦化 (start=0, end=ndim-1)
            if (start == 0 && end == ndim - 1)
            {
                // Reshape(tensor, new int[] { -1 }) に任せるのが最適
                return Reshape(tensor, new int[] { -1 });
            }

            // ケース2: それ以外
            // 新しい次元数を計算
            // [0..start-1] (start個) 
            // + [start..end] (1個) 
            // + [end+1..ndim-1] (ndim-1-end個)
            int newNdim = (start) + 1 + (ndim - 1 - end);
            int[] newSize = new int[newNdim];
            
            int newIdx = 0;

            // [0, startIndex - 1] までをコピー
            for (int i = 0; i < start; i++)
            {
                newSize[newIdx] = tensor.Size[i];
                newIdx++;
            }

            // [startIndex, endIndex] を平坦化
            long flattenedDimSize = 1; 
            for (int i = start; i <= end; i++)
            {
                flattenedDimSize *= tensor.Size[i];
            }
            newSize[newIdx] = (int)flattenedDimSize;
            newIdx++;
            
            // [endIndex + 1, ndim - 1] までをコピー
            for (int i = end + 1; i < ndim; i++)
            {
                newSize[newIdx] = tensor.Size[i];
                newIdx++;
            }

            // 3. Reshape の呼び出し
            return Reshape(tensor, newSize);
        }

        /// <summary>
        /// Stridesを操作することで転置を行う. データはコピーされない(View).
        /// </summary>
        /// <param name="tensor">入力テンソル</param>
        /// <param name="dim0">入れ替える次元1</param>
        /// <param name="dim1">入れ替える次元2</param>
        /// <returns>転置されたテンソル</returns>
        public static Tensor Transpose(Tensor tensor, int dim0, int dim1)
        {
            int ndim = tensor.NDim;
            if (dim0 < 0) dim0 += ndim;
            if (dim1 < 0) dim1 += ndim;

            if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim)
            {
                throw new ArgumentOutOfRangeException($"Dimension out of range. dim0:{dim0}, dim1:{dim1}, ndim:{ndim}");
            }

            if (dim0 == dim1)
            {
                return tensor;
            }

            // サイズとストライドをコピーして入れ替える
            int[] newSize = (int[])tensor.Size.Clone();
            int[] newStrides = (int[])tensor.Strides.Clone();

            (newSize[dim1], newSize[dim0]) = (newSize[dim0], newSize[dim1]);
            (newStrides[dim1], newStrides[dim0]) = (newStrides[dim0], newStrides[dim1]);

            // データ共有で新しいTensor(View)を作成
            return new Tensor(tensor._data, newSize, newStrides);
        }
    }
}