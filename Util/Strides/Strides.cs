using System;

namespace SimpleNN.Util
{
    public static partial class Strides
    {
        /// <summary>
        /// 指定された形状に対するC-Contiguous（行優先）なストライドを生成して返す.
        /// </summary>
        /// <param name="size">テンソルの形状</param>
        /// <returns>計算されたストライド配列</returns>
        public static int[] CStrides(int[] size)
        {
            var strides = new int[size.Length];
            ToCStrides(size, strides);
            return strides;
        }

        /// <summary>
        /// ストライドの順序（軸の順列）から, メモリ充填順序（Fill Order）を計算する.
        /// 通常, ストライドの昇順（ストライドが小さい次元＝メモリ上で近い次元）のインデックス列を指す.
        /// </summary>
        /// <param name="strideOrder">ストライドに基づく軸の順序</param>
        /// <returns>充填順序</returns>
        public static int[] FillOrder(int[] strideOrder)
        {
            // ArrayDPの機能を用いて, 順序をソートして返す
            return ArrayExt.ArgMin(strideOrder);
        }

        /// <summary>
        /// ストライド配列から, 軸の順序（Stride Order）を計算する.
        /// C-Contiguousの場合, ストライドが大きい順（外側から内側）に並ぶ.
        /// </summary>
        /// <param name="strides">ストライド配列</param>
        /// <returns>ストライドの降順（大きい順）に対応する軸インデックスの配列</returns>
        public static int[] StrideOrder(int[] strides)
        {
            // ストライドが大きい次元順（降順）のインデックスを取得する
            return ArrayExt.ArgMax(strides);
        }

        /// <summary>
        /// 指定された形状に基づいてC-Contiguousなストライドを計算し, 引数の配列に格納する.
        /// </summary>
        /// <param name="size">テンソルの形状</param>
        /// <param name="strides">結果を格納する配列（sizeと同じ長さであること）</param>
        public static void ToCStrides(int[] size, int[] strides)
        {
            // size.length == strides.lengthは満たされている前提
            int n = size.Length;
            if (n == 0) return;

            int currentStride = 1;
            
            // 後ろの次元から順にストライドを計算する
            for (int i = n - 1; i >= 0; i--)
            {
                if (size[i] <= 1) continue;
                strides[i] = currentStride;
                currentStride *= size[i];
            }
        }

        /// <summary>
        /// ストライドが行優先: 降順の形式になっているか判定する.
        /// </summary>
        /// <param name="strides">判定するストライド配列</param>
        /// <returns>Cスタイル(降順)であればtrue</returns>
        public static bool IsCStrides(int[] strides)
        {
            int len = strides.Length;
            if (len <= 1) return true;

            for (int i = 0; i < len - 1; i++)
            {
                if (strides[i] < strides[i + 1])
                {
                    return false;
                }
            }
            return true;
        }
        /// <summary>
        /// テンソルがメモリ上でC-Contiguous（行優先かつ隙間なく連続）であるか判定する.
        /// </summary>
        /// <param name="size">テンソルの形状</param>
        /// <param name="strides">判定するストライド配列</param>
        /// <returns>C-Contiguousであればtrue</returns>
        public static bool IsCContiguous(int[] size, int[] strides)
        {
            if (size.Length != strides.Length)
            {
                return false;
            }
            int ndim = size.Length;
            long currentStride = 1;
            for (int i = ndim - 1; i >= 0; i--)
            {
                if (size[i] <= 1) continue;
                if (strides[i] != currentStride)
                {
                    return false;
                }
                currentStride *= Math.Max(size[i],1);
            }
            return true;
        }
    }
}