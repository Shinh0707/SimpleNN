namespace SimpleNN.Util
{
    using System.Collections.Generic;
    public static partial class ArrayExt
    {
        /// <summary>
        /// 2つの配列が同じ内容（順序・要素）を持つか高速に判定する.
        /// </summary>
        /// <remarks>
        /// 要素の比較には EqualityComparer<T>.Default を使用する.
        /// これは T 型が IEquatable<T> を実装していればそれを利用し,
        /// null チェックも安全に行うため, 高速かつ安全である.
        /// </remarks>
        /// <param name="a">比較する配列1</param>
        /// <param name="b">比較する配列2</param>
        /// <typeparam name="T">比較する要素の型</typeparam>
        /// <returns>配列の内容が同一の場合は true, それ以外は false</returns>
        public static bool IsSameArray<T>(T[] a, T[] b)
        {
            if (a == b)
            {
                return true;
            }
            if (a == null || b == null)
            {
                return false;
            }

            int length = a.Length;
            if (length != b.Length)
            {
                return false;
            }
            EqualityComparer<T> comparer = EqualityComparer<T>.Default;

            for (int i = 0; i < length; i++)
            {
                if (!comparer.Equals(a[i], b[i]))
                {
                    return false;
                }
            }
            return true;
        }
    }
}