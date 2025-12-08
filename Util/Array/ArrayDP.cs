namespace SimpleNN.Util
{
    using System;
    public static partial class ArrayExt
    {
        /// <summary>
        /// 配列の値を降順にソートした際のインデックス配列を返す (ArgSort Descending).
        /// </summary>
        /// <param name="values">値の配列</param>
        /// <returns>降順ソート後のインデックス配列</returns>
        public static int[] ArgMax(int[] values)
        {
            int length = values.Length;
            int[] indices = new int[length];
            for (int i = 0; i < length; i++)
            {
                indices[i] = i;
            }

            // カスタム比較でインデックスをソートする (降順: b - a)
            Array.Sort(indices, (a, b) => values[b].CompareTo(values[a]));
            
            return indices;
        }

        /// <summary>
        /// 配列の値を昇順にソートした際のインデックス配列を返す (ArgSort Ascending).
        /// </summary>
        /// <param name="values">値の配列</param>
        /// <returns>昇順ソート後のインデックス配列</returns>
        public static int[] ArgMin(int[] values)
        {
            int length = values.Length;
            int[] indices = new int[length];
            for (int i = 0; i < length; i++)
            {
                indices[i] = i;
            }

            // カスタム比較でインデックスをソートする (昇順: a - b)
            Array.Sort(indices, (a, b) => values[a].CompareTo(values[b]));

            return indices;
        }

        /// <summary>
        /// 指定された順序に従って配列を並べ替えた新しい配列を返す.
        /// </summary>
        /// <param name="order">並べ替え順序を示すインデックス配列</param>
        /// <param name="values">並べ替える対象の配列</param>
        /// <returns>並べ替えられた新しい配列</returns>
        /// <exception cref="ArgumentException">orderとvaluesの長さが異なる場合</exception>
        public static int[] OrderSorted(int[] order, int[] values)
        {
            if (order.Length != values.Length) throw new ArgumentException($"Order and values length mismatch: {order.Length} vs {values.Length}");
            
            var sortedValues = new int[values.Length];
            for(int i = 0; i < order.Length; i++)
            {
                sortedValues[order[i]] = values[i];
            }
            return sortedValues;
        }
    }
}