namespace SimpleNN.Util
{
    using System.Numerics;
    public static partial class ArrayExt
    {
        public static int IntProd(int[] values)
        {
            int length = values.Length;
            int res = values[0];
            for (int i = 1; i < length; i++)
            {
                res *= values[i];
            }
            return res;
        }
        public static int IntProdReverse(int[] values)
        {
            int length = values.Length;
            int res = values[length-1];
            for (int i = length-2; i >= 0; i--)
            {
                res *= values[i];
            }
            return res;
        }
    }
}