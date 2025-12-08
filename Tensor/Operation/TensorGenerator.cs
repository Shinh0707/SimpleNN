namespace SimpleNN.Tensor
{
    using System;
    using SimpleNN.Util;

    public partial class Tensor
    {
        public static Tensor Fill(int[] size, float value, int[] strides = null)
        {
            int n = ArrayExt.IntProd(size);
            var data = new float[n];
            Array.Fill(data, value);
            return (strides == null) ? new(data, size) : new(data, size, strides);
        }
        public static Tensor FillLike(Tensor t, float value) => Fill(t.Size, value, t.Strides);
        public static Tensor Ones(int[] size, int[] strides = null) => Fill(size, 1f, strides);
        public static Tensor OnesLike(Tensor t) => Ones(t.Size, t.Strides);
        public static Tensor Zeros(int[] size, int[] strides = null) => Fill(size, 0f, strides);
        public static Tensor ZerosLike(Tensor t) => Zeros(t.Size, t.Strides);
        public static Tensor Random(int[] size, float min = 0.0f, float max = 1.0f, int[] strides = null)
        {
            int n = ArrayExt.IntProd(size);
            var data = ArrayExt.Random(n, min, max);
            return (strides == null) ? new(data, size) : new(data, size, strides);
        }
        public static Tensor RandomLike(Tensor t, float min = 0.0f, float max = 1.0f) => Random(t.Size, min, max, t.Strides);
        public static Tensor Normal(int[] size, float loc = 0.0f, float std = 1.0f, int[] strides = null)
        {
            int n = ArrayExt.IntProd(size);
            var data = ArrayExt.Normal(n, loc, std);
            return (strides == null) ? new(data, size) : new(data, size, strides);
        }
        public static Tensor NormalLike(Tensor t, float loc = 0.0f, float std = 1.0f) => Normal(t.Size, loc, std, t.Strides);
    }
}