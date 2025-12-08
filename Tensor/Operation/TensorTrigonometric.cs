using System;

namespace SimpleNN.Tensor
{
    public partial class Tensor
    {
        public static Tensor Sin(Tensor x)
        {
            return ApplyMap(x, x => MathF.Sin(x % Util.Constants.TWO_PI));
        }
        public static Tensor Cos(Tensor x)
        {
            return ApplyMap(x, x => MathF.Cos(x % Util.Constants.TWO_PI));
        }
        public static Tensor Tan(Tensor x)
        {
            return ApplyMap(x, x => MathF.Tan(x % Util.Constants.TWO_PI));
        }
        public static Tensor Csc(Tensor a)
        {
            return ApplyMap(a, x => 1.0f/MathF.Sin(x % Util.Constants.TWO_PI));
        }
        public static Tensor Sec(Tensor a)
        {
            return ApplyMap(a, x => 1.0f/MathF.Cos(x % Util.Constants.TWO_PI));
        }
        public static Tensor Cot(Tensor a)
        {
            return ApplyMap(a, x => 1.0f/MathF.Tan(x % Util.Constants.TWO_PI));
        }
    }
}