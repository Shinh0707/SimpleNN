namespace SimpleNN.Tensor
{
    using System;
    using System.Numerics;
    
    public partial class Tensor
    {
        public static Tensor Maximum_(Tensor a, Tensor b)
        {
            return ApplyBroadcastedOperation(a, b, Vector.Max, MathF.Max);
        }
        public static Tensor Maximum(Tensor a, Tensor b)
        {
            return ApplyBroadcastOperation(a, b, Maximum_);
        }
        public static Tensor Maximum(Tensor a, float b)
        {
            var v = new Vector<float>(b);
            return ApplyMap(a,
                x => Vector.Max(x, v),
                x => MathF.Max(x, b));
        }
        public static Tensor Maximum(float b, Tensor a) => Maximum(a, b);
        public static Tensor Minimum_(Tensor a, Tensor b)
        {
            return ApplyBroadcastedOperation(a, b, Vector.Min, MathF.Min);
        }
        public static Tensor Minimum(Tensor a, Tensor b)
        {
            return ApplyBroadcastOperation(a, b, Minimum_);
        }
        public static Tensor Minimum(Tensor a, float b)
        {
            var v = new Vector<float>(b);
            return ApplyMap(a,
                x => Vector.Min(x, v),
                x => MathF.Min(x, b));
        }
        public static Tensor Minimum(float b, Tensor a) => Minimum(a, b);
        public static Tensor GtEq_(Tensor a, Tensor b)
        {
            return ApplyBroadcastedOperation(a, b,
                (v1, v2) => Vector.GreaterThanOrEqual<float>(v1, v2),
                (f1, f2) => f1 >= f2 ? 1.0f : 0.0f);
        }
        public static Tensor GtEq(Tensor a, Tensor b)
        {
            return ApplyBroadcastOperation(a, b, GtEq_);
        }
        public static Tensor GtEq(Tensor a, float b)
        {
            var v = new Vector<float>(b);
            return ApplyMap(a,
                x => Vector.GreaterThanOrEqual<float>(x, v),
                x => x >= b ? 1.0f : 0.0f);
        }
        public static Tensor GtEq(float b, Tensor a)
        {
            var v = new Vector<float>(b);
            return ApplyMap(a,
                x => Vector.GreaterThanOrEqual<float>(v, x),
                x => b >= x ? 1.0f : 0.0f);
        }
        public static Tensor Gt_(Tensor a, Tensor b)
        {
            return ApplyBroadcastedOperation(a, b,
                (v1, v2) => Vector.GreaterThan<float>(v1, v2),
                (f1, f2) => f1 >= f2 ? 1.0f : 0.0f);
        }
        public static Tensor Gt(Tensor a, Tensor b)
        {
            return ApplyBroadcastOperation(a, b, Gt_);
        }
        public static Tensor Gt(Tensor a, float b)
        {
            var v = new Vector<float>(b);
            return ApplyMap(a,
                x => Vector.GreaterThan<float>(x, v),
                x => x > b ? 1.0f : 0.0f);
        }
        public static Tensor Gt(float b, Tensor a)
        {
            var v = new Vector<float>(b);
            return ApplyMap(a,
                x => Vector.GreaterThan<float>(v, x),
                x => b > x ? 1.0f : 0.0f);
        }
        public static Tensor LtEq_(Tensor a, Tensor b)
        {
            return ApplyBroadcastedOperation(a, b,
                (v1, v2) => Vector.LessThanOrEqual<float>(v1, v2),
                (f1, f2) => f1 <= f2 ? 1.0f : 0.0f);
        }
        public static Tensor LtEq(Tensor a, Tensor b)
        {
            return ApplyBroadcastOperation(a, b, LtEq_);
        }
        public static Tensor LtEq(Tensor a, float b)
        {
            var v = new Vector<float>(b);
            return ApplyMap(a,
                x => Vector.LessThanOrEqual<float>(x, v),
                x => x <= b ? 1.0f : 0.0f);
        }
        public static Tensor LtEq(float b, Tensor a)
        {
            var v = new Vector<float>(b);
            return ApplyMap(a,
                x => Vector.LessThanOrEqual<float>(v, x),
                x => b <= x ? 1.0f : 0.0f);
        }
        public static Tensor Lt_(Tensor a, Tensor b)
        {
            return ApplyBroadcastedOperation(a, b,
                (v1, v2) => Vector.LessThan<float>(v1, v2),
                (f1, f2) => f1 <= f2 ? 1.0f : 0.0f);
        }
        public static Tensor Lt(Tensor a, Tensor b)
        {
            return ApplyBroadcastOperation(a, b, Lt_);
        }
        public static Tensor Lt(Tensor a, float b)
        {
            var v = new Vector<float>(b);
            return ApplyMap(a,
                x => Vector.LessThan<float>(x, v),
                x => x < b ? 1.0f : 0.0f);
        }
        public static Tensor Lt(float b, Tensor a)
        {
            var v = new Vector<float>(b);
            return ApplyMap(a,
                x => Vector.LessThan<float>(v, x),
                x => b < x ? 1.0f : 0.0f);
        }
        public static Tensor Eq_(Tensor a, Tensor b)
        {
            return ApplyBroadcastedOperation(a, b,
                (v1, v2) => Vector.Equals<float>(v1, v2),
                (f1, f2) => f1 == f2 ? 1.0f : 0.0f);
        }
        public static Tensor Eq(Tensor a, Tensor b)
        {
            return ApplyBroadcastOperation(a, b, Eq_);
        }
        public static Tensor Eq(Tensor a, float b)
        {
            var v = new Vector<float>(b);
            return ApplyMap(a,
                x => Vector.Equals<float>(x, v),
                x => x == b ? 1.0f : 0.0f);
        }
        public static Tensor Eq(float a, Tensor b) => Eq(b, a);
        public static Tensor NEq_(Tensor a, Tensor b)
        {
            return ApplyBroadcastedOperation(a, b,
                (v1, v2) => Vector<float>.One - Vector.Equals<float>(v1, v2),
                (f1, f2) => f1 != f2 ? 1.0f : 0.0f);
        }
        public static Tensor NEq(Tensor a, Tensor b)
        {
            return ApplyBroadcastOperation(a, b, NEq_);
        }
        public static Tensor NEq(Tensor a, float b)
        {
            var v = new Vector<float>(b);
            return ApplyMap(a,
                x => Vector<float>.One - Vector.Equals<float>(x, v),
                x => x != b ? 1.0f : 0.0f);
        }
        public static Tensor operator >=(Tensor a, Tensor b) => GtEq(a, b);
        public static Tensor operator >=(Tensor a, float b) => GtEq(a, b);
        public static Tensor operator >=(float a, Tensor b) => GtEq(a, b);
        public static Tensor operator >(Tensor a, Tensor b) => Gt(a, b);
        public static Tensor operator >(Tensor a, float b) => Gt(a, b);
        public static Tensor operator >(float a, Tensor b) => Gt(a, b);
        public static Tensor operator <=(Tensor a, Tensor b) => LtEq(a, b);
        public static Tensor operator <=(Tensor a, float b) => LtEq(a, b);
        public static Tensor operator <=(float a, Tensor b) => LtEq(a, b);
        public static Tensor operator <(Tensor a, Tensor b) => Lt(a, b);
        public static Tensor operator <(Tensor a, float b) => Lt(a, b);
        public static Tensor operator <(float a, Tensor b) => Lt(a, b);
        public static Tensor operator ==(Tensor a, Tensor b) => Eq(a, b);
        public static Tensor operator ==(Tensor a, float b) => Eq(a, b);
        public static Tensor operator ==(float a, Tensor b) => Eq(a, b);
        public static Tensor operator !=(Tensor a, Tensor b) => NEq(a, b);
        public static Tensor operator !=(Tensor a, float b) => NEq(a, b);
        public static Tensor operator !=(float a, Tensor b) => NEq(a, b);
        public static Tensor ReLU(Tensor a)
        {
            return ApplyMap(a,
                x => Vector.Max(x, Vector<float>.Zero),
                x => MathF.Max(x, 0.0f));
        }
        public static Tensor LeakyReLU(Tensor a, float negativeSlope)
        {
            var v = new Vector<float>(negativeSlope);
            return ApplyMap(a,
                x => Vector.ConditionalSelect(Vector.LessThan(x, Vector<float>.Zero), x * v, x),
                x => x < 0.0f ? x * negativeSlope : x);
        }
    }
}