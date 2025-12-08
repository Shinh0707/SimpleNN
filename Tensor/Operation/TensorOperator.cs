using System;
using System.Numerics;
using SimpleNN.Util;
using UnityEngine;

namespace SimpleNN.Tensor
{
    public partial class Tensor
    {
        public delegate Tensor ApplyBroadcastOperationFunction(Tensor a, Tensor b);
        private static Tensor ApplyBroadcastOperation(Tensor a, Tensor b, ApplyBroadcastOperationFunction func)
        {
            (var x, var y) = BroadcastBoth(a, b);
            return func(x, y);
        }
        private static Tensor ApplyBroadcastedOperation(Tensor a, Tensor b, Operation.FVBinaryOperation vectorOperation, Operation.FBinaryOperation operation)
        {
            if (IsSameSizeAndStrides(a,b))
            {
                var newData = ArrayExt.ApplyBinaryOperation(a._data, b._data, vectorOperation, operation);
                return new(newData, a.Size);
            }
            return ApplyStridedOperation(a, b, operation);
        }
        private static Tensor ApplyStridedOperation(Tensor a, Tensor b, Operation.FBinaryOperation op)
        {
            var totalSize = a.TotalSize;
            var outData = new float[totalSize];
            var outSize = (int[])a.Size.Clone();
            int ndim = a.NDim;
            int[] counters = new int[ndim];
            var aStrides = a.Strides;
            var bStrides = b.Strides;
            var aData = a._data;
            var bData = b._data;
            for (uint i = 0; i < totalSize; i++)
            {
                int aIndex = 0;
                int bIndex = 0;
                for (int d = 0; d < ndim; d++)
                {
                    aIndex += counters[d] * aStrides[d];
                    bIndex += counters[d] * bStrides[d];
                }
                outData[i] = op(aData[aIndex], bData[bIndex]);
                for (int d = ndim - 1; d >= 0; d--)
                {
                    counters[d]++;
                    if (counters[d] < outSize[d])
                    {
                        break;
                    }
                    counters[d] = 0;
                }
            }
            return new Tensor(outData, outSize);
        }
        private static Tensor ApplyMap(Tensor a,Operation.FUnaryOperation op)
        {
            return new Tensor(ArrayExt.ApplyUnaryOperation(a._data, op), (int[])a.Size.Clone());
        }
        private static Tensor ApplyMap(Tensor a, Operation.FVUnaryOperation vop, Operation.FUnaryOperation op)
        {
            return new Tensor(ArrayExt.ApplyUnaryOperation(a._data, vop, op), (int[])a.Size.Clone());
        }
        public static Tensor Add_(Tensor a, Tensor b)
        {
            return ApplyBroadcastedOperation(a, b, (x, y) => x + y, (x, y) => x + y);
        }
        public static Tensor operator +(Tensor a, Tensor b)
        {
            return ApplyBroadcastOperation(a, b, Add_);
        }
        public static Tensor Add_(Tensor a, float value)
        {
            Vector<float> v = new(value);
            return ApplyMap(a, x => x + v, x => x + value);
        }
        public static Tensor Add_(float value, Tensor a)
        {
            return Add_(a, value);
        }
        public static Tensor operator +(Tensor a, float value) => Add_(a, value);
        public static Tensor operator +(float value, Tensor a) => Add_(a, value);
        public static Tensor Mul_(Tensor a, Tensor b)
        {
            return ApplyBroadcastedOperation(a, b, (x, y) => x * y, (x, y) => x * y);
        }
        public static Tensor operator *(Tensor a, Tensor b)
        {
            return ApplyBroadcastOperation(a, b, Mul_);
        }
        public static Tensor Mul_(Tensor a, float value)
        {
            Vector<float> v = new(value);
            return ApplyMap(a, x => x * v, x => x * value);
        }
        public static Tensor Mul_(float value, Tensor a)
        {
            return Mul_(a, value);
        }
        public static Tensor operator *(Tensor a, float value) => Mul_(a, value);
        public static Tensor operator *(float value, Tensor a) => Mul_(a, value);
        public static Tensor Div_(Tensor a, Tensor b)
        {
            return ApplyBroadcastedOperation(a, b, (x, y) => x / y, (x, y) => x / y);
        }
        public static Tensor operator /(Tensor a, Tensor b)
        {
            return ApplyBroadcastOperation(a, b, Div_);
        }
        public static Tensor Div_(Tensor a, float value)
        {
            Vector<float> v = new(value);
            return ApplyMap(a, x => x / v, x => x / value);
        }
        public static Tensor Div_(float value, Tensor a)
        {
            Vector<float> v = new(value);
            return ApplyMap(a, x => v / x, x => value / x);
        }
        public static Tensor operator /(Tensor a, float value) => Div_(a, value);
        public static Tensor operator /(float value, Tensor a) => Div_(value, a);
        public static Tensor Sub_(Tensor a, Tensor b)
        {
            return ApplyBroadcastedOperation(a, b, (x, y) => x - y, (x, y) => x - y);
        }
        public static Tensor operator -(Tensor a, Tensor b)
        {
            return ApplyBroadcastOperation(a, b, Sub_);
        }
        public static Tensor Sub_(Tensor a, float value)
        {
            Vector<float> v = new(value);
            return ApplyMap(a, x => x - v, x => x - value);
        }
        public static Tensor Sub_(float value, Tensor a)
        {
            Vector<float> v = new(value);
            return ApplyMap(a, x => v - x, x => value - x);
        }
        public static Tensor operator -(Tensor a, float value) => Sub_(a, value);
        public static Tensor operator -(float value, Tensor a) => Sub_(value, a);
        public static Tensor Pow_(Tensor a, Tensor b)
        {
            return ApplyStridedOperation(a, b, (x, y) => MathF.Pow(x,y));
        }
        public static Tensor Pow(Tensor a, Tensor b)
        {
            return ApplyBroadcastOperation(a, b, Pow_);
        }
        private static Tensor Pow(Tensor a, float power)
        {
            if (3.0f <= power)
            {
                if (3.0f == power)
                {
                    return Cube(a);
                }
                float p = power / 3.0f;
                if (p == (int)p)
                {
                    return Cube(Pow(a, p));
                }
            }
            if (2.0f <= power)
            {
                if (2.0f == power)
                {
                    return Square(a);
                }
                return Square(Pow(a, power * 0.5f));
            }
            if (1.0f <= power)
            {
                if (1.0f == power)
                {
                    return a;
                }
                return a * Pow(a, power - 1.0f);
            }
            if (0.5f <= power)
            {
                if (0.5f == power)
                {
                    return Sqrt(a);
                }
                float p = power - 0.5f;
                return Sqrt(a) * ApplyMap(a, x => MathF.Pow(x, p));
            }
            if (0.0f <= power)
            {
                if (0.0f == power)
                {
                    return OnesLike(a);
                }
                return ApplyMap(a, x => MathF.Pow(x, power));
            }
            return Reciprocal(Pow(a, -power));
        }
        public static Tensor Pow(float _base, Tensor a)
        {
            return ApplyMap(a, x => MathF.Pow(_base, x));
        }
        public static Tensor Neg(Tensor a)
        {
            return ApplyMap(a, x => -x, x => -x);
        }
        public static Tensor operator -(Tensor a) => Neg(a);
        public static Tensor Sqrt(Tensor a)
        {
            return ApplyMap(a, Vector.SquareRoot, MathF.Sqrt);
        }
        public static Tensor Square(Tensor a)
        {
            return ApplyMap(a, x => x * x, x => x * x);
        }
        public static Tensor Cube(Tensor a)
        {
            return ApplyMap(a, x => x * x * x, x => x * x * x);
        }
        public static Tensor Abs(Tensor a)
        {
            return ApplyMap(a, Vector.Abs, MathF.Abs);
        }
        public static Tensor Reciprocal(Tensor a)
        {
            return Div_(1.0f, a);
        }
        public static Tensor Sign(Tensor a)
        {
            return ApplyMap(a,
                VectorExt.Sign,
                x => MathF.Sign(x));
        }
        public static Tensor Shrink(Tensor a, float threshold)
        {
            var v0 = Vector<float>.Zero;
            var v1 = Vector<float>.One;
            var vt = new Vector<float>(threshold);
            return ApplyMap(a,
                x => {
                    var absX = Vector.Abs(x);
                    var lessThan = Vector.LessThan(absX, vt);
                    
                    var sign = VectorExt.Sign(x);
                    var shrunkValue = vt * sign;
                    
                    return Vector.ConditionalSelect(lessThan, shrunkValue, x);
                },
                x => {
                    return MathF.Abs(x) < threshold ? (threshold * (x == 0f ? 1f : MathF.Sign(x))) : x;
                });
        }
        public static Tensor Exp(Tensor a)
        {
            return ApplyMap(a, MathF.Exp);
        }
        public static Tensor MExp(Tensor a)
        {
            return ApplyMap(a, (x) => MathF.Exp(-x));
        }
        public static Tensor Tanh(Tensor a)
        {
            return ApplyMap(a, MathF.Tanh);
        }
        public static Tensor Sech(Tensor a)
        {
            return ApplyMap(a, x => 1.0f / MathF.Cosh(x));
        }
        public static Tensor Log(Tensor a)
        {
            return ApplyMap(a, MathF.Log);
        }
        public static Tensor Softplus(Tensor a)
        {
            return ApplyMap(a, x => MathF.Log(1.0f + MathF.Exp(x)));
        }
        public static Tensor Sigmoid(Tensor a)
        {
            return ApplyMap(a, x => 1.0f / (1.0f + MathF.Exp(-x)));
        }
        public static Tensor LGamma(Tensor a)
        {
            return ApplyMap(a, Operation.LGamma);
        }
        public static Tensor Digamma(Tensor a)
        {
            return ApplyMap(a, Operation.Digamma);
        }
        public static Tensor Trigamma(Tensor a)
        {
            return ApplyMap(a, Operation.Trigamma);
        }
    }
}