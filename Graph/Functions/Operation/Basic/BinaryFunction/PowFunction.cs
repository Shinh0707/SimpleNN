namespace SimpleNN.Graph.Functions
{
    using System;
    using SimpleNN.Tensor;

    public class PowFunction : SingleFunction<PowFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Pow_(ctx.GetInput(0), ctx.GetInput(1));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var cache = ctx.Tensor;
            var x = ctx.GetInput(0);
            var y = ctx.GetInput(1);
            return new[] { grad * y * Tensor.Pow_(x, y - 1), grad * cache * Tensor.Log(x) };
        }
    }
    public class PowRSclFunction : KwargsFunction<PowRSclFunction, PowRSclFunction.Kwargs>
    {
        public class Kwargs
        {
            public float Power;
        }
        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            var x = ctx.GetInput(0);
            var xpow = Tensor.Pow(x, kwargs.Power-1.0f);
            ctx.RegisterTensors(xpow);
            return xpow * x;
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            var p = kwargs.Power;
            var xpow = ctx.GetRegisteredTensor(0);
            var derivative = Tensor.Mul_(p, xpow);
            return new[] { Tensor.Mul_(grad, derivative) };
        }
    }
    public class PowLSclFunction : KwargsFunction<PowLSclFunction, PowLSclFunction.Kwargs>
    {
        public class Kwargs
        {
            public float Base;
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.Pow(kwargs.Base, ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            var cache = ctx.Tensor;
            return new[] { grad * cache * MathF.Log(kwargs.Base) };
        }
    }
    public class SqrtFunction : SingleFunction<SqrtFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Sqrt(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var derivative = Tensor.Div_(0.5f, ctx.Tensor);
            return new[] { Tensor.Mul_(grad, derivative) };
        }
    }
    public class SquareFunction : SingleFunction<SquareFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Square(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {   
            var x = ctx.GetInput(0);
            var derivative = Tensor.Mul_(2.0f, x);
            return new[] { Tensor.Mul_(grad, derivative) };
        }
    }
    public class CubeFunction : SingleFunction<CubeFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            var x = ctx.GetInput(0);
            var xsq = Tensor.Square(x);
            ctx.RegisterTensors(xsq);
            return xsq * x;
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {   
            var xsq = ctx.GetRegisteredTensor(0);
            var derivative = Tensor.Mul_(3.0f, xsq);
            return new[] { Tensor.Mul_(grad, derivative) };
        }
    }
}