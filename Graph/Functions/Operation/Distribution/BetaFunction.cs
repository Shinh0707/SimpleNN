namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    public class BetaLogprobFunction : SingleFunction<BetaLogprobFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            var a = ctx.GetInput(0);
            var b = ctx.GetInput(1);
            var x = ctx.GetInput(2);
            var apb = Tensor.Add_(a, b);
            var gamapb = Tensor.LGamma(apb);
            var gama = Tensor.LGamma(a);
            var gamb = Tensor.LGamma(b);
            var logx = Tensor.Log(x);
            var onemx = 1.0f - x;
            var log1mx = Tensor.Log(onemx);
            ctx.RegisterTensors(apb, logx, onemx, log1mx);
            return Tensor.Add_(Tensor.Sub_(Tensor.Sub_(gamapb,gama),gamb),Tensor.Add_((a - 1.0f) * logx,(b - 1.0f) * log1mx));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var a = ctx.GetInput(0);
            var b = ctx.GetInput(1);
            var x = ctx.GetInput(2);
            var apb = ctx.GetRegisteredTensor(0);
            var digapb = Tensor.Digamma(apb);
            var diga = Tensor.Digamma(a);
            var digb = Tensor.Digamma(b);
            var logx = ctx.GetRegisteredTensor(1);
            var onemx = ctx.GetRegisteredTensor(2);
            var log1mx = ctx.GetRegisteredTensor(3);
            return new[]
            {
                grad * (digapb-diga+logx),
                grad * (digapb-digb+log1mx),
                grad * ((a-1.0f)/x-(b-1.0f)/onemx)
            };
        }
    }
    public class LBetaFunction : SingleFunction<LBetaFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            var a = ctx.GetInput(0);
            var b = ctx.GetInput(1);
            var apb = Tensor.Add_(a, b);
            var gamapb = Tensor.LGamma(apb);
            var gama = Tensor.LGamma(a);
            var gamb = Tensor.LGamma(b);
            ctx.RegisterTensors(apb);
            return Tensor.Sub_(Tensor.Sub_(gamapb,gama),gamb);
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var a = ctx.GetInput(0);
            var b = ctx.GetInput(1);
            var apb = ctx.GetRegisteredTensor(0);
            var digapb = Tensor.Digamma(apb);
            var diga = Tensor.Digamma(a);
            var digb = Tensor.Digamma(b);
            return new[]
            {
                grad * (digapb - diga),
                grad * (digapb - digb),
            };
        }
    }
    public class BetaExpFunction : SingleFunction<BetaExpFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            var a = ctx.GetInput(0);
            var b = ctx.GetInput(1);
            var apb = Tensor.Add_(a, b);
            ctx.RegisterTensors(apb);
            return Tensor.Div_(a, apb);
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var a = ctx.GetInput(0);
            var b = ctx.GetInput(1);
            var apb = ctx.GetRegisteredTensor(0);
            var sqapb = Tensor.Square(apb);
            return new[]
            {
                grad * b / sqapb,
                -(grad * a / sqapb),
            };
        }
    }
    public class BetaModeFunction : SingleFunction<BetaModeFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            var a = ctx.GetInput(0);
            var b = ctx.GetInput(1);
            var apbm2 = Tensor.Add_(a, b) - 2.0f;
            var am1 = a - 1.0f;
            ctx.RegisterTensors(apbm2, -am1);
            return Tensor.Div_(am1, apbm2);
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var a = ctx.GetInput(0);
            var b = ctx.GetInput(1);
            var apbm2 = ctx.GetRegisteredTensor(0);
            var onema = ctx.GetRegisteredTensor(1);
            var sqapbm2 = Tensor.Square(apbm2);
            return new[]
            {
                grad * (b-1.0f) / sqapbm2,
                -(grad * onema / sqapbm2),
            };
        }
    }
    public class BetaEntropyFunction : SingleFunction<BetaEntropyFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            var a = ctx.GetInput(0);
            var b = ctx.GetInput(1);
            var apb = Tensor.Add_(a, b);
            var gamapb = Tensor.LGamma(apb);
            var gama = Tensor.LGamma(a);
            var gamb = Tensor.LGamma(b);
            var digapb = Tensor.Digamma(apb);
            var diga = Tensor.Digamma(a);
            var digb = Tensor.Digamma(b);
            ctx.RegisterTensors(apb);
            return Tensor.Sub_(
                Tensor.Sub_(Tensor.Sub_(gamapb, gama), gamb),
                Tensor.Add_(
                    Tensor.Add_(
                    Tensor.Mul_(a - 1.0f, diga),
                    Tensor.Mul_(b - 1.0f, digb)),
                Tensor.Mul_(apb - 2.0f, digapb))
            );
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var a = ctx.GetInput(0);
            var b = ctx.GetInput(1);
            var apb = ctx.GetRegisteredTensor(0);
            var trigapb = Tensor.Trigamma(apb);
            var triga = Tensor.Trigamma(a);
            var trigb = Tensor.Trigamma(b);
            var term0 = (apb - 2.0f) * trigapb;
            return new[]
            {
                grad * (term0 - (a-1.0f)*triga),
                grad * (term0 - (b-1.0f)*trigb),
            };
        }
    }
}