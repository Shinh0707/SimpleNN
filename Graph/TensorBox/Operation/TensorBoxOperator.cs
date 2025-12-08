namespace SimpleNN.Graph
{
    using SimpleNN.Graph.Functions;
    using SimpleNN.Tensor;
    public partial class TensorBox
    {
        private delegate Context ApplyOperator(Context[] inputContexts);
        private static TensorBox Apply(TensorBox a, TensorBox b, ApplyOperator method)
        {
            (var x, var y) = BroadcastBothCtx(a, b);
            return new(method(new[] { x, y }));
        }
        private static TensorBox Apply(TensorBox a, TensorBox b, TensorBox c, ApplyOperator method)
        {
            (var x, var y) = BroadcastBothCtx(a, b);
            return new(method(new[] { x, y, c.BroadcastCtx(x)}));
        }
        public static TensorBox operator +(TensorBox a, TensorBox b)
        {
            return Apply(a, b, AddFunction.Forward);
        }
        public static TensorBox operator +(TensorBox a, float value)
        {
            return new(AddSclFunction.Forward(new() { Value = value }, a._ctx));
        }
        public static TensorBox operator +(float value, TensorBox a)
        {
            return a + value;
        }
        public static TensorBox operator *(TensorBox a, TensorBox b)
        {
            return Apply(a, b, MulFunction.Forward);
        }
        public static TensorBox operator *(TensorBox a, float value)
        {
            return new(MulSclFunction.Forward(new() { Value = value }, a._ctx));
        }
        public static TensorBox operator *(float value, TensorBox a)
        {
            return a * value;
        }
        public static TensorBox operator /(TensorBox a, TensorBox b)
        {
            return Apply(a, b, DivFunction.Forward);
        }
        public static TensorBox operator /(TensorBox a, float value)
        {
            return new(DivRSclFunction.Forward(new() { Value = value }, a._ctx));
        }
        public static TensorBox operator /(float value, TensorBox a)
        {
             return new(DivLSclFunction.Forward(new() { Value = value }, a._ctx));
        }
        public static TensorBox operator -(TensorBox a, TensorBox b)
        {
            return Apply(a, b, SubFunction.Forward);
        }
        public static TensorBox operator -(TensorBox a, float value)
        {
            return new(SubRSclFunction.Forward(new() { Value = value }, a._ctx));
        }
        public static TensorBox operator -(float value, TensorBox a)
        {
            return new(SubLSclFunction.Forward(new() { Value = value }, a._ctx));
        }
        public static TensorBox operator -(TensorBox a)
        {
            return new(NegFunction.Forward(a._ctx));
        }
        public static TensorBox MatMul(TensorBox a, TensorBox b)
        {
            return new(MatMulFunction.Forward(new[] { a._ctx, b._ctx }));
        }
        public static TensorBox Pow(TensorBox a, TensorBox b)
        {
            return Apply(a, b, PowFunction.Forward);
        }
        public TensorBox Pow(TensorBox b) => Pow(this, b);
        public static TensorBox Pow(TensorBox a, float power)
        {
            return new(PowRSclFunction.Forward(new() { Power = power }, a._ctx));
        }
        public TensorBox Pow(float power) => Pow(this, power);
        public static TensorBox Pow(float baseValue, TensorBox a)
        {
            return new(PowLSclFunction.Forward(new() { Base = baseValue }, a._ctx));
        }
        public TensorBox Sqrt()
        {
            return new(SqrtFunction.Forward(_ctx));
        }
        public TensorBox Square()
        {
            return new(SquareFunction.Forward(_ctx));
        }
        public TensorBox Cube()
        {
            return new(CubeFunction.Forward(_ctx));
        }
        public TensorBox Sign()
        {
            return new(Tensor.Sign(_ctx.Tensor), false);
        }
        public TensorBox Abs()
        {
            return new(AbsFunction.Forward(_ctx));
        }
        public TensorBox Exp()
        {
            return new(ExpFunction.Forward(_ctx));
        }
        public TensorBox MExp()
        {
            return new(MExpFunction.Forward(_ctx));
        }
        public TensorBox Tanh()
        {
            return new(TanhFunction.Forward(_ctx));
        }
        public TensorBox Sech()
        {
            return new(SechFunction.Forward(_ctx));
        }
        public TensorBox Log()
        {
            return new(LogFunction.Forward(_ctx));
        }
        public TensorBox Softplus()
        {
            return new(SoftplusFunction.Forward(_ctx));
        }
        public TensorBox Sigmoid()
        {
            return new(SigmoidFunction.Forward(_ctx));
        }
    }
}