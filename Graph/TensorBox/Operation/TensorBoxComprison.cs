namespace SimpleNN.Graph
{
    using SimpleNN.Graph.Functions;
    public partial class TensorBox
    {
        public static TensorBox Maximum(TensorBox a, TensorBox b)
        {
            return Apply(a, b, MaximumFunction.Forward);
        }
        public TensorBox Maximum(float value)
        {
            return new(MaximumScalerFunction.Forward(new() { Scalar = value }, _ctx));
        }
        public static TensorBox Minimum(TensorBox a, TensorBox b)
        {
            return Apply(a, b, MinimumFunction.Forward);
        }
        public TensorBox Minimum(float value)
        {
            return new(MinimumScalerFunction.Forward(new() { Scalar = value }, _ctx));
        }
        public static TensorBox operator <(TensorBox a, TensorBox b) => new(a.ctxTensor < b.ctxTensor, false);
        public static TensorBox operator <(TensorBox a, float b) => new(a.ctxTensor < b, false);
        public static TensorBox operator <(float a, TensorBox b) => new(a < b.ctxTensor, false);
        public static TensorBox operator >(TensorBox a, TensorBox b) => new(a.ctxTensor > b.ctxTensor, false);
        public static TensorBox operator >(TensorBox a, float b) => new(a.ctxTensor > b, false);
        public static TensorBox operator >(float a, TensorBox b) => new(a > b.ctxTensor, false);
        public static TensorBox operator <=(TensorBox a, TensorBox b) => new(a.ctxTensor <= b.ctxTensor, false);
        public static TensorBox operator <=(TensorBox a, float b) => new(a.ctxTensor <= b, false);
        public static TensorBox operator <=(float a, TensorBox b) => new(a < b.ctxTensor, false);
        public static TensorBox operator >=(TensorBox a, TensorBox b) => new(a.ctxTensor >= b.ctxTensor, false);
        public static TensorBox operator >=(TensorBox a, float b) => new(a.ctxTensor >= b, false);
        public static TensorBox operator >=(float a, TensorBox b) => new(a >= b.ctxTensor, false);
        public static TensorBox operator ==(TensorBox a, TensorBox b) => new(a.ctxTensor == b.ctxTensor, false);
        public static TensorBox operator ==(TensorBox a, float b) => new(a.ctxTensor == b, false);
        public static TensorBox operator ==(float a, TensorBox b) => new(a == b.ctxTensor, false);
        public static TensorBox operator !=(TensorBox a, TensorBox b) => new(a.ctxTensor != b.ctxTensor, false);
        public static TensorBox operator !=(TensorBox a, float b) => new(a.ctxTensor != b, false);
        public static TensorBox operator !=(float a, TensorBox b) => new(a != b.ctxTensor, false);
        public static bool operator ==(TensorBox a, float? b){
            if (b.HasValue)
            {
                return a.ctxTensor.IsScaler && (b == a.ctxTensor.Item());
            }
            return a is null;
        }
        public static bool operator ==(float? a, TensorBox b) => b == a;
        public static bool operator !=(TensorBox a, float? b) => !(a==b);
        public static bool operator !=(float? a, TensorBox b) => !(b==a);
        public TensorBox ReLU()
        {
            return new(ReLUFunction.Forward(_ctx));
        }
        public TensorBox LeakyReLU(float negativeSlope = 0.01f)
        {
            return new(LeakyReLUFunction.Forward(new(){NegativeSlope=negativeSlope},_ctx));
        }
    }
}