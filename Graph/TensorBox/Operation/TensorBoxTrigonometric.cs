namespace SimpleNN.Graph
{
    using SimpleNN.Graph.Functions;
    public partial class TensorBox
    {
        public TensorBox Sin()
        {
            return new(SinFunction.Forward(_ctx));
        }
        public TensorBox Cos()
        {
            return new(CosFunction.Forward(_ctx));
        }
        public TensorBox Tan()
        {
            return new(TanFunction.Forward(_ctx));
        }
    }
}