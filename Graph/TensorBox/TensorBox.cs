namespace SimpleNN.Graph
{
    using SimpleNN.Graph.Functions;
    using SimpleNN.Tensor;
    using Unity.VisualScripting;

    public partial class TensorBox
    {
        public Tensor GetTensor() => ctxTensor;
        public Tensor GetGrad() => _ctx.Grad;
        public bool RequireGrad
        {
            get
            {
                return _ctx.RequireGrad;
            }
            set
            {
                _ctx.RequireGrad = value;
            }
        }
        public void ZeroGrad() => _ctx.ZeroGrad();
        public void StepGrad(Context.StepFunction stepFunc) => _ctx.Step(stepFunc);
        private readonly Context _ctx;
        private Tensor ctxTensor => _ctx.Tensor;
        public int[] Size => ctxTensor.Size;
        public int NDim => ctxTensor.NDim;
        public TensorBox(Tensor tensor, bool requireGrad = true)
        {
            _ctx = new(tensor, requireGrad);
        }
        public TensorBox(Context ctx)
        {
            _ctx = ctx;
        }
        public void Backward()
        {
            _ctx.Backward();
        }
        public TensorBox Detach()
        {
            return new TensorBox(Tensor.Clone(ctxTensor), false);
        }
        public static implicit operator TensorBox(Tensor tensor)
        {
            return new(tensor, false);
        }
        public override string ToString()
        {
            return _ctx.ToString();
        }
    }
}