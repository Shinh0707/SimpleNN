using System.Collections.Generic;
namespace SimpleNN.Optimizer
{
    using SimpleNN.Graph;
    using SimpleNN.Tensor;
    public abstract class Optimizer
    {
        private TensorBox[] _parameters = new TensorBox[0];
        public int NumParameters => _parameters.Length; 
        public Optimizer(params List<TensorBox>[] parameters)
        {
            List<TensorBox> param = new();
            foreach (var ps in parameters)
            {
                param.AddRange(ps);
            }
            _parameters = param.ToArray();
        }
        public void ZeroGrad()
        {
            foreach (var p in _parameters)
            {
                p.ZeroGrad();
            }
        }
        protected TensorBox GetParameter(int id) => _parameters[id];
        public void Step()
        {
            PreStep();
            int numParameters = _parameters.Length;
            for(int id = 0; id < numParameters; id++)
            {
                GetParameter(id).StepGrad((tensor, grad) => StepGrad(id, tensor, grad));
            }
        }
        public virtual void PreStep(){}
        protected abstract Tensor StepGrad(int id, Tensor tensor, Tensor grad);
    }
}