using SimpleNN.Graph;

namespace SimpleNN.Module
{
    public class Sequential : SingleInSingleOutModule
    {
        public Sequential(params SingleInSingleOutModule[] modules)
        {
            int c = modules.Length;
            for (int i = 0; i < c; i++)
            {
                AddModule(modules[i]);
            }
        }
        public override TensorBox Forward(TensorBox inputTensor)
        {
            var t = inputTensor;
            int c = _children.Count;
            for (int i = 0; i < c; i++)
            {
                var child = _children.Dequeue() as SingleInSingleOutModule;
                t = child.Forward(t);
                _children.Enqueue(child);
            }
            return t;
        }
    }
}