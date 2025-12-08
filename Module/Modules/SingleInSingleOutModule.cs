namespace SimpleNN.Module
{
    using SimpleNN.Graph;
    public abstract class SingleInSingleOutModule : Module
    {
        public abstract TensorBox Forward(TensorBox inputTensor);
    }
}