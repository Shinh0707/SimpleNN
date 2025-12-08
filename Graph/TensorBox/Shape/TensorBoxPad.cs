namespace SimpleNN.Graph
{
    using SimpleNN.Graph.Functions;
    using SimpleNN.Tensor;

    public partial class TensorBox
    {
        public TensorBox Pad((int left, int right)[] paddingSizes, PaddingMode mode = PaddingMode.CONSTANT, float value = 0.0f)
        {
            Context resultCtx;
            switch (mode)
            {
                case PaddingMode.CONSTANT:
                    resultCtx = PadConstantFunction.Forward(new PadConstantFunction.Kwargs { paddingSizes = paddingSizes, value = value }, _ctx);
                    break;
                case PaddingMode.REFLECT:
                    resultCtx = PadReflectFunction.Forward(new PadReflectFunction.Kwargs { paddingSizes = paddingSizes }, _ctx);
                    break;
                case PaddingMode.REPLICATE:
                    resultCtx = PadReplicateFunction.Forward(new PadReplicateFunction.Kwargs { paddingSizes = paddingSizes }, _ctx);
                    break;
                case PaddingMode.CIRCULAR:
                    resultCtx = PadCircularFunction.Forward(new PadCircularFunction.Kwargs { paddingSizes = paddingSizes }, _ctx);
                    break;
                default:
                    throw new System.ArgumentException($"{mode} is not supported");
            }
            return new TensorBox(resultCtx);
        }
    }
}
