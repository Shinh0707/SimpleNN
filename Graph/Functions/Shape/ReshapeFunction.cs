namespace SimpleNN.Graph.Functions
{
    using UnityEngine;
    using SimpleNN.Tensor;

    /// <summary>
    /// テンソルの形状を変更する関数.
    /// </summary>
    public class ReshapeFunction : KwargsFunction<ReshapeFunction, ReshapeFunction.Kwargs>
    {
        public class Kwargs
        {
            public int[] newSize;
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            //Debug.Log($"(Reshape) Forward : {Util.StringExt.ArrayString("TargetSize",kwargs.newSize)}, Input={Util.StringExt.ArrayString("InputSize",ctx.GetInput(0).Size)}");
            return Tensor.Reshape(ctx.GetInput(0), kwargs.newSize);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            // 逆伝播では、勾配を入力テンソルの形状に戻す
            var inputSize = ctx.GetInput(0).Size;
            //Debug.Log($"(Reshape) Backward : {Util.StringExt.ArrayString("TargetSize",kwargs.newSize)}, ForwardInput={Util.StringExt.ArrayString("InputSize",inputSize)}, GradInput={grad}");
            return new[] { Tensor.Reshape(grad, inputSize) };
        }
    }
}