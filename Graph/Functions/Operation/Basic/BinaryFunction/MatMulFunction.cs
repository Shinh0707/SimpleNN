namespace SimpleNN.Graph.Functions
{
    using UnityEngine;
    using SimpleNN.Tensor;

    public class MatMulFunction : SingleFunction<MatMulFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.MatMul(ctx.GetInput(0), ctx.GetInput(1));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            Tensor input = ctx.GetInput(0);  // A
            Tensor weight = ctx.GetInput(1); // B

            // Forward: Y = A @ B
            // Backward:
            // dL/dA = dL/dY @ B^T
            // dL/dB = A^T @ dL/dY

            // B^T (最後の2次元を転置)
            Tensor weightT = Tensor.Transpose(weight, -1, -2);
            //Debug.Log($"Input A (grad): {grad}, Input B (weight transposed): {weightT} / (weight: {weight}) ");
            Tensor gradInput = Tensor.MatMul(grad, weightT);

            // A^T
            Tensor inputT = Tensor.Transpose(input, -1, -2);
            //Debug.Log($"Input A (input transposed): {inputT} / (input: {input}), Input B (grad): {grad}");
            Tensor gradWeight = Tensor.MatMul(inputT, grad);

            return new[] { gradInput, gradWeight };
        }
    }
}