namespace SimpleNN.Tensor
{
    public partial class Tensor
    {
        // --- public AveragePool3d (Overloads) ---
        public static (Tensor, int[]) AveragePool3d(
            Tensor input,
            int kernelSize,
            int stride = -1,
            int padding = 0)
        {
            return Pool3d(
                input,
                AvgPoolInit,
                AvgPoolAccumulate,
                AvgPoolFinalize,
                kernelSize, stride, padding
            );
        }

        public static (Tensor, int[]) AveragePool3d(
            Tensor input,
            int kernelD, int kernelH, int kernelW,
            int strideD, int strideH, int strideW,
            int padD, int padH, int padW)
        {
            return Pool3d(
                input,
                AvgPoolInit,
                AvgPoolAccumulate,
                AvgPoolFinalize,
                kernelD, kernelH, kernelW,
                strideD, strideH, strideW,
                padD, padH, padW
            );
        }
    }
}
