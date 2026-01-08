namespace SimpleNN.Tensor
{
    public partial class Tensor
    {
        // --- public AveragePool1d (Overloads) ---
        public static (Tensor, int[]) AveragePool1d(
            Tensor input,
            int kernelSize,
            int stride = -1,
            int padding = 0)
        {
            return Pool1d(
                input,
                AvgPoolInit,
                AvgPoolAccumulate,
                AvgPoolFinalize,
                kernelSize, stride, padding
            );
        }

        public static (Tensor, int[]) AveragePool1d(
            Tensor input,
            int kernelSize,
            int stride,
            int padLeft,
            int padRight)
        {
            return Pool1d(
                input,
                AvgPoolInit,
                AvgPoolAccumulate,
                AvgPoolFinalize,
                kernelSize,
                stride,
                padLeft,
                padRight
            );
        }
    }
}
