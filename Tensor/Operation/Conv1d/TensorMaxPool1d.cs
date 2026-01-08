namespace SimpleNN.Tensor
{
    public partial class Tensor
    {
        // --- public MaxPool1d (Overloads) ---
        public static (Tensor, int[]) MaxPool1d(
            Tensor input,
            int kernelSize,
            int stride = -1,
            int padding = 0)
        {
            return Pool1d(
                input,
                MaxPoolInit,
                MaxPoolAccumulate,
                MaxPoolFinalize,
                kernelSize, stride, padding
            );
        }

        public static (Tensor, int[]) MaxPool1d(
            Tensor input,
            int kernelSize,
            int stride,
            int padLeft,
            int padRight)
        {
            return Pool1d(
                input,
                MaxPoolInit,
                MaxPoolAccumulate,
                MaxPoolFinalize,
                kernelSize,
                stride,
                padLeft,
                padRight
            );
        }
    }
}
