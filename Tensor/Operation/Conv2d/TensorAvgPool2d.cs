namespace SimpleNN.Tensor
{
    public partial class Tensor
    {
        // AveragePool 用のデリゲート
        private static (float val, int ptr) AvgPoolInit()
        {
            // (Sum, Count)
            // Count はここで管理するので、ptr の初期値は 0
            return (0.0f, 0);
        }

        private static (float val, int ptr) AvgPoolAccumulate(
            float inData, int inPtr, (float val, int ptr) currentState)
        {
            // currentState = (currentSum, currentCount)
            float newSum = currentState.val + inData;
            int newCount = currentState.ptr + 1;

            // (newSum, newCount)
            return (newSum, newCount);
        }

        private static (float val, int ptr) AvgPoolFinalize(
            (float val, int ptr) finalState, int windowElementCount)
        {
            float avg = 0.0f;
            int count = finalState.ptr; // (TotalCount)

            if (count > 0)
            {
                avg = finalState.val / count; // (TotalSum / TotalCount)
            }
            return (avg, count);
        }


        // --- public AveragePool2d (Overloads) ---
        public static (Tensor, int[]) AveragePool2d(
            Tensor input,
            int kernelSize,
            int stride = -1,
            int padding = 0)
        {
            return Pool2d(
                input,
                AvgPoolInit,
                AvgPoolAccumulate,
                AvgPoolFinalize,
                kernelSize, stride, padding
            );
        }

        public static (Tensor, int[]) AveragePool2d(
            Tensor input,
            int kernelH, int kernelW,
            int strideH, int strideW,
            int padH, int padW)
        {
            return Pool2d(
                input,
                AvgPoolInit,
                AvgPoolAccumulate,
                AvgPoolFinalize,
                kernelH, kernelW,
                strideH, strideW,
                padH, padW
            );
        }
    }
}