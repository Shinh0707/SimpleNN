namespace SimpleNN.Tensor
{
    public partial class Tensor
    {
        // MaxPool 用のデリゲート
        private static (float val, int ptr) MaxPoolInit()
        {
            // (MaxVal, MaxIndex)
            return (float.NegativeInfinity, -1);
        }

        private static (float val, int ptr) MaxPoolAccumulate(
            float inData, int inPtr, (float val, int ptr) currentState)
        {
            // (currentVal, currentPtr)
            if (inData > currentState.val)
            {
                // (newMaxVal, newMaxIndex)
                return (inData, inPtr);
            }
            return currentState;
        }

        private static (float val, int ptr) MaxPoolFinalize(
            (float val, int ptr) finalState, int windowElementCount)
        {
            // MaxPool は最終化処理なし. そのまま返す.
            return finalState;
        }

        // --- public MaxPool2d (Overloads) ---
        public static (Tensor, int[]) MaxPool2d(
            Tensor input,
            int kernelSize,
            int stride = -1,
            int padding = 0)
        {
            return Pool2d(
                input,
                MaxPoolInit,
                MaxPoolAccumulate,
                MaxPoolFinalize,
                kernelSize, stride, padding
            );
        }

        public static (Tensor, int[]) MaxPool2d(
            Tensor input,
            int kernelH, int kernelW,
            int strideH, int strideW,
            int padH, int padW)
        {
            return Pool2d(
                input,
                MaxPoolInit,
                MaxPoolAccumulate,
                MaxPoolFinalize,
                kernelH, kernelW,
                strideH, strideW,
                padH, padW
            );
        }
    }
}