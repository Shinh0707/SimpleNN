using SimpleNN.Graph;

namespace SimpleNN.Module
{
    public class Reshape : SingleInSingleOutModule
    {
        private int[] _size;
        public Reshape(params int[] size)
        {
            _size = size;
        }
        public override TensorBox Forward(TensorBox inputTensor)
        {
            return inputTensor.Reshape(_size);
        }
    }
    public class Flatten : SingleInSingleOutModule
    {
        private int _startIndex;
        private int _endIndex;
        public Flatten(int startIndex = 0, int endIndex = -1)
        {
            _startIndex = startIndex;
            _endIndex = endIndex;
        }
        public override TensorBox Forward(TensorBox inputTensor)
        {
            return inputTensor.Flatten(_startIndex, _endIndex);
        }
    }
    public class Transpose : SingleInSingleOutModule
    {
        private int _dim0;
        private int _dim1;
        public Transpose(int dim0, int dim1)
        {
            _dim0 = dim0;
            _dim1 = dim1;
        }
        public override TensorBox Forward(TensorBox inputTensor)
        {
            return inputTensor.Transpose(_dim0, _dim1);
        }
    }
}