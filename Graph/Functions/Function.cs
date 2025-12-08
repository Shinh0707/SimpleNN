namespace SimpleNN.Graph.Functions
{
    using UnityEngine;
    using System.Text;
    using SimpleNN.Tensor;
    public class ContextData
    {
        public Tensor[] tensorData;
        public float[] valueData;
    }
    public class Context
    {
        internal delegate Tensor[] BackwardFunction(Context ctx, Tensor grad);
        internal delegate Tensor ForwardFunction(Context ctx);
        private Tensor _outputTensor = null;
        private Context[] _inputCtx = null;
        internal BackwardFunction _gradFn = null;
        private Tensor _grad = null;
        public Tensor Grad => _grad;
        public Tensor Tensor => _outputTensor;
        private object[] _registeredDatas;
        private Tensor[] _registeredTensors;
        private bool _requireGrad;
        public bool RequireGrad
        {
            get {return _requireGrad;}
            set
            {
                _requireGrad = value && Config.UseGrad;
            }
        }
        internal Context(Tensor tensor, bool requireGrad = true)
        {
            RequireGrad = requireGrad;
            _outputTensor = tensor;
            //CheckInfinite(_outputTensor, () => $"at assigned tensor.");
        }
        internal Context(Context[] inputCtx, ForwardFunction forFn, BackwardFunction gradFn)
        {
            _inputCtx = inputCtx;
            /*
            for (int i = 0; i < _inputCtx.Length; i++)
            {
                CheckInfinite(_inputCtx[i].Tensor, () => $"at Forward:{forFn.Target} input {i}.");
            }*/
            _gradFn = gradFn;
            _outputTensor = forFn(this);
            //Debug.Log($"(Forward) {forFn.Target}, {this}");
            //CheckInfinite(_outputTensor, () => $"at Forward:{forFn.Target} output.");
        }
        public void RegisterTensors(params Tensor[] tensors)
        {
            _registeredTensors = tensors;
        }
        public Tensor GetRegisteredTensor(int index) => _registeredTensors[(index < 0) ? (_registeredTensors.Length + index) : index];
        public void RegisterDatas(params object[] data)
        {
            _registeredDatas = data;
        }
        public T GetRegisteredData<T>(int index) => (T)_registeredDatas[(index < 0) ? _registeredDatas.Length + index : index];
        internal Tensor GetInput(int index)
        {
            return _inputCtx[(index < 0) ? (_inputCtx.Length+index) : index]._outputTensor;
        }
        internal bool TryGetInput(int index, out Tensor tensor)
        {
            tensor = null;
            if (_inputCtx == null) return false;
            if (index < 0) return TryGetInput(index, out tensor);
            if (index >= _inputCtx.Length) return false;
            tensor = _inputCtx[index]._outputTensor;
            return true;
        }
        public void ZeroGrad()
        {
            _grad = null;
        }
        public delegate Tensor StepFunction(Tensor tensor, Tensor grad);
        public void Step(StepFunction stepFunc)
        {
            if ((_outputTensor is not null) && (_grad is not null))
            {
                //Debug.Log($"(Tensor Updated) {_outputTensor} += {_grad} | {this}");
                /*CheckInfinite(_grad, () => $"at step {stepFunc.Target} grad.");
                CheckInfinite(_outputTensor, () => $"at before step {stepFunc.Target}.");*/
                _outputTensor = stepFunc(_outputTensor, _grad);
                //CheckInfinite(_outputTensor, () => $"at after step {stepFunc.Target}.");
            }
        }
        public void Backward() => Backward(_outputTensor);
        public void Backward(Tensor inputGrad)
        {
            if (_gradFn == null)
            {
                if (!RequireGrad) return;
                //Debug.Log($"(Grad Updated) {_grad} += {inputGrad} | {this}");
                //CheckInfinite(inputGrad, () => $"at assigned grad.");
                _grad = (_grad is null) ? inputGrad : (_grad + inputGrad);
                return;
            }
            //Debug.Log($"(Backward) {this}");
            /*CheckInfinite(inputGrad, () => $"at Backward:{_gradFn.Target} input grad.");
            for (int i = 0; i < _inputCtx.Length; i++)
            {
                CheckInfinite(_inputCtx[i].Tensor, () => $"at Backward:{_gradFn.Target} input {i}.");
            }*/
            var grads = _gradFn(this, inputGrad);
            for (int i = 0; i < _inputCtx.Length; i++)
            {
                //CheckInfinite(grads[i], () => $"at Backward:{_gradFn.Target} output {i}.");
                _inputCtx[i].Backward(grads[i]);
            }
        }
        private delegate string InfiniteMessage();
        private void CheckInfinite(Tensor tensor, InfiniteMessage msg)
        {
            if (tensor.HasInfinite(out float infv))
            {
                Debug.LogWarning($"{infv} was detected! {msg()} ({this})");
            }
        }
        /// <summary>
        /// コンテキストの情報（保持しているテンソル、勾配関数、入力元）を文字列として返す.
        /// </summary>
        public override string ToString()
        {
            StringBuilder sb = new();
            
            sb.Append("Context(");
            
            // Output Tensor
            sb.Append("Tensor: ");
            if (_outputTensor is not null)
            {
                sb.Append(_outputTensor.ToString());
            }
            else
            {
                sb.Append("null");
            }
            sb.Append(", ");

            // GradFn (関数名を表示)
            sb.Append("GradFn: ");
            if (_gradFn != null)
            {
                // MethodInfoからメソッド名を取得
                sb.Append(_gradFn.Target);
            }
            else
            {
                sb.Append("null");
            }
            sb.Append(", ");

            // Inputs (深さ1: 再帰せずに存在のみを示す)
            sb.Append("Inputs: ");
            if (_inputCtx != null && _inputCtx.Length > 0)
            {
                sb.Append("[");
                for (int i = 0; i < _inputCtx.Length; i++)
                {
                    // 中身を展開せず、Context型であることを示すのみに留める
                    sb.Append("Ctx");
                    if (i < _inputCtx.Length - 1)
                    {
                        sb.Append(", ");
                    }
                }
                sb.Append("]");
            }
            else
            {
                sb.Append("None");
            }

            sb.Append(")");
            return sb.ToString();
        }
    }
    public abstract class Function<T> where T : Function<T>, new()
    {
        internal abstract Tensor Forward(Context ctx);
        internal abstract Tensor[] Backward(Context ctx, Tensor grad);
    }
    public abstract class SingleFunction<T> : Function<T> where T : Function<T>, new()
    {
        public static Context Forward(params Tensor[] inputTensors)
        {
            var inputCtx = new Context[inputTensors.Length];
            for (int i = 0; i < inputTensors.Length; i++)
            {
                inputCtx[i] = new(inputTensors[i]);
            }
            return Forward(inputCtx);
        }
        public static Context Forward(params Context[] inputCtx)
        {
            var func = new T();
            return new Context(inputCtx, func.Forward, func.Backward);
        }
    }
    public abstract class KwargsFunction<T,U> : Function<T> where T : KwargsFunction<T,U>, new() where U : class, new()
    {
        private U _kwargs;
        public static Context Forward(U kwargs, params Tensor[] inputTensors)
        {
            var inputCtx = new Context[inputTensors.Length];
            for (int i = 0; i < inputTensors.Length; i++)
            {
                inputCtx[i] = new(inputTensors[i]);
            }
            return Forward(kwargs, inputCtx);
        }
        public static Context Forward(U kwargs, params Context[] inputCtx)
        {
            var func = new T
            {
                _kwargs = kwargs ?? new()
            };
            return new Context(inputCtx, func.Forward, func.Backward);
        }
        internal override Tensor Forward(Context ctx)
        {
            return Forward(_kwargs, ctx);
        }
        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            return Backward(_kwargs, ctx, grad);
        }
        internal abstract Tensor Forward(U kwargs,Context ctx);
        internal abstract Tensor[] Backward(U kwargs, Context ctx, Tensor grad);
    }
}