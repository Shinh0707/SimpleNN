
namespace SimpleNN.Module
{
    using System.Collections.Generic;
    using System.Text;
    using SimpleNN.Graph;
    public abstract class Module
    {
        protected Queue<Module> _children = new();
        protected Queue<TensorBox> _parameters = new();
        public List<TensorBox> GetParameters()
        {
            var ps = new List<TensorBox>();
            int pc = _parameters.Count;
            if ((_parameters != null) && (pc > 0))
            {
                ps.AddRange(_parameters);
            }
            int c = _children.Count;
            for (int i = 0; i < c; i++)
            {
                var child = _children.Dequeue();
                ps.AddRange(child.GetParameters());
                _children.Enqueue(child);
            }
            return ps;
        }
        public List<TensorBox> SetParameters(List<TensorBox> parameters)
        {
            int pc = _parameters.Count;
            if ((_parameters != null) && (pc > 0))
            {
                for (int i = 0; i < pc; i++)
                {
                    var p = _parameters.Dequeue();
                    p.GetTensor().SetData(parameters[i].GetTensor());
                    _parameters.Enqueue(p);
                }
            }

            var remain = parameters.GetRange(pc, parameters.Count - pc);

            int c = _children.Count;
            for (int i = 0; i < c; i++)
            {
                var child = _children.Dequeue();
                remain = child.SetParameters(remain);
                _children.Enqueue(child);
            }
            return remain;
        }
        protected T AddModule<T>(T module) where T : Module
        {
            _children.Enqueue(module);
            return module;
        }
        protected TensorBox AddParameter(TensorBox param)
        {
            _parameters.Enqueue(param);
            return param;
        }
        protected TensorBox[] AddParameters(params TensorBox[] param)
        {
            var _param = new TensorBox[param.Length];
            for(int i = 0; i < param.Length; i++)
            {
                _param[i] = AddParameter(param[i]);
            }
            return param;
        }
        protected List<TensorBox> AddParameters(List<TensorBox> param)
        {
            var _param = new TensorBox[param.Count];
            for(int i = 0; i < param.Count; i++)
            {
                _param[i] = AddParameter(param[i]);
            }
            return param;
        }
        /// <summary>
        /// モジュールの概要（子モジュール, 総パラメータ数, 総サイズ）を文字列で返す.
        /// </summary>
        /// <returns>モジュールの概要文字列</returns>
        public override string ToString()
        {
            var sb = new StringBuilder();
            ToString(sb, "");
            return sb.ToString();
        }

        /// <summary>
        /// ToStringの内部実装. StringBuilderに概要を追記する.
        /// </summary>
        /// <param name="sb">追記対象のStringBuilder</param>
        /// <param name="prefix">インデント用のプレフィックス文字列</param>
        private void ToString(StringBuilder sb, string prefix = "")
        {
            sb.Append(prefix);
            sb.Append(GetType().Name);
            sb.Append(" (");

            string indent = prefix + "    ";
            if (_children != null && _children.Count > 0)
            {
                sb.AppendLine("");
                int i = 0;
                foreach (var child in _children)
                {
                    sb.Append(indent);
                    sb.Append($"({i}): ");
                    if (child == null)
                    {
                        sb.AppendLine("null");
                    }
                    else if (child != this)
                    {
                        child.ToString(sb, indent); 
                    }
                    else
                    {
                        sb.AppendLine($"{GetType().Name} (Self reference)");
                    }
                    i++;
                }
                sb.Append(prefix); // 閉じ括弧のインデントを合わせる
                sb.AppendLine(")");
            }
            else
            {
                // 子がいない場合
                sb.Append(")"); // 例: "Module ()"
                sb.AppendLine();
            }

            sb.Append(prefix);
            sb.AppendLine("---");

            // --- パラメータ ---
            long totalSize = 0;
            int totalParamsCount = 0;

            if (_parameters != null)
            {
                totalParamsCount = _parameters.Count;
                foreach (var param in _parameters)
                {
                    if (param is null || param.Size == null)
                    {
                        continue;
                    }

                    long currentSize = 1;
                    int[] shape = param.Size;
                    int dims = shape.Length;
                    
                    if (dims == 0)
                    {
                        currentSize = 1;
                    }
                    else
                    {
                        for (int j = 0; j < dims; j++)
                        {
                            currentSize *= shape[j];
                        }
                    }
                    totalSize += currentSize;
                }
            }
            
            sb.Append(prefix);
            sb.AppendLine($"Total Parameters (Tensors): {totalParamsCount}");
            sb.Append(prefix);
            sb.AppendLine($"Total Size (Elements): {totalSize}");
        }
    }
}