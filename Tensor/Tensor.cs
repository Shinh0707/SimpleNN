namespace SimpleNN.Tensor
{
    using System;
    using System.Text;
    using SimpleNN.Util;

    public partial class Tensor
    {
        protected float[] _data;
        public float[] Data => _data;
        protected int[] _size;
        public int[] Size
        {
            get { return _size; }
            set
            {
                _size = value;
                Refresh();
            }
        }
        private ulong _totalSize = 0;
        public ulong TotalSize => _totalSize;
        private int[] _strides;
        public int[] Strides
        {
            get { return _strides; }
        }
        private int _ndim = 0;
        public int NDim => _ndim;
        private bool _isContiguous = true;
        public bool IsContiguous => _isContiguous;
        private bool _isScaler = false;
        public bool IsScaler => _isScaler;
        public float Item()
        {
            if (!IsScaler)
            {
                throw new NotImplementedException("This is not scaler tensor.");
            }
            return _data[0];
        }
        public Tensor(float[] data, int[] size, int[] strides)
        {
            _data = data;
            _size = size;
            _strides = strides;
            Refresh();
        }
        public static Tensor Clone(Tensor other)
        {
            return new(
                (float[])other.Data.Clone(),
                (int[])other.Size.Clone(),
                (int[])other.Strides.Clone()
            );
        }
        public void SetData(Tensor other)
        {
            SetData(other._data);
        }
        public void SetData(float[] data)
        {
            if (data.Length == _data.Length)
            {
                Array.Copy(data, _data, data.Length);
                return;
            }
            long requiredSize = 1;
            if (_size.Length > 0)
            {
                long maxOffset = 0;
                for (int i = 0; i < _size.Length; i++)
                {
                    if (_size[i] > 1)
                    {
                        maxOffset += (_size[i] - 1) * _strides[i];
                    }
                }
                requiredSize = maxOffset + 1;
            }

            if (data.Length < requiredSize)
            {
                throw new ArgumentException($"Data length {data.Length} is too small for tensor with size [{string.Join(", ", _size)}] and strides [{string.Join(", ", _strides)}]. Required: {requiredSize}");
            }

            _data = data;
        }
        private void Refresh()
        {
            _ndim = _size.Length;
            _totalSize = 1;
            uint d;
            for (uint i = 0; i < _size.Length; i++)
            {
                if (_size[i] <= 1) continue;
                d = (uint)_size[i];
                _totalSize *= Math.Max(d, 1);
            }
            _isScaler = _totalSize == 1;
            _isContiguous = Util.Strides.IsCContiguous(_size, _strides);
        }
        public static bool IsSameSizeAndStrides(Tensor a, Tensor b)
        {
            return ArrayExt.IsSameArray(a.Size, b.Size) && ArrayExt.IsSameArray(a.Strides, b.Strides);
        }
        public static bool IsSameSize(Tensor a, Tensor b)
        {
            if (a.NDim != b.NDim) return false;
            for (int i = 0; i < a.NDim; i++)
            {
                if (a.Size[i] != b.Size[i]) return false;
            }
            return true;
        }
        public static implicit operator Tensor(float value)
        {
            return new(new[] { value }, new[] { 1 });
        }
        public static implicit operator Tensor(int value)
        {
            return new(new float[] { value }, new[] { 1 });
        }
        public static implicit operator Tensor(float[] values)
        {
            return new(values, new[] { values.Length });
        }
        public static implicit operator Tensor(int[] values)
        {
            float[] fvalues = new float[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                fvalues[i] = values[i];
            }
            return new(fvalues, new[] { values.Length });
        }
        public static implicit operator Tensor(float[,] values)
        {
            int height = values.GetLength(0);
            int width = values.GetLength(1);
            float[] flatData = new float[height * width];
            Buffer.BlockCopy(values, 0, flatData, 0, flatData.Length * sizeof(float));
            return new(flatData, new[] { height, width });
        }
        public bool HasNaN()
        {
            foreach (var d in _data)
            {
                if (float.IsNaN(d))
                {
                    return true;
                }
            }
            return false;
        }
        public bool HasInfinite(out float infiniteValue)
        {
            foreach (var d in _data)
            {
                if (!float.IsFinite(d))
                {
                    infiniteValue = d;
                    return true;
                }
            }
            infiniteValue = 0;
            return false;
        }

        public override bool Equals(object obj)
        {
            if (obj is not Tensor other) return false;
            if (ReferenceEquals(this, other)) return true;
            if (!IsSameSizeAndStrides(this, other)) return false;
            if (_data.Length != other._data.Length) return false;

            for (int i = 0; i < _data.Length; i++)
            {
                if (!_data[i].Equals(other._data[i])) return false;
            }
            return true;
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                if (_size != null)
                {
                    foreach (var s in _size) hash = hash * 23 + s;
                }
                if (_strides != null)
                {
                    foreach (var s in _strides) hash = hash * 23 + s;
                }
                if (_data != null)
                {
                    foreach (var d in _data) hash = hash * 23 + d.GetHashCode();
                }
                return hash;
            }
        }
        public override string ToString()
        {
            StringBuilder sb = new();

            Util.StringExt.ArrayString(sb, "Size", _size);
            sb.Append(", ");

            Util.StringExt.ArrayString(sb, "Strides", _strides);
            sb.Append(", ");

            sb.Append("Data:[");
            if (_data != null)
            {
                int previewLimit = 5;
                int count = (_data.Length < previewLimit) ? _data.Length : previewLimit;
                
                for (int i = 0; i < count; i++)
                {
                    sb.Append(_data[i].ToString("F2")); // 小数点2桁まで
                    if (i < count - 1)
                    {
                        sb.Append(", ");
                    }
                }

                if (_data.Length > previewLimit)
                {
                    sb.Append(", ...");
                }
            }
            sb.Append("]");

            return sb.ToString();
        }
    }
}