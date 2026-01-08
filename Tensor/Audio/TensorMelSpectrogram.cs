namespace SimpleNN.Tensor
{
    using System;
    using SimpleNN.Tensor.Audio;

    public enum MelSpecScaleType
    {
        Magnitude, // 幅をそのまま出力
        MinMaxNormalizedMagnitude, // 幅を0-1に正規化
        Decibel, // 幅をデシベルに変換
        MinMaxNormalizedDecibel, // 幅をデシベルに変換して0-1に正規化
        NormalizedDecibel
    }
    public enum MelSpecReductionType
    {
        None, // 何もしない
        TimeMean, // 時間方向に平均をとる
        TimeMax, // 時間方向に最大値をとる
    }
    public class TensorMelSpectrogram
    {
        private int n_fft;
        private int hop_length;
        private int win_length;
        private Tensor window;
        private bool center;
        private PaddingMode mode;
        private bool normalized;
        private bool onesided;
        private float power;
        private MelSpecScaleType scaleType;
        private MelSpecReductionType reductionType;

        private Tensor melBasis;

        // MelSpectrogramのパラメータをここで設定(メルフィルタバンクの設定など)
        // STFTのパラメータもここで設定する
        public TensorMelSpectrogram(
            int sample_rate = 22050,
            int n_fft = 2048,
            int hop_length = 512,
            int win_length = -1,
            int n_mels = 128,
            int f_min = 0,
            int? f_max = null,
            Tensor window = null,
            float power = 2.0f,
            bool center = true,
            PaddingMode mode = PaddingMode.REFLECT,
            bool normalized = false,
            bool onesided = true,
            MelSpecScaleType scaleType = MelSpecScaleType.Decibel,
            MelSpecReductionType reductionType = MelSpecReductionType.None
        )
        {
            this.n_fft = n_fft;
            this.hop_length = hop_length;
            this.win_length = win_length <= 0 ? n_fft : win_length;
            this.window = window;
            this.center = center;
            this.mode = mode;
            this.normalized = normalized;
            this.onesided = onesided;
            this.power = power;
            this.scaleType = scaleType;
            this.reductionType = reductionType;

            float fMaxVal = f_max ?? (sample_rate / 2.0f);
            this.melBasis = CreateMelFilterBank(sample_rate, n_fft, n_mels, f_min, fMaxVal);
        }

        // 入力の音声をMelSpectrogramに変換する
        public Tensor FromAudio(Tensor audio)
        {
            // STFT expects (..., time)
            // TensorFFT.STFT returns (freq, time, 2)
            var stft = TensorFFT.STFT(audio, n_fft, hop_length, win_length, window, center, mode, normalized, onesided);
            return FromSTFT(stft);
        }

        // STFTをMelSpectrogramに変換する
        public Tensor FromSTFT(Tensor stft)
        {
            Tensor magnitude;
            var parts = Tensor.Unstack(stft.NDim - 1, stft); // Split real/imag
            var real = parts[0];
            var imag = parts[1];
            
            var magSq = Tensor.Square(real) + Tensor.Square(imag);
            
            if (power == 2.0f)
            {
                magnitude = magSq;
            }
            else
            {
                magnitude = Tensor.Pow(magSq, power / 2.0f); 
            }

            Tensor melSpec = Tensor.MatMul(melBasis, magnitude);

            switch (reductionType)
            {
                case MelSpecReductionType.TimeMean:
                    melSpec = Tensor.Mean(melSpec, 1, false); // dim 1 is time
                    break;
                case MelSpecReductionType.TimeMax:
                    melSpec = Tensor.Max(melSpec, 1, false);
                    break;
                case MelSpecReductionType.None:
                default:
                    break;
            }
            switch (scaleType)
            {
                case MelSpecScaleType.Magnitude:
                    // Do nothing
                    break;
                case MelSpecScaleType.Decibel:
                    melSpec = AmplitudeToDB(melSpec);
                    break;
                case MelSpecScaleType.MinMaxNormalizedMagnitude:
                    melSpec = MinMaxNormalize(melSpec);
                    break;
                case MelSpecScaleType.MinMaxNormalizedDecibel:
                    melSpec = MinMaxNormalize(AmplitudeToDB(melSpec));
                    break;
                case MelSpecScaleType.NormalizedDecibel:
                    melSpec = Normalize(AmplitudeToDB(melSpec), -20, 0);
                    break;
            }
            return melSpec;
        }

        private Tensor AmplitudeToDB(Tensor x)
        {
            // 10 * log10(x) if power, 20 * log10(x) if magnitude
            // We assumed 'x' here is what we got from MatMul.
            // If power=2, x is power. 10 * log10(x).
            // If power=1, x is magnitude. 20 * log10(x).
            
            // Avoid log(0)
            float amin = 1e-10f;
            
            // log10(x) = ln(x) / ln(10)
            float log10Div = 1.0f / MathF.Log(10.0f);
            
            Tensor logSpec = Tensor.Log(Tensor.Maximum(x, amin)) * log10Div;
            
            float factor = (power == 2.0f) ? 10.0f : 20.0f;
            
            return logSpec * factor;
        }

        private Tensor MinMaxNormalize(Tensor x)
        {
            float minVal = -Tensor.Max(-x).Item(); // Min
            float maxVal = Tensor.Max(x).Item();
            return Normalize(x, minVal, maxVal);
        }

        private Tensor Normalize(Tensor x, float min, float max)
        {   
            if (MathF.Abs(max - min) < 1e-6f)
            {
                return Tensor.ZerosLike(x);
            }   
            return (Tensor.Maximum(x,min) - min) / (max - min);
        }

        private Tensor CreateMelFilterBank(int sampleRate, int nFft, int nMels, int fMin, float fMax)
        {
            // 1. Convert to Mel
            float melMin = HzToMel(fMin);
            float melMax = HzToMel(fMax);

            // 2. Create Mel points
            float[] melPoints = new float[nMels + 2];
            for (int i = 0; i < melPoints.Length; i++)
            {
                melPoints[i] = melMin + (melMax - melMin) * i / (nMels + 1);
            }

            // 3. Convert back to Hz and find bin indices
            int[] binPoints = new int[nMels + 2];
            for (int i = 0; i < melPoints.Length; i++)
            {
                float hz = MelToHz(melPoints[i]);
                // bin = hz * (n_fft + 1) / sample_rate  <-- Wait, n_fft/2 + 1 bins map to 0..Nyquist
                // Freq resolution = sample_rate / n_fft
                // bin_index = hz / (sample_rate / n_fft) = hz * n_fft / sample_rate
                // But we need to map to 0..(n_fft/2)
                binPoints[i] = (int)MathF.Floor((n_fft + 1) * hz / sampleRate);
                // Clamp to valid bins
                if (binPoints[i] > n_fft / 2) binPoints[i] = n_fft / 2;
            }

            // 4. Create Filter Bank Matrix (nMels x (n_fft/2 + 1))
            int nFreq = n_fft / 2 + 1;
            float[] weights = new float[nMels * nFreq];

            for (int i = 0; i < nMels; i++)
            {
                int start = binPoints[i];
                int center = binPoints[i + 1];
                int end = binPoints[i + 2];

                for (int f = start; f < center; f++)
                {
                    if (f >= 0 && f < nFreq)
                        weights[i * nFreq + f] = (float)(f - start) / (center - start);
                }
                for (int f = center; f < end; f++)
                {
                    if (f >= 0 && f < nFreq)
                        weights[i * nFreq + f] = (float)(end - f) / (end - center);
                }
            }

            return new Tensor(weights, new int[] { nMels, nFreq });
        }

        private float HzToMel(float hz)
        {
            return 2595.0f * MathF.Log10(1.0f + hz / 700.0f);
        }

        private float MelToHz(float mel)
        {
            return 700.0f * (MathF.Pow(10.0f, mel / 2595.0f) - 1.0f);
        }
    }
}