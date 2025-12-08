using System;

namespace SimpleNN.Util
{
    public static partial class Operation
    {
        private static readonly float LOG_SQRT_2_PI = MathF.Log(MathF.Sqrt(2.0f * MathF.PI));
        private static readonly float LOG_PI = MathF.Log(MathF.PI);
        private static readonly float SQ_PI = MathF.PI*MathF.PI;
        private const float GAMMA_G = 5;
        private static readonly float[] GAMMA_P = {
            1.0000018972739440364f,
            76.180082222642137322f,
            -86.505092037054859197f,
            24.012898581922685900f,
            -1.2296028490285820771f
        };
        public static float LGamma(float z)
        {
            if (z < 0.5f)
            {
                return LOG_PI - MathF.Log(MathF.Sin(MathF.PI * z)) - LGamma(1.0f - z);
            }
            var zm1 = z - 1;
            var x = GAMMA_P[0];
            for (int i = 1; i < GAMMA_P.Length; i++)
            {
                x += GAMMA_P[i] / (zm1 + i);
            }
            var t = zm1 + GAMMA_G + 0.5f;
            return LOG_SQRT_2_PI + (zm1 + 0.5f) * MathF.Log(t) - t + MathF.Log(x);
        }
        public static float Digamma(float z)
        {
            if (z < 0.5f)
            {
                float cot = MathF.Cos(MathF.PI * z) / MathF.Sin(MathF.PI * z);
                return Digamma(1.0f - z) - MathF.PI * cot;
            }
            var zm1 = z - 1;
            var t = zm1 + GAMMA_G + 0.5f;
            var x = GAMMA_P[0];
            for (int i = 1; i < GAMMA_P.Length; i++)
            {
                x += GAMMA_P[i] / (zm1 + i);
            }
            var dx_dz = 0.0f;
            for (int i = 1; i < GAMMA_P.Length; i++)
            {
                var term = zm1 + i;
                dx_dz -= GAMMA_P[i] / (term * term);
            }
            return MathF.Log(t) - (GAMMA_G / t) + (dx_dz / x);
        }
        public static float Trigamma(float z)
        {
            if (z < 0.5f)
            {
                float sin_pi_z = MathF.Sin(MathF.PI * z);
                float pi_csc_sq = SQ_PI / (sin_pi_z * sin_pi_z);
                return pi_csc_sq - Trigamma(1.0f - z);
            }
            var zm1 = z - 1;
            var t = zm1 + GAMMA_G + 0.5f;
            var x = GAMMA_P[0];
            var dx_dz = 0.0f;
            var d2x_dz2 = 0.0f;

            for (int i = 1; i < GAMMA_P.Length; i++)
            {
                var term = zm1 + i;
                var term_inv = 1.0f / term;
                var term_inv_sq = term_inv * term_inv;
                var term_inv_cb = term_inv_sq * term_inv;

                x       += GAMMA_P[i] * term_inv;
                dx_dz   -= GAMMA_P[i] * term_inv_sq;
                d2x_dz2 += 2.0f * GAMMA_P[i] * term_inv_cb;
            }
            var term1 = 1.0f / t;
            var term2 = GAMMA_G / (t * t);
            var term3_numerator = (d2x_dz2 * x) - (dx_dz * dx_dz);
            var term3 = term3_numerator / (x * x);
            return term1 + term2 + term3;
        }
    }
}