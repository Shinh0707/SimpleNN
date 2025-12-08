using System;
using UnityEngine;

namespace SimpleNN.Util
{
    public static class ColorUtil
    {
        public static void RGBToHSV(float r, float g, float b, out float h, out float s, out float v)
        {
            Color.RGBToHSV(new Color(r, g, b), out h, out s, out v);
        }

        public static void HSVToConical(float h, float s, float v, out float x, out float y, out float z)
        {
            // User requirement: H=phi, S=r, V=h => r*h*cos(phi), r*h*sin(phi), h
            // H in Unity is 0-1, so convert to radians: phi = H * 2 * PI
            float phi = h * 2.0f * Mathf.PI;
            float r = s;
            float height = v;

            x = r * height * Mathf.Cos(phi);
            y = r * height * Mathf.Sin(phi);
            z = height;
        }
    }
}
