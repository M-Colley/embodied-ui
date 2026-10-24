using EmbodiedUI.Protocol;
using UnityEngine;

namespace EmbodiedUI
{
    /// <summary>
    /// A camera image as an H x W x 3 uint8 attachment, top row first: a
    /// driver's view for perception models, or "screen.image" for a display.
    /// </summary>
    [AddComponentMenu("Embodied UI/Camera Output")]
    [RequireComponent(typeof(Camera))]
    public sealed class EmbodiedCameraOutput : EmbodiedOutput
    {
        public string outputName = "camera.main";
        public int width = 320;
        public int height = 180;
        [Tooltip("Capture every n-th frame; other frames omit the image.")]
        [Min(1)] public int everyNthFrame = 1;

        Camera source;
        RenderTexture target;
        Texture2D readback;
        byte[] flipped;

        void Awake() { source = GetComponent<Camera>(); }

        public override void Collect(Outputs outputs, EmbodiedBridge bridge, bool capture)
        {
            if (bridge.Frame % everyNthFrame != 0) return;
            if (target == null)
            {
                target = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
                readback = new Texture2D(width, height, TextureFormat.RGB24, false);
                flipped = new byte[width * height * 3];
            }
            var previousTarget = source.targetTexture;
            var previousActive = RenderTexture.active;
            source.targetTexture = target;
            source.Render();
            RenderTexture.active = target;
            readback.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
            readback.Apply(false);
            source.targetTexture = previousTarget;
            RenderTexture.active = previousActive;
            var raw = readback.GetRawTextureData();
            var row = width * 3;
            for (int y = 0; y < height; y++)
                System.Buffer.BlockCopy(raw, (height - 1 - y) * row, flipped, y * row, row);
            outputs.Image(outputName, height, width, 3, (byte[])flipped.Clone());
        }

        protected override void OnDisable()
        {
            base.OnDisable();
            if (target != null) target.Release();
            target = null;
        }
    }
}
