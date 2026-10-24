// Values Unity reports to the MuJoCo side after each frame.

using System.Collections.Generic;
using EmbodiedUI.Protocol;
using UnityEngine;

namespace EmbodiedUI
{
    /// <summary>Base class: add components to report poses, lamps, images...</summary>
    public abstract class EmbodiedOutput : MonoBehaviour
    {
        static readonly List<EmbodiedOutput> all = new List<EmbodiedOutput>();
        public static IReadOnlyList<EmbodiedOutput> All => all;

        protected virtual void OnEnable() { all.Add(this); }
        protected virtual void OnDisable() { all.Remove(this); }

        /// <param name="capture">False for the welcome, where no frame was simulated yet.</param>
        public abstract void Collect(Outputs outputs, EmbodiedBridge bridge, bool capture);
    }

    /// <summary>A transform's pose, e.g. "cup.pose" for an embodied-ui Prop.</summary>
    [AddComponentMenu("Embodied UI/Pose Output")]
    public sealed class EmbodiedPoseOutput : EmbodiedOutput
    {
        public string outputName = "prop.pose";
        [Tooltip("Pose relative to this transform; the bridge reference if empty.")]
        public Transform relativeTo;

        public override void Collect(Outputs outputs, EmbodiedBridge bridge, bool capture)
        {
            outputs.Set(outputName, bridge.PoseOf(transform, relativeTo));
        }
    }

    /// <summary>
    /// Vehicle telemetry in SI units and Unity axes: pose, velocity,
    /// acceleration, angular velocity (rad/s) and speed, under a prefix.
    /// </summary>
    [AddComponentMenu("Embodied UI/Vehicle Output")]
    [RequireComponent(typeof(Rigidbody))]
    public sealed class EmbodiedVehicleOutput : EmbodiedOutput
    {
        public string prefix = "vehicle";
        Rigidbody body;
        Vector3 lastVelocity;
        bool hasLast;

        void Awake() { body = GetComponent<Rigidbody>(); }

        public override void Collect(Outputs outputs, EmbodiedBridge bridge, bool capture)
        {
#if UNITY_6000_0_OR_NEWER
            var velocity = body.linearVelocity;
#else
            var velocity = body.velocity;
#endif
            var acceleration = hasLast && capture ? (velocity - lastVelocity) / (float)bridge.Dt : Vector3.zero;
            lastVelocity = velocity;
            hasLast = capture;
            var origin = bridge.reference;
            var inverse = origin != null ? Quaternion.Inverse(origin.rotation) : Quaternion.identity;
            velocity = inverse * velocity;
            acceleration = inverse * acceleration;
            var omega = inverse * body.angularVelocity;
            outputs.Set(prefix + ".pose", bridge.PoseOf(transform));
            outputs.Set(prefix + ".velocity", velocity.x, velocity.y, velocity.z);
            outputs.Set(prefix + ".acceleration", acceleration.x, acceleration.y, acceleration.z);
            outputs.Set(prefix + ".angular_velocity", omega.x, omega.y, omega.z);
            outputs.Set(prefix + ".speed", velocity.magnitude);
        }
    }

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

    /// <summary>A lamp state, e.g. "hazard_lamp.on" for an embodied-ui Indicator.</summary>
    [AddComponentMenu("Embodied UI/Lamp Output")]
    public sealed class EmbodiedLampOutput : EmbodiedOutput
    {
        public string outputName = "lamp.on";
        [Tooltip("Reports whether this component is enabled, e.g. a Light; or set 'on' from scripts.")]
        public Behaviour follow;
        public bool on;

        public override void Collect(Outputs outputs, EmbodiedBridge bridge, bool capture)
        {
            outputs.Set(outputName, follow != null ? follow.isActiveAndEnabled : on);
        }
    }

    /// <summary>A number set from scripts or animations, e.g. "wheel.torque".</summary>
    [AddComponentMenu("Embodied UI/Value Output")]
    public sealed class EmbodiedValueOutput : EmbodiedOutput
    {
        public string outputName = "wheel.torque";
        public float value;

        public override void Collect(Outputs outputs, EmbodiedBridge bridge, bool capture)
        {
            outputs.Set(outputName, value);
        }
    }
}
