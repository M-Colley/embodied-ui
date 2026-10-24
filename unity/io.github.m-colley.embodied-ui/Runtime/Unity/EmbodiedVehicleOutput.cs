using EmbodiedUI.Protocol;
using UnityEngine;

namespace EmbodiedUI
{
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
}
