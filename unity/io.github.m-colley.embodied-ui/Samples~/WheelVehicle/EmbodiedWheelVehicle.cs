// A minimal WheelCollider car driven by the embodied-ui vehicle controls
// (vehicle.throttle, vehicle.brake in [0, 1]; vehicle.steer in [-1, 1],
// positive to the right). Add EmbodiedVehicleOutput to report telemetry.

using UnityEngine;

namespace EmbodiedUI.Samples
{
    [RequireComponent(typeof(Rigidbody))]
    public sealed class EmbodiedWheelVehicle : MonoBehaviour
    {
        public WheelCollider frontLeft, frontRight, rearLeft, rearRight;
        public float maxMotorTorque = 1500f;
        public float maxBrakeTorque = 3000f;
        public float maxSteerAngle = 35f;

        void FixedUpdate()
        {
            var bridge = EmbodiedBridge.Active;
            if (bridge == null) return;
            var throttle = Mathf.Clamp01(bridge.GetControl("vehicle.throttle"));
            var brake = Mathf.Clamp01(bridge.GetControl("vehicle.brake"));
            var steer = Mathf.Clamp(bridge.GetControl("vehicle.steer"), -1f, 1f);
            frontLeft.steerAngle = frontRight.steerAngle = steer * maxSteerAngle;
            rearLeft.motorTorque = rearRight.motorTorque = throttle * maxMotorTorque;
            foreach (var wheel in new[] { frontLeft, frontRight, rearLeft, rearRight })
                wheel.brakeTorque = brake * maxBrakeTorque;
        }
    }
}
