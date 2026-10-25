using EmbodiedUI.Protocol;
using UnityEngine;

namespace EmbodiedUI.Samples.Tests
{
    /// <summary>
    /// Reports what the sample test checks: frame time, physics steps and
    /// the wheel colliders' grounding, steer angle, motor and brake torques.
    /// </summary>
    public sealed class WheelVehicleProbe : EmbodiedOutput
    {
        public EmbodiedWheelVehicle vehicle;
        int fixedSteps;

        void FixedUpdate() { fixedSteps++; }

        public override void Collect(Outputs outputs, EmbodiedBridge bridge, bool capture)
        {
            outputs.Set("unity.delta_time", Time.deltaTime);
            outputs.Set("unity.fixed_steps", fixedSteps);
            var grounded = 0;
            foreach (var wheel in new[] { vehicle.frontLeft, vehicle.frontRight, vehicle.rearLeft, vehicle.rearRight })
                if (wheel.isGrounded) grounded++;
            outputs.Set("wheels.grounded", grounded);
            outputs.Set("wheels.steer_angle", vehicle.frontLeft.steerAngle);
            outputs.Set("wheels.motor_torque", vehicle.rearLeft.motorTorque);
            outputs.Set("wheels.brake_torque", vehicle.frontLeft.brakeTorque);
            outputs.Set("wheels.rear_rpm", vehicle.rearLeft.rpm);
        }
    }
}
