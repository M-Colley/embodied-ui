using EmbodiedUI.Protocol;
using UnityEngine;
using UnityEngine.Rendering;

namespace EmbodiedUI.PlayerTest
{
    /// <summary>
    /// Reports what the player test checks: frame time, physics steps, the
    /// throttle the input action read, and how the player runs.
    /// </summary>
    public sealed class PlayerProbe : EmbodiedOutput
    {
        public PlayerCar car;

        public override void Collect(Outputs outputs, EmbodiedBridge bridge, bool capture)
        {
            outputs.Set("unity.delta_time", Time.deltaTime);
            outputs.Set("unity.fixed_steps", car.fixedSteps);
            outputs.Set("unity.throttle_input", car.lastThrottle);
            outputs.Set("unity.is_editor", Application.isEditor);
            outputs.Set("unity.batch_mode", Application.isBatchMode);
            outputs.Set("unity.graphics_device", SystemInfo.graphicsDeviceType.ToString());
            outputs.Set("unity.renders", SystemInfo.graphicsDeviceType != GraphicsDeviceType.Null);
            outputs.Set("unity.version", Application.unityVersion);
        }
    }
}
