using EmbodiedUI.Protocol;
using UnityEngine;

namespace EmbodiedUI
{
    /// <summary>
    /// A vibration for an embodied-ui Vibrotactor: "seat.amplitude" (peak
    /// acceleration, m/s²) and "seat.frequency" (Hz), set from scripts.
    /// </summary>
    [AddComponentMenu("Embodied UI/Vibration Output")]
    public sealed class EmbodiedVibrationOutput : EmbodiedOutput
    {
        [Tooltip("The embodied-ui Vibrotactor's name.")]
        public string device = "seat";
        [Tooltip("Peak acceleration, m/s².")]
        public float amplitude;
        [Tooltip("Hz.")]
        public float frequency = 250f;

        public override void Collect(Outputs outputs, EmbodiedBridge bridge, bool capture)
        {
            outputs.Set(device + ".amplitude", amplitude);
            outputs.Set(device + ".frequency", frequency);
        }
    }
}
