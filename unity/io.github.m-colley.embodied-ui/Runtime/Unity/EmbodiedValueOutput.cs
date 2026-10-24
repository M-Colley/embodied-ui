using EmbodiedUI.Protocol;
using UnityEngine;

namespace EmbodiedUI
{
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
