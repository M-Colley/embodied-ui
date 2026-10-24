using EmbodiedUI.Protocol;
using UnityEngine;

namespace EmbodiedUI
{
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
}
