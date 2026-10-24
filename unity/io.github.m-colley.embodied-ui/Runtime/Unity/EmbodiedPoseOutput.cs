using EmbodiedUI.Protocol;
using UnityEngine;

namespace EmbodiedUI
{
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
}
