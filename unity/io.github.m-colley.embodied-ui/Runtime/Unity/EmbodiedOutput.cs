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
}
