using EmbodiedUI.Protocol;
using UnityEngine;

namespace EmbodiedUI
{
    /// <summary>
    /// A sound for an embodied-ui Speaker: "chime.level" (dB SPL at the
    /// speaker's reference distance) and "chime.frequency" (Hz). The level
    /// is 0 while silent. Unity's volume is not calibrated, so the level at
    /// full volume is set here and scaled by the source's volume.
    /// </summary>
    [AddComponentMenu("Embodied UI/Audio Output")]
    public sealed class EmbodiedAudioOutput : EmbodiedOutput
    {
        [Tooltip("The embodied-ui Speaker's name.")]
        public string device = "chime";
        [Tooltip("Reports while this source plays; or set 'playing' from scripts.")]
        public AudioSource follow;
        public bool playing;
        [Tooltip("Sound pressure level at full volume, dB SPL at the reference distance.")]
        public float levelAtFullVolume = 70f;
        [Tooltip("Dominant frequency, Hz.")]
        public float frequency = 1000f;

        public override void Collect(Outputs outputs, EmbodiedBridge bridge, bool capture)
        {
            var on = follow != null ? follow.isPlaying : playing;
            var volume = follow != null ? follow.volume : 1f;
            var level = on && volume > 0f ? levelAtFullVolume + 20f * Mathf.Log10(volume) : 0f;
            outputs.Set(device + ".level", level);
            outputs.Set(device + ".frequency", frequency);
        }
    }
}
