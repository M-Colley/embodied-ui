// Inspector-wired reactions to device events, e.g. a press of "start".

using EmbodiedUI.Protocol;
using UnityEngine;
using UnityEngine.Events;

namespace EmbodiedUI
{
    [AddComponentMenu("Embodied UI/Event Listener")]
    public sealed class EmbodiedEventListener : MonoBehaviour
    {
        [Tooltip("Device (or device.region) whose events to receive.")]
        public string source = "start";

        [Tooltip("press, release, toggle, detent, touch_begin, touch_move or touch_end; empty for all.")]
        public string kind = "press";

        [Tooltip("Called with the event value: the new state, detent index or touch id.")]
        public UnityEvent<float> onEvent = new UnityEvent<float>();

        void OnEnable() { EmbodiedBridge.EventReceived += Handle; }
        void OnDisable() { EmbodiedBridge.EventReceived -= Handle; }

        void Handle(DeviceEvent e)
        {
            if (e.Source == source && (string.IsNullOrEmpty(kind) || e.Kind == kind))
                onEvent.Invoke((float)e.Value);
        }
    }
}
