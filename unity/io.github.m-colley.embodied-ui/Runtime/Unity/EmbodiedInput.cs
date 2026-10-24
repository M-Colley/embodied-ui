// The simulated body as Unity input hardware.
//
// Every control of a session becomes one float control of a virtual Input
// System device laid out as "EmbodiedUI" (names with '.' become '_', so
// "start.pressed" binds as "<EmbodiedUI>/start_pressed"). Controls ending in
// ".pressed" or ".state" are buttons. Touch events become touches of a
// virtual Touchscreen, so uGUI and UI Toolkit receive them like real touches.

using System;
using System.Collections.Generic;
using EmbodiedUI.Protocol;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.Layouts;
using UnityEngine.InputSystem.LowLevel;
using UnityEngine.InputSystem.Utilities;
using TouchPhase = UnityEngine.InputSystem.TouchPhase;

namespace EmbodiedUI
{
    public sealed class EmbodiedInputDevice : IDisposable
    {
        public const string Layout = "EmbodiedUI";

        readonly InputDevice device;
        readonly Dictionary<string, AxisControl> map = new Dictionary<string, AxisControl>();

        public EmbodiedInputDevice(IList<string> controls)
        {
            var names = new List<string>(controls);
            InputSystem.RegisterLayoutBuilder(() => BuildLayout(names), Layout);
            device = InputSystem.AddDevice(Layout);
            foreach (var name in names)
                map[name] = device.TryGetChildControl<AxisControl>(ControlName(name));
        }

        public InputDevice Device => device;

        public static string ControlName(string channel)
        {
            var chars = channel.ToCharArray();
            for (int i = 0; i < chars.Length; i++)
                if (!char.IsLetterOrDigit(chars[i])) chars[i] = '_';
            return new string(chars);
        }

        static bool IsButton(string channel)
        {
            return channel.EndsWith(".pressed") || channel.EndsWith(".state");
        }

        static InputControlLayout BuildLayout(IList<string> names)
        {
            var builder = new InputControlLayout.Builder
            {
                name = Layout,
                type = typeof(InputDevice),
                stateFormat = new FourCC('E', 'M', 'B', 'D'),
                stateSizeInBytes = Math.Max(4, 4 * names.Count),
            };
            for (int i = 0; i < names.Count; i++)
            {
                builder.AddControl(ControlName(names[i]))
                    .WithLayout(IsButton(names[i]) ? "Button" : "Axis")
                    .WithFormat(InputStateBlock.FormatFloat)
                    .WithByteOffset((uint)(4 * i))
                    .WithDisplayName(names[i]);
            }
            return builder.Build();
        }

        /// <summary>Queue one state event with every control's value (NaN as 0).</summary>
        public void Write(IReadOnlyDictionary<string, double> values)
        {
            using (StateEvent.From(device, out var eventPtr))
            {
                foreach (var pair in map)
                {
                    if (pair.Value == null) continue;
                    var value = values.TryGetValue(pair.Key, out var v) && !double.IsNaN(v) ? (float)v : 0f;
                    pair.Value.WriteValueIntoEvent(value, eventPtr);
                }
                InputSystem.QueueEvent(eventPtr);
            }
        }

        public void Dispose()
        {
            if (device != null && device.added) InputSystem.RemoveDevice(device);
        }
    }

    /// <summary>Touch surfaces as a virtual touchscreen.</summary>
    public sealed class EmbodiedTouch : IDisposable
    {
        /// <summary>
        /// Map a touch surface's (u, v) to pixels; by default the whole screen.
        /// Assign a function to route a surface to part of the screen or to a
        /// camera that renders an in-vehicle display.
        /// </summary>
        public static Func<string, double, double, Vector2> ToScreen = (surface, u, v) =>
            new Vector2((float)(u * Screen.width), (float)((1 - v) * Screen.height));

        Touchscreen touchscreen;
        readonly HashSet<int> active = new HashSet<int>();

        public void Handle(DeviceEvent e)
        {
            TouchPhase phase;
            switch (e.Kind)
            {
                case "touch_begin": phase = TouchPhase.Began; break;
                case "touch_move": phase = TouchPhase.Moved; break;
                case "touch_end": phase = TouchPhase.Ended; break;
                default: return;
            }
            if (touchscreen == null) touchscreen = InputSystem.AddDevice<Touchscreen>("EmbodiedTouchscreen");
            var id = (int)e.Value;
            if (phase == TouchPhase.Began) active.Add(id);
            else if (phase == TouchPhase.Ended) active.Remove(id);
            InputSystem.QueueStateEvent(touchscreen, new TouchState
            {
                touchId = id,
                phase = phase,
                position = ToScreen(e.Source, e.Get("u", 0), e.Get("v", 0)),
                pressure = (float)e.Get("force", 0),
            });
        }

        /// <summary>End every active touch, e.g. on reset.</summary>
        public void Reset()
        {
            if (touchscreen == null) return;
            foreach (var id in active)
                InputSystem.QueueStateEvent(touchscreen, new TouchState { touchId = id, phase = TouchPhase.Canceled });
            active.Clear();
        }

        public void Dispose()
        {
            Reset();
            if (touchscreen != null && touchscreen.added) InputSystem.RemoveDevice(touchscreen);
            touchscreen = null;
        }
    }
}
