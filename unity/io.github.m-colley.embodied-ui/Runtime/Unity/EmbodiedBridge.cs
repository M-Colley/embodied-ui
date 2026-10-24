// Lockstep server inside a Unity player or the Editor's play mode.
//
// One request is handled at the very start of each frame, before Unity
// updates its clock: a step's controls are written to a virtual Input
// System device (and touches to a virtual touchscreen), the frame runs
// with Time.captureDeltaTime equal to the shared period, and after
// rendering the outputs of every EmbodiedOutput are returned.

using System;
using System.Collections.Generic;
using EmbodiedUI.Protocol;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.LowLevel;

namespace EmbodiedUI
{
    [DisallowMultipleComponent]
    [AddComponentMenu("Embodied UI/Embodied Bridge")]
    public sealed class EmbodiedBridge : MonoBehaviour
    {
        [Tooltip("Listening address. 127.0.0.1 accepts local clients only.")]
        public string host = "127.0.0.1";

        public int port = Wire.DefaultPort;

        [Tooltip("Physics steps per lockstep frame.")]
        [Min(1)] public int physicsSubsteps = 1;

        [Tooltip("Process Input System events once per lockstep frame, so FixedUpdate and Update read the same values.")]
        public bool driveInputUpdates = true;

        [Tooltip("Seconds to wait for the next request before letting the game run freely.")]
        public float requestTimeout = 30f;

        [Tooltip("Block the first frame until a client connects (players started by scripts).")]
        public bool waitForClient;

        [Tooltip("Origin of exported devices and reported poses; the world if empty.")]
        public Transform reference;

        public string engineName = "Unity";

        /// <summary>The active bridge, for scripts that read controls.</summary>
        public static EmbodiedBridge Active { get; private set; }

        /// <summary>Raised for every device event of a step (press, toggle, touch...).</summary>
        public static event Action<DeviceEvent> EventReceived;

        /// <summary>Raised on reset requests; restore your scenario here.</summary>
        public static event Action<int?, JObject> ResetRequested;

        public bool InSession { get; private set; }
        public long Frame { get; private set; }
        public double Dt { get; private set; }
        public double SessionTime => Frame * Dt;

        readonly Dictionary<string, double> controls = new Dictionary<string, double>();
        readonly Dictionary<string, JToken> customOutputs = new Dictionary<string, JToken>();
        readonly Outputs outputs = new Outputs();
        LockstepServer server;
        EmbodiedInputDevice inputDevice;
        EmbodiedTouch touch;
        bool stepPending, resetPending, everConnected;
        float savedCapture, savedFixed, savedMaximum;
        InputSettings.UpdateMode savedMode;

        // Controls ------------------------------------------------------------

        public IReadOnlyDictionary<string, double> Controls => controls;

        public bool TryGetControl(string name, out double value)
        {
            return controls.TryGetValue(name, out value) && !double.IsNaN(value);
        }

        public float GetControl(string name, float fallback = 0f)
        {
            return TryGetControl(name, out var value) ? (float)value : fallback;
        }

        /// <summary>Report a value with this frame's state (cleared after sending).</summary>
        public void SetOutput(string name, double value)
        {
            customOutputs[name] = Wire.Number(value);
        }

        public void SetOutput(string name, params double[] values)
        {
            var array = new JArray();
            foreach (var v in values) array.Add(Wire.Number(v));
            customOutputs[name] = array;
        }

        // Lifecycle -----------------------------------------------------------

        void OnEnable()
        {
            if (Active != null && Active != this)
            {
                Debug.LogError("Only one EmbodiedBridge may be active.", this);
                enabled = false;
                return;
            }
            Active = this;
            server = new LockstepServer(host, port);
            PlayerLoopHook.Install(BeforeFrame, AfterFrame);
            Debug.Log($"Embodied UI listening on {host}:{server.Port}", this);
        }

        void OnDisable()
        {
            if (Active != this) return;
            PlayerLoopHook.Uninstall();
            EndSession();
            server?.Dispose();
            server = null;
            Active = null;
        }

        // Frame hooks ---------------------------------------------------------

        void BeforeFrame()
        {
            if (server == null) return;
            if (!server.HasClient)
            {
                var wait = waitForClient && !everConnected ? -1 : 0;
                if (!server.Accept(wait)) return;
                everConnected = true;
            }
            var timeout = requestTimeout > 0 ? (int)(requestTimeout * 1000) : -1;
            while (true)
            {
                var request = server.Next(timeout);
                if (request == null)
                {
                    EndSession();
                    return;
                }
                switch (request.Kind)
                {
                    case RequestKind.Describe:
                        var meshes = new List<Attachment>();
                        var manifest = ManifestBuilder.Build(reference, meshes);
                        server.Reply(Replies.Description(engineName, "unity", manifest), meshes);
                        break;
                    case RequestKind.Hello:
                        BeginSession(request);
                        Collect(false);
                        server.Reply(Replies.Welcome(engineName, Dt, "unity", Frame, SessionTime, outputs), outputs.Attachments);
                        break;
                    case RequestKind.Reset:
                        if (!InSession) { server.ReplyError("send hello first"); break; }
                        Frame = 0;
                        controls.Clear();
                        touch?.Reset();
                        ResetRequested?.Invoke(request.Seed, request.Options);
                        // Let one frame apply the reset before reporting it.
                        resetPending = true;
                        return;
                    case RequestKind.Step:
                        if (!InSession) { server.ReplyError("send hello first"); break; }
                        if (request.Frame != Frame) { server.ReplyError($"expected frame {Frame}, got {request.Frame}"); break; }
                        Apply(request);
                        stepPending = true;
                        return;
                    case RequestKind.Close:
                        server.Reply(Replies.Bye());
                        server.Drop();
                        EndSession();
                        return;
                }
            }
        }

        void AfterFrame()
        {
            if (resetPending)
            {
                resetPending = false;
                Collect(true);
                server.Reply(Replies.State("state", Frame, SessionTime, outputs), outputs.Attachments);
            }
            else if (stepPending)
            {
                stepPending = false;
                Frame += 1;
                Collect(true);
                server.Reply(Replies.State("state", Frame, SessionTime, outputs), outputs.Attachments);
            }
        }

        // Session -------------------------------------------------------------

        void BeginSession(Request hello)
        {
            EndSession();
            Dt = hello.Dt;
            Frame = 0;
            savedCapture = Time.captureDeltaTime;
            savedFixed = Time.fixedDeltaTime;
            savedMaximum = Time.maximumDeltaTime;
            Time.captureDeltaTime = (float)Dt;
            Time.fixedDeltaTime = (float)(Dt / physicsSubsteps);
            Time.maximumDeltaTime = Mathf.Max(savedMaximum, (float)Dt);
            if (driveInputUpdates)
            {
                savedMode = InputSystem.settings.updateMode;
                InputSystem.settings.updateMode = InputSettings.UpdateMode.ProcessEventsManually;
            }
            inputDevice = new EmbodiedInputDevice(hello.Controls);
            touch = new EmbodiedTouch();
            InSession = true;
        }

        void EndSession()
        {
            if (!InSession) return;
            InSession = false;
            stepPending = resetPending = false;
            Time.captureDeltaTime = savedCapture;
            Time.fixedDeltaTime = savedFixed;
            Time.maximumDeltaTime = savedMaximum;
            if (driveInputUpdates) InputSystem.settings.updateMode = savedMode;
            inputDevice?.Dispose();
            touch?.Dispose();
            inputDevice = null;
            touch = null;
            controls.Clear();
        }

        void Apply(Request step)
        {
            controls.Clear();
            foreach (var pair in step.Values) controls[pair.Key] = pair.Value;
            inputDevice.Write(controls);
            foreach (var e in step.Events)
            {
                touch.Handle(e);
                EventReceived?.Invoke(e);
            }
            if (driveInputUpdates) InputSystem.Update();
        }

        void Collect(bool capture)
        {
            outputs.Clear();
            foreach (var output in EmbodiedOutput.All) output.Collect(outputs, this, capture);
            foreach (var pair in customOutputs) outputs.Values[pair.Key] = pair.Value;
            customOutputs.Clear();
        }

        /// <summary>A pose relative to <see cref="reference"/> as [x, y, z, qw, qx, qy, qz].</summary>
        public double[] PoseOf(Transform target, Transform relativeTo = null)
        {
            var origin = relativeTo != null ? relativeTo : reference;
            var position = target.position;
            var rotation = target.rotation;
            if (origin != null)
            {
                var inverse = Quaternion.Inverse(origin.rotation);
                position = inverse * (position - origin.position);
                rotation = inverse * rotation;
            }
            return new double[] { position.x, position.y, position.z, rotation.w, rotation.x, rotation.y, rotation.z };
        }
    }

    /// <summary>Runs callbacks at the very start and very end of each frame.</summary>
    static class PlayerLoopHook
    {
        struct EmbodiedBeforeFrame { }
        struct EmbodiedAfterFrame { }

        static PlayerLoopSystem original;
        static bool installed;

        public static void Install(PlayerLoopSystem.UpdateFunction before, PlayerLoopSystem.UpdateFunction after)
        {
            if (installed) Uninstall();
            original = PlayerLoop.GetCurrentPlayerLoop();
            var loop = original;
            var systems = (PlayerLoopSystem[])loop.subSystemList.Clone();
            for (int i = 0; i < systems.Length; i++)
            {
                // Before the clock update in Initialization, so a new
                // captureDeltaTime applies to the frame it is set in.
                if (systems[i].type == typeof(UnityEngine.PlayerLoop.Initialization))
                    systems[i].subSystemList = Insert(systems[i].subSystemList, 0, typeof(EmbodiedBeforeFrame), before);
                // After rendering, so cameras show the frame just simulated.
                else if (systems[i].type == typeof(UnityEngine.PlayerLoop.PostLateUpdate))
                    systems[i].subSystemList = Insert(systems[i].subSystemList, systems[i].subSystemList.Length, typeof(EmbodiedAfterFrame), after);
            }
            loop.subSystemList = systems;
            PlayerLoop.SetPlayerLoop(loop);
            installed = true;
        }

        public static void Uninstall()
        {
            if (!installed) return;
            PlayerLoop.SetPlayerLoop(original);
            installed = false;
        }

        static PlayerLoopSystem[] Insert(PlayerLoopSystem[] list, int index, Type type, PlayerLoopSystem.UpdateFunction function)
        {
            var result = new List<PlayerLoopSystem>(list ?? Array.Empty<PlayerLoopSystem>());
            result.Insert(index, new PlayerLoopSystem { type = type, updateDelegate = function });
            return result.ToArray();
        }
    }
}
