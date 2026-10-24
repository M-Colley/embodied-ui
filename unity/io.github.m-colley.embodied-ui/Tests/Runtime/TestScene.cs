// A small scene for the bridge tests: a car pushed through an Input System
// action bound to the embodied-ui device, a lamp, a chime, a seat vibration,
// a camera, a designed button and a probe that reports frame timing,
// physics steps and touches.

using System.Collections.Generic;
using EmbodiedUI.Protocol;
using UnityEngine;
using UnityEngine.InputSystem;

namespace EmbodiedUI.Tests
{
    /// <summary>
    /// A rigid body accelerated by the vehicle.throttle control; full
    /// throttle lights the lamp, sounds the chime and vibrates the seat.
    /// </summary>
    public sealed class TestCar : MonoBehaviour
    {
        public InputAction throttle;
        public Rigidbody body;
        public Light lamp;
        public EmbodiedAudioOutput chime;
        public EmbodiedVibrationOutput seat;
        public int fixedSteps;
        public float lastThrottle;

        void Awake()
        {
            body = GetComponent<Rigidbody>();
            if (body == null) body = gameObject.AddComponent<Rigidbody>();
            body.useGravity = false;
            throttle = new InputAction("throttle", InputActionType.Value, "<EmbodiedUI>/vehicle_throttle");
            throttle.Enable();
        }

        void OnDestroy()
        {
            throttle.Disable();
            throttle.Dispose();
        }

        void FixedUpdate()
        {
            fixedSteps++;
            lastThrottle = throttle.ReadValue<float>();
            body.AddForce(transform.forward * 5f * lastThrottle, ForceMode.Acceleration);
            var full = lastThrottle > 0.5f;
            if (lamp != null) lamp.enabled = full;
            if (chime != null) chime.playing = full;
            if (seat != null) seat.amplitude = full ? 2f : 0f;
        }
    }

    /// <summary>Reports what the tests check: frame time, physics steps, input, touches.</summary>
    public sealed class FrameProbe : EmbodiedOutput
    {
        public TestCar car;

        public override void Collect(Outputs outputs, EmbodiedBridge bridge, bool capture)
        {
            outputs.Set("unity.delta_time", Time.deltaTime);
            outputs.Set("unity.fixed_steps", car != null ? car.fixedSteps : 0);
            outputs.Set("unity.throttle_input", car != null ? car.lastThrottle : 0f);
            // The raw control, to tell device problems from action problems.
            var embodied = InputSystem.GetDevice(EmbodiedInputDevice.Layout);
            var control = embodied?.TryGetChildControl<UnityEngine.InputSystem.Controls.AxisControl>("vehicle_throttle");
            outputs.Set("unity.device_throttle", control != null ? control.ReadValue() : -1f);
            outputs.Set("unity.device_enabled", embodied != null && embodied.enabled);
            var touches = 0;
            foreach (var device in InputSystem.devices)
                if (device is Touchscreen screen)
                    foreach (var touch in screen.touches)
                        if (touch.isInProgress) touches++;
            outputs.Set("unity.touches", touches);
        }
    }

    public sealed class Scene
    {
        public EmbodiedBridge Bridge;
        public TestCar Car;
        public readonly List<GameObject> Objects = new List<GameObject>();

        public void Destroy()
        {
            foreach (var o in Objects)
                if (o != null) Object.Destroy(o);
            Objects.Clear();
        }

        /// <summary>Build the scene with a bridge on <paramref name="port"/> (0: any free port).</summary>
        public static Scene Build(int port)
        {
            var scene = new Scene();
            var reference = new GameObject("reference");
            scene.Objects.Add(reference);

            var carObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            carObject.name = "car";
            scene.Car = carObject.AddComponent<TestCar>();
            carObject.AddComponent<EmbodiedVehicleOutput>();
            scene.Objects.Add(carObject);

            var lampObject = new GameObject("lamp");
            var light = lampObject.AddComponent<Light>();
            light.enabled = false;
            scene.Car.lamp = light;
            var lamp = lampObject.AddComponent<EmbodiedLampOutput>();
            lamp.outputName = "lamp.on";
            lamp.follow = light;
            scene.Objects.Add(lampObject);

            // Set from the car rather than an AudioSource: batch-mode
            // editors may run without an audio device.
            var chime = new GameObject("chime").AddComponent<EmbodiedAudioOutput>();
            chime.device = "chime";
            chime.levelAtFullVolume = 64f;
            scene.Car.chime = chime;
            scene.Objects.Add(chime.gameObject);
            var seat = new GameObject("seat").AddComponent<EmbodiedVibrationOutput>();
            seat.device = "seat";
            scene.Car.seat = seat;
            scene.Objects.Add(seat.gameObject);

            var cameraObject = new GameObject("camera");
            cameraObject.transform.position = new Vector3(0f, 2f, -5f);
            cameraObject.AddComponent<Camera>();
            var cameraOutput = cameraObject.AddComponent<EmbodiedCameraOutput>();
            cameraOutput.outputName = "camera.main";
            cameraOutput.width = 64;
            cameraOutput.height = 48;
            scene.Objects.Add(cameraObject);

            var probe = new GameObject("probe").AddComponent<FrameProbe>();
            probe.car = scene.Car;
            scene.Objects.Add(probe.gameObject);

            // A designed control: a 15 mm cap, 6 mm thick, 0.4 m ahead, pressed along +z.
            var button = GameObject.CreatePrimitive(PrimitiveType.Cube);
            button.name = "start button";
            button.transform.localScale = new Vector3(0.015f, 0.015f, 0.006f);
            button.transform.position = new Vector3(0f, 1f, 0.4f);
            Object.Destroy(button.GetComponent<Collider>());
            var device = button.AddComponent<EmbodiedDevice>();
            device.deviceName = "start";
            device.type = EmbodiedDeviceType.PushButton;
            device.mesh = button.GetComponent<MeshFilter>();
            device.meshField = "cap";
            device.fit = "box";
            device.parameters = "{\"travel\": 0.004}";
            scene.Objects.Add(button);

            var bridgeObject = new GameObject("bridge");
            bridgeObject.SetActive(false);
            scene.Bridge = bridgeObject.AddComponent<EmbodiedBridge>();
            scene.Bridge.port = port;
            scene.Bridge.requestTimeout = 60f;
            scene.Bridge.reference = reference.transform;
            scene.Bridge.engineName = "Unity test scene";
            bridgeObject.SetActive(true);
            scene.Objects.Add(bridgeObject);
            return scene;
        }
    }
}
