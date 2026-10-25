// PlayMode test of the package's Wheel Vehicle sample, driven from Python.
//
// tests/test_unity_sample_live.py copies Samples~/WheelVehicle into
// Assets/Samples as Unity's sample import does, adds these tests as the
// sample's Tests folder (their own assembly) and sets
// EMBODIED_UI_SAMPLE_PORT. A RemoteEngine then sends vehicle.throttle,
// vehicle.brake and vehicle.steer and checks that the car accelerates,
// turns and brakes.

using System;
using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Object = UnityEngine.Object;

namespace EmbodiedUI.Samples.Tests
{
    public sealed class WheelVehicleTests
    {
        readonly List<GameObject> objects = new List<GameObject>();
        EmbodiedBridge bridge;

        [TearDown]
        public void Clean()
        {
            foreach (var o in objects)
                if (o != null) Object.Destroy(o);
            objects.Clear();
        }

        static WheelCollider Wheel(Transform car, string name, Vector3 position)
        {
            var wheelObject = new GameObject(name);
            wheelObject.transform.SetParent(car, false);
            wheelObject.transform.localPosition = position;
            var wheel = wheelObject.AddComponent<WheelCollider>();
            wheel.radius = 0.35f;
            wheel.mass = 20f;
            wheel.suspensionDistance = 0.3f;
            wheel.suspensionSpring = new JointSpring { spring = 35000f, damper = 4500f, targetPosition = 0.5f };
            return wheel;
        }

        /// <summary>A 1500 kg car with the sample's controller on flat ground.</summary>
        void Build(int port)
        {
            var ground = GameObject.CreatePrimitive(PrimitiveType.Cube);
            ground.name = "ground";
            ground.transform.localScale = new Vector3(1000f, 1f, 1000f);
            ground.transform.position = new Vector3(0f, -0.5f, 0f);
            objects.Add(ground);

            var car = new GameObject("wheel vehicle");
            car.transform.position = new Vector3(0f, 0.6f, 0f);
            var body = car.AddComponent<Rigidbody>();
            body.mass = 1500f;
            var chassis = car.AddComponent<BoxCollider>();
            chassis.center = new Vector3(0f, 0.25f, 0f);
            chassis.size = new Vector3(1.8f, 0.5f, 4.2f);
            body.centerOfMass = new Vector3(0f, -0.2f, 0f);
            var vehicle = car.AddComponent<EmbodiedWheelVehicle>();
            vehicle.frontLeft = Wheel(car.transform, "front left", new Vector3(-0.8f, 0f, 1.4f));
            vehicle.frontRight = Wheel(car.transform, "front right", new Vector3(0.8f, 0f, 1.4f));
            vehicle.rearLeft = Wheel(car.transform, "rear left", new Vector3(-0.8f, 0f, -1.4f));
            vehicle.rearRight = Wheel(car.transform, "rear right", new Vector3(0.8f, 0f, -1.4f));
            car.AddComponent<EmbodiedVehicleOutput>();
            objects.Add(car);

            var probe = new GameObject("probe").AddComponent<WheelVehicleProbe>();
            probe.vehicle = vehicle;
            objects.Add(probe.gameObject);

            var bridgeObject = new GameObject("bridge");
            bridgeObject.SetActive(false);
            bridge = bridgeObject.AddComponent<EmbodiedBridge>();
            bridge.port = port;
            bridge.requestTimeout = 60f;
            bridge.engineName = "Wheel vehicle sample";
            bridgeObject.SetActive(true);
            objects.Add(bridgeObject);
        }

        [UnityTest]
        [Timeout(900000)]
        public IEnumerator PythonClientDrivesTheWheelVehicle()
        {
            var port = Environment.GetEnvironmentVariable("EMBODIED_UI_SAMPLE_PORT");
            if (string.IsNullOrEmpty(port))
                Assert.Ignore("run tests/test_unity_sample_live.py, which sets EMBODIED_UI_SAMPLE_PORT");
            Build(int.Parse(port));
            var deadline = Time.realtimeSinceStartup + 600f;
            var started = false;
            long frames = 0;
            while (Time.realtimeSinceStartup < deadline)
            {
                if (bridge.InSession)
                {
                    started = true;
                    frames = Math.Max(frames, bridge.Frame);
                }
                else if (started)
                {
                    break;
                }
                yield return null;
            }
            Assert.IsTrue(started, "no Python client connected");
            Assert.Greater(frames, 0);
            Debug.Log($"[EmbodiedUI sample] the Python session ended after {frames} frames");
        }
    }
}
