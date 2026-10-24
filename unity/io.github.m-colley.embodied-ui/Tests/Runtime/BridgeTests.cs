// PlayMode tests of the lockstep bridge inside the Unity player loop.
//
// ACSharpClientStepsTheSceneInLockstep drives the bridge from a background
// thread speaking the protocol. PythonClientDrivesTheScene waits for the
// Python test tests/test_unity_live.py, which sets EMBODIED_UI_LIVE_PORT.

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Threading;
using EmbodiedUI.Protocol;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace EmbodiedUI.Tests
{
    public sealed class BridgeTests
    {
        Scene scene;

        [TearDown]
        public void Clean()
        {
            scene?.Destroy();
            scene = null;
        }

        static JObject Step(int frame, double throttle, double pressed, JArray events)
        {
            return new JObject
            {
                ["type"] = "step",
                ["frame"] = frame,
                ["time"] = frame * 0.02,
                ["controls"] = new JObject { ["vehicle.throttle"] = throttle, ["b.pressed"] = pressed },
                ["events"] = events,
            };
        }

        static JObject Touch(string kind, double time)
        {
            return new JObject
            {
                ["time"] = time,
                ["source"] = "screen",
                ["kind"] = kind,
                ["value"] = 1,
                ["data"] = new JObject { ["u"] = 0.5, ["v"] = 0.5, ["force"] = 0.3 },
            };
        }

        [UnityTest]
        [Timeout(120000)]
        public IEnumerator ACSharpClientStepsTheSceneInLockstep()
        {
            scene = Scene.Build(0);
            yield return null;
            var port = scene.Bridge.ListeningPort;
            Assert.Greater(port, 0);
            var replies = new ConcurrentQueue<Message>();
            Exception failure = null;
            var thread = new Thread(() =>
            {
                try
                {
                    using (var client = new TcpClient("127.0.0.1", port) { NoDelay = true })
                    {
                        var stream = client.GetStream();
                        Wire.Send(stream, new JObject
                        {
                            ["type"] = "hello",
                            ["protocol"] = Wire.Protocol,
                            ["dt"] = 0.02,
                            ["controls"] = new JArray("vehicle.throttle", "b.pressed"),
                            ["outputs"] = new JArray(),
                        });
                        replies.Enqueue(Wire.Receive(stream));
                        for (int k = 0; k < 20; k++)
                        {
                            var events = new JArray();
                            if (k == 5) events.Add(new JObject { ["time"] = 0.1, ["source"] = "b", ["kind"] = "press", ["value"] = 1 });
                            if (k == 8) events.Add(Touch("touch_begin", 0.16));
                            if (k == 12) events.Add(Touch("touch_end", 0.24));
                            Wire.Send(stream, Step(k, 1.0, k >= 5 ? 1.0 : 0.0, events));
                            replies.Enqueue(Wire.Receive(stream));
                        }
                        Wire.Send(stream, new JObject { ["type"] = "reset", ["seed"] = 1 });
                        replies.Enqueue(Wire.Receive(stream));
                        Wire.Send(stream, new JObject { ["type"] = "close" });
                        replies.Enqueue(Wire.Receive(stream));
                    }
                }
                catch (Exception error)
                {
                    failure = error;
                }
            });
            var received = new List<string>();
            Action<DeviceEvent> handler = e => received.Add(e.Source + ":" + e.Kind);
            EmbodiedBridge.EventReceived += handler;
            thread.Start();
            var deadline = Time.realtimeSinceStartup + 60f;
            while (thread.IsAlive && Time.realtimeSinceStartup < deadline) yield return null;
            EmbodiedBridge.EventReceived -= handler;
            Assert.IsNull(failure, failure?.ToString());
            Assert.IsFalse(thread.IsAlive, "the client did not finish");

            var list = replies.ToArray();
            Assert.AreEqual(23, list.Length);
            Assert.AreEqual("welcome", list[0].Type);
            Assert.AreEqual(0.02, (double)list[0].Header["dt"], 1e-12);
            Assert.AreEqual("unity", (string)list[0].Header["convention"]);
            var steps = new List<double>();
            for (int k = 1; k <= 20; k++)
            {
                var state = list[k];
                Assert.AreEqual("state", state.Type);
                Assert.AreEqual(k, (long)state.Header["frame"]);
                Assert.AreEqual(k * 0.02, (double)state.Header["time"], 1e-9);
                var outputs = (JObject)state.Header["outputs"];
                // Every lockstep frame lasts exactly the session period.
                Assert.AreEqual(0.02, (double)outputs["unity.delta_time"], 1e-6, $"frame {k}");
                steps.Add((double)outputs["unity.fixed_steps"]);
                var camera = state.Attachments.Single(a => a.Name == "camera.main");
                CollectionAssert.AreEqual(new[] { 48, 64, 3 }, camera.Shape);
            }
            // One physics step per frame (physicsSubsteps = 1) after the first.
            for (int k = 2; k < steps.Count; k++)
                Assert.AreEqual(1.0, steps[k] - steps[k - 1], $"physics steps in frame {k + 1}");
            var last = (JObject)list[20].Header["outputs"];
            Assert.AreEqual(1.0, (double)last["unity.device_enabled"], "the virtual device is enabled");
            Assert.AreEqual(1.0, (double)last["unity.device_throttle"], 1e-6, "the device holds the control");
            Assert.AreEqual(1.0, (double)last["unity.throttle_input"], 1e-6, "the action saw the control");
            Assert.Greater((double)last["vehicle.speed"], 0.5);
            Assert.AreEqual(1.0, (double)last["lamp.on"]);
            Assert.AreEqual(1.0, (double)((JObject)list[9].Header["outputs"])["unity.touches"], "touch began");
            Assert.AreEqual(0.0, (double)((JObject)list[16].Header["outputs"])["unity.touches"], "touch ended");
            CollectionAssert.Contains(received, "b:press");
            CollectionAssert.Contains(received, "screen:touch_begin");
            Assert.AreEqual("state", list[21].Type);
            Assert.AreEqual(0, (long)list[21].Header["frame"]);
            Assert.AreEqual("bye", list[22].Type);
            Assert.IsFalse(scene.Bridge.InSession);
        }

        [UnityTest]
        [Timeout(600000)]
        public IEnumerator PythonClientDrivesTheScene()
        {
            var port = Environment.GetEnvironmentVariable("EMBODIED_UI_LIVE_PORT");
            if (string.IsNullOrEmpty(port))
                Assert.Ignore("run tests/test_unity_live.py, which sets EMBODIED_UI_LIVE_PORT");
            scene = Scene.Build(int.Parse(port));
            var deadline = Time.realtimeSinceStartup + 300f;
            var started = false;
            long frames = 0;
            while (Time.realtimeSinceStartup < deadline)
            {
                if (scene.Bridge.InSession)
                {
                    started = true;
                    frames = Math.Max(frames, scene.Bridge.Frame);
                }
                else if (started)
                {
                    break;
                }
                yield return null;
            }
            Assert.IsTrue(started, "no Python client connected");
            Assert.Greater(frames, 0);
            Debug.Log($"[EmbodiedUI live] the Python session ended after {frames} frames");
        }
    }
}
