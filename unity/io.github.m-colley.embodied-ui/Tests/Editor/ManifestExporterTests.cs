// EditMode tests of Tools > Embodied UI > Export Manifest.
//
// A new scene gets EmbodiedDevice components; the exporter's routine writes
// them to a temporary folder and the JSON and meshes are checked. With
// EMBODIED_UI_MANIFEST_OUT set (tests/test_unity_editor_live.py), the export
// is also kept there for Python to load with Interface.from_manifest.

using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Object = UnityEngine.Object;

namespace EmbodiedUI.Editor.Tests
{
    public sealed class ManifestExporterTests
    {
        const string MenuPath = "Tools/Embodied UI/Export Manifest...";
        string folder;
        Transform reference;

        [SetUp]
        public void Open()
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            folder = Path.Combine(Path.GetTempPath(), "embodied-ui-export-" + Guid.NewGuid().ToString("N"));
            // A rotated dashboard origin: exported poses are relative to it.
            reference = new GameObject("dashboard origin").transform;
            reference.SetPositionAndRotation(new Vector3(1f, 0.5f, 2f), Quaternion.Euler(0f, 90f, 0f));
        }

        [TearDown]
        public void Close()
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            if (Directory.Exists(folder)) Directory.Delete(folder, true);
        }

        EmbodiedDevice Device(string name, EmbodiedDeviceType type, Vector3 position, Quaternion rotation,
            PrimitiveType? primitive = null, Vector3? scale = null)
        {
            var gameObject = primitive.HasValue ? GameObject.CreatePrimitive(primitive.Value) : new GameObject();
            gameObject.name = name;
            gameObject.transform.SetPositionAndRotation(
                reference.TransformPoint(position), reference.rotation * rotation);
            if (scale.HasValue) gameObject.transform.localScale = scale.Value;
            var device = gameObject.AddComponent<EmbodiedDevice>();
            device.deviceName = name;
            device.type = type;
            if (primitive.HasValue) device.mesh = gameObject.GetComponent<MeshFilter>();
            return device;
        }

        void BuildDashboard()
        {
            // A 15 x 15 mm cap, 6 mm thick, pressed along the reference's +z.
            var start = Device("start", EmbodiedDeviceType.PushButton, new Vector3(0f, 0.5f, 0.4f),
                Quaternion.identity, PrimitiveType.Cube, new Vector3(0.015f, 0.015f, 0.006f));
            start.meshField = "cap";
            start.fit = "box";
            start.parameters = "{\"travel\": 0.004}";
            var volume = Device("volume", EmbodiedDeviceType.Knob, new Vector3(0.06f, 0.5f, 0.4f), Quaternion.identity);
            volume.parameters = "{\"push\": true, \"detents\": 12}";
            var screen = Device("screen", EmbodiedDeviceType.TouchSurface, new Vector3(-0.15f, 0.45f, 0.45f),
                Quaternion.Euler(20f, 0f, 0f));
            screen.parameters = "{\"size\": [0.2, 0.12], \"regions\": {\"ok\": [0.0, 0.0, 0.5, 0.5]}}";
            var lamp = Device("hazard_lamp", EmbodiedDeviceType.Indicator, new Vector3(0.1f, 0.55f, 0.42f),
                Quaternion.identity, PrimitiveType.Sphere, Vector3.one * 0.02f);
            lamp.meshField = "shape";
            lamp.fit = "sphere";
        }

        static double[] Numbers(JToken token)
        {
            return token.Select(v => (double)v).ToArray();
        }

        static float[][] Vertices(string obj)
        {
            return File.ReadAllLines(obj)
                .Where(line => line.StartsWith("v "))
                .Select(line => line.Substring(2).Split(' ')
                    .Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray())
                .ToArray();
        }

        [Test]
        public void TheMenuItemCallsTheExportRoutine()
        {
            var menu = typeof(ManifestExporter)
                .GetMethods(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
                .SelectMany(m => m.GetCustomAttributes<MenuItem>())
                .Select(a => a.menuItem);
            CollectionAssert.Contains(menu.ToArray(), MenuPath);
            var export = typeof(ManifestExporter).GetMethod("Export", new[] { typeof(string), typeof(Transform) });
            Assert.IsNotNull(export, "Export(string path, Transform reference) is public");
        }

        [Test]
        public void ExportWritesTheScenesDevicesAndMeshes()
        {
            BuildDashboard();
            var path = Path.Combine(folder, "interface.json");
            ManifestExporter.Export(path, reference);

            var manifest = JObject.Parse(File.ReadAllText(path));
            Assert.AreEqual("embodied-ui/manifest", (string)manifest["format"]);
            Assert.AreEqual(1, (int)manifest["version"]);
            Assert.AreEqual("unity", (string)manifest["frame"]);
            // Python's Interface needs a nonempty name without '/', '.' or spaces,
            // also when the scene was never saved.
            StringAssert.IsMatch(@"^[^/.\s]+$", (string)manifest["name"]);
            var devices = (JArray)manifest["devices"];
            CollectionAssert.AreEqual(new[] { "hazard_lamp", "screen", "start", "volume" },
                devices.Select(d => (string)d["name"]).ToArray(), "sorted by name");

            var start = (JObject)devices[2];
            Assert.AreEqual("PushButton", (string)start["type"]);
            Assert.AreEqual("world", (string)start["mount"]);
            Assert.AreEqual(0.004, (double)start["travel"], 1e-12);
            Assert.AreEqual("Pose", (string)start["pose"]["type"]);
            var position = Numbers(start["pose"]["pos"]);
            var quaternion = Numbers(start["pose"]["quat"]);
            Assert.That(position, Is.EqualTo(new[] { 0.0, 0.5, 0.4 }).Within(1e-5), "relative to the reference");
            Assert.That(quaternion, Is.EqualTo(new[] { 1.0, 0.0, 0.0, 0.0 }).Within(1e-5));
            var cap = (JObject)start["cap"];
            Assert.AreEqual("Part", (string)cap["type"]);
            Assert.AreEqual(0.005, (double)cap["mass"], 1e-6);
            Assert.AreEqual("Mesh", (string)cap["shape"]["type"]);
            Assert.AreEqual("meshes/start.obj", (string)cap["shape"]["file"]);
            Assert.AreEqual("unity", (string)cap["shape"]["frame"]);
            Assert.AreEqual("box", (string)cap["shape"]["fit"]);

            var screen = (JObject)devices[1];
            var tilt = Quaternion.Euler(20f, 0f, 0f);
            Assert.That(Numbers(screen["pose"]["quat"]),
                Is.EqualTo(new double[] { tilt.w, tilt.x, tilt.y, tilt.z }).Within(1e-5));
            Assert.That(Numbers(screen["size"]), Is.EqualTo(new[] { 0.2, 0.12 }).Within(1e-12));
            Assert.That(Numbers(screen["regions"]["ok"]), Is.EqualTo(new[] { 0.0, 0.0, 0.5, 0.5 }));

            var volume = (JObject)devices[3];
            Assert.AreEqual("Knob", (string)volume["type"]);
            Assert.IsTrue((bool)volume["push"]);
            Assert.AreEqual(12, (int)volume["detents"]);
            Assert.IsFalse(volume.ContainsKey("cap"), "no mesh, no part");

            var lamp = (JObject)devices[0];
            Assert.AreEqual(0.0, (double)lamp["shape"]["mass"], "indicators are static");
            Assert.AreEqual("sphere", (string)lamp["shape"]["shape"]["fit"]);

            // Meshes beside the manifest, in device axes, metres, scale applied.
            var cube = Vertices(Path.Combine(folder, "meshes", "start.obj"));
            Assert.AreEqual(24, cube.Length);
            var extent = Enumerable.Range(0, 3).Select(axis => cube.Max(v => Math.Abs(v[axis]))).ToArray();
            Assert.That(extent, Is.EqualTo(new[] { 0.0075f, 0.0075f, 0.003f }).Within(1e-6f));
            Assert.AreEqual(12, File.ReadAllLines(Path.Combine(folder, "meshes", "start.obj")).Count(l => l.StartsWith("f ")));
            var sphere = Vertices(Path.Combine(folder, "meshes", "hazard_lamp.obj"));
            Assert.AreEqual(0.01, sphere.Max(v => Math.Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])), 1e-5);

            var keep = Environment.GetEnvironmentVariable("EMBODIED_UI_MANIFEST_OUT");
            if (!string.IsNullOrEmpty(keep))
            {
                ManifestExporter.Export(keep, reference);
                Debug.Log($"[EmbodiedUI export] kept the manifest at {keep}");
            }
        }

        [Test]
        public void DuplicateDeviceNamesAreRejected()
        {
            BuildDashboard();
            Device("start", EmbodiedDeviceType.PushButton, Vector3.zero, Quaternion.identity);
            var path = Path.Combine(folder, "duplicate.json");
            var error = Assert.Throws<ArgumentException>(() => ManifestExporter.Export(path, reference));
            StringAssert.Contains("start", error.Message);
            Assert.IsFalse(File.Exists(path));
        }

        [Test]
        public void PosesAreInWorldCoordinatesWithoutAReference()
        {
            var device = Device("start", EmbodiedDeviceType.PushButton, new Vector3(0f, 0.5f, 0.4f), Quaternion.identity);
            var path = Path.Combine(folder, "world.json");
            ManifestExporter.Export(path, null);
            var manifest = JObject.Parse(File.ReadAllText(path));
            var world = device.transform.position;
            Assert.That(Numbers(manifest["devices"][0]["pose"]["pos"]),
                Is.EqualTo(new double[] { world.x, world.y, world.z }).Within(1e-5));
            Object.DestroyImmediate(device.gameObject);
        }
    }
}
