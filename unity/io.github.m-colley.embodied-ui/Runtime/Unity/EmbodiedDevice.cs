// Design devices in Unity; the MuJoCo side builds them from a manifest.
//
// Orient a device's GameObject so local +Z points into the panel (the press
// direction, as for a world-space canvas) and +Y up. Python's
// Interface.from_manifest converts poses and meshes with that convention.

using System.Collections.Generic;
using System.Globalization;
using System.Text;
using EmbodiedUI.Protocol;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace EmbodiedUI
{
    public enum EmbodiedDeviceType
    {
        PushButton, ToggleSwitch, Knob, Slider, Lever, Pedal, SteeringWheel,
        Joystick, TouchSurface, Screen, Indicator, Prop,
    }

    [AddComponentMenu("Embodied UI/Embodied Device")]
    public sealed class EmbodiedDevice : MonoBehaviour
    {
        [Tooltip("Unique name; channels are name.channel.")]
        public string deviceName = "button";
        public EmbodiedDeviceType type = EmbodiedDeviceType.PushButton;

        [Tooltip("Body of the MuJoCo model carrying the device, or 'world'.")]
        public string mount = "world";

        [Tooltip("Further fields of the Python device as JSON, e.g. {\"travel\": 0.004, \"toggle\": true}.")]
        [TextArea(2, 6)] public string parameters = "{}";

        [Tooltip("Export this mesh as the device's moving (or static) part.")]
        public MeshFilter mesh;

        [Tooltip("Device field the mesh replaces: cap (PushButton), lever (ToggleSwitch), shape (Indicator)...")]
        public string meshField = "cap";

        [Tooltip("Collision primitive fitted to the mesh: box, cylinder, capsule, sphere or mesh.")]
        public string fit = "box";

        [Tooltip("Moving mass of the mesh part, kg.")]
        public float mass = 0.005f;

        public JObject ToManifest(Transform reference, List<Attachment> meshes)
        {
            JObject device;
            try
            {
                device = JObject.Parse(string.IsNullOrWhiteSpace(parameters) ? "{}" : parameters);
            }
            catch (System.Exception error)
            {
                throw new System.ArgumentException($"{name}: parameters are not a JSON object: {error.Message}");
            }
            device["type"] = type.ToString();
            device["name"] = deviceName;
            device["mount"] = mount;
            var pose = Pose(reference);
            device["pose"] = new JObject
            {
                ["type"] = "Pose",
                ["pos"] = new JArray(pose[0], pose[1], pose[2]),
                ["quat"] = new JArray(pose[3], pose[4], pose[5], pose[6]),
            };
            if (mesh != null && mesh.sharedMesh != null && !string.IsNullOrEmpty(meshField))
            {
                var file = "meshes/" + EmbodiedInputDevice.ControlName(deviceName) + ".obj";
                var bytes = Encoding.UTF8.GetBytes(ToObj(mesh.sharedMesh, mesh.transform, transform));
                meshes.Add(new Attachment(file, "uint8", new[] { bytes.Length }, bytes));
                var part = new JObject
                {
                    ["type"] = "Part",
                    ["shape"] = new JObject
                    {
                        ["type"] = "Mesh",
                        ["file"] = file,
                        ["frame"] = "unity",
                        ["fit"] = fit,
                    },
                    ["mass"] = type == EmbodiedDeviceType.Indicator || type == EmbodiedDeviceType.Screen ? 0.0 : mass,
                };
                device[meshField] = meshField == "parts" ? (JToken)new JArray(part) : part;
            }
            return device;
        }

        double[] Pose(Transform reference)
        {
            var position = transform.position;
            var rotation = transform.rotation;
            if (reference != null)
            {
                var inverse = Quaternion.Inverse(reference.rotation);
                position = inverse * (position - reference.position);
                rotation = inverse * rotation;
            }
            return new double[] { position.x, position.y, position.z, rotation.w, rotation.x, rotation.y, rotation.z };
        }

        /// <summary>
        /// Wavefront OBJ of a mesh in this device's local axes (metres, scale
        /// applied, rotation and translation relative to the device). Needs a
        /// readable mesh at runtime; the Editor exporter reads any mesh.
        /// </summary>
        static string ToObj(Mesh source, Transform meshTransform, Transform device)
        {
            var text = new StringBuilder();
            var culture = CultureInfo.InvariantCulture;
            foreach (var v in source.vertices)
            {
                var local = device.InverseTransformPoint(meshTransform.TransformPoint(v));
                local.Scale(device.lossyScale);
                text.Append("v ").Append(local.x.ToString("R", culture)).Append(' ')
                    .Append(local.y.ToString("R", culture)).Append(' ')
                    .Append(local.z.ToString("R", culture)).Append('\n');
            }
            var triangles = source.triangles;
            for (int i = 0; i < triangles.Length; i += 3)
                text.Append("f ").Append(triangles[i] + 1).Append(' ')
                    .Append(triangles[i + 1] + 1).Append(' ')
                    .Append(triangles[i + 2] + 1).Append('\n');
            return text.ToString();
        }
    }

    public static class ManifestBuilder
    {
        /// <summary>The embodied-ui manifest of every EmbodiedDevice in the loaded scenes.</summary>
        public static JObject Build(Transform reference, List<Attachment> meshes, string name = null)
        {
#pragma warning disable CS0618 // the sort-mode overload is deprecated in Unity 6.6 but the newer overloads are missing in 2022.3
#if UNITY_2023_1_OR_NEWER
            var devices = Object.FindObjectsByType<EmbodiedDevice>(FindObjectsSortMode.None);
#else
            var devices = Object.FindObjectsOfType<EmbodiedDevice>();
#endif
#pragma warning restore CS0618
            // A stable order by name, independent of object identities.
            System.Array.Sort(devices, (a, b) => string.CompareOrdinal(a.deviceName, b.deviceName));
            var list = new JArray();
            var names = new HashSet<string>();
            foreach (var device in devices)
            {
                if (!names.Add(device.deviceName))
                    throw new System.ArgumentException("duplicate device name " + device.deviceName);
                list.Add(device.ToManifest(reference, meshes));
            }
            return new JObject
            {
                ["format"] = "embodied-ui/manifest",
                ["version"] = 1,
                ["frame"] = "unity",
                ["name"] = name ?? EmbodiedInputDevice.ControlName(UnityEngine.SceneManagement.SceneManager.GetActiveScene().name),
                ["devices"] = list,
            };
        }
    }
}
