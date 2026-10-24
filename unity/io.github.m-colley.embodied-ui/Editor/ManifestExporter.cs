// Tools > Embodied UI > Export Manifest: write the scene's devices for
// embodied_ui.Interface.from_manifest, with meshes beside the manifest.

using System.Collections.Generic;
using System.IO;
using EmbodiedUI.Protocol;
using Newtonsoft.Json;
using UnityEditor;
using UnityEngine;

namespace EmbodiedUI.Editor
{
    public static class ManifestExporter
    {
        [MenuItem("Tools/Embodied UI/Export Manifest...")]
        static void ExportMenu()
        {
            var path = EditorUtility.SaveFilePanel("Export embodied-ui manifest", "", "interface.json", "json");
            if (string.IsNullOrEmpty(path)) return;
            var reference = Selection.activeTransform;
            Export(path, reference);
            Debug.Log($"Exported embodied-ui manifest to {path}" +
                      (reference != null ? $" relative to {reference.name}" : " in world coordinates"));
        }

        /// <summary>Write the manifest and its meshes; poses relative to <paramref name="reference"/>.</summary>
        public static void Export(string path, Transform reference)
        {
            var meshes = new List<Attachment>();
            var manifest = ManifestBuilder.Build(reference, meshes);
            var folder = Path.GetDirectoryName(path);
            foreach (var mesh in meshes)
            {
                var file = Path.Combine(folder, mesh.Name);
                Directory.CreateDirectory(Path.GetDirectoryName(file));
                File.WriteAllBytes(file, mesh.Data);
            }
            File.WriteAllText(path, manifest.ToString(Formatting.Indented));
        }
    }
}
