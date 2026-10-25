// Builds a Windows player of a minimal bridge scene, for
// tests/test_unity_player_live.py:
//
//   Unity.exe -batchmode -quit -projectPath unity/EmbodiedUITests -buildTarget Win64
//     -executeMethod EmbodiedUI.PlayerTest.PlayerBuild.Build -embodiedBuildPath <folder>
//
// The scene is generated into Assets/Generated, which git ignores. The player
// waits for a client on -embodiedPort (default 47474) and quits after the
// session.

using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace EmbodiedUI.PlayerTest
{
    public static class PlayerBuild
    {
        public const string ScenePath = "Assets/Generated/BridgePlayer.unity";
        public const string Executable = "EmbodiedUIPlayer.exe";

        public static void Build()
        {
            var folder = Argument("-embodiedBuildPath") ?? Path.GetFullPath("Builds/BridgePlayer");
            CreateScene();
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = Path.Combine(folder, Executable),
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None,
            });
            var summary = report.summary;
            Debug.Log($"[EmbodiedUI build] {summary.result}: {summary.outputPath}, " +
                      $"{summary.totalSize} bytes, {summary.totalErrors} errors, {summary.totalWarnings} warnings, " +
                      $"{summary.totalTime.TotalSeconds:F1} s");
            if (summary.result != BuildResult.Succeeded) EditorApplication.Exit(1);
        }

        static string Argument(string name)
        {
            var args = Environment.GetCommandLineArgs();
            for (int i = 0; i + 1 < args.Length; i++)
                if (args[i] == name) return args[i + 1];
            return null;
        }

        /// <summary>A lit car cube, a lamp, a camera, a probe and a bridge waiting for a client.</summary>
        static void CreateScene()
        {
            Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var sun = new GameObject("sun").AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            var cameraObject = new GameObject("camera");
            cameraObject.transform.SetPositionAndRotation(new Vector3(0f, 2f, -5f), Quaternion.Euler(15f, 0f, 0f));
            var camera = cameraObject.AddComponent<Camera>();
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.2f, 0.3f, 0.5f);
            var cameraOutput = cameraObject.AddComponent<EmbodiedCameraOutput>();
            cameraOutput.outputName = "camera.main";
            cameraOutput.width = 64;
            cameraOutput.height = 48;

            var carObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            carObject.name = "car";
            var car = carObject.AddComponent<PlayerCar>();
            carObject.AddComponent<EmbodiedVehicleOutput>();

            var lampObject = new GameObject("lamp");
            var light = lampObject.AddComponent<Light>();
            light.enabled = false;
            car.lamp = light;
            var lamp = lampObject.AddComponent<EmbodiedLampOutput>();
            lamp.outputName = "lamp.on";
            lamp.follow = light;

            var probe = new GameObject("probe").AddComponent<PlayerProbe>();
            probe.car = car;

            // Inactive until the bootstrap has set the port.
            var bridgeObject = new GameObject("bridge");
            bridgeObject.SetActive(false);
            var bridge = bridgeObject.AddComponent<EmbodiedBridge>();
            bridge.waitForClient = true;
            bridge.requestTimeout = 60f;
            bridge.engineName = "Unity player";

            var bootstrap = new GameObject("bootstrap").AddComponent<PlayerBootstrap>();
            bootstrap.bridge = bridge;

            if (!EditorSceneManager.SaveScene(scene, ScenePath))
                throw new IOException("could not save " + ScenePath);
        }
    }
}
