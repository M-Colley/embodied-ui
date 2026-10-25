using System;
using System.Globalization;
using UnityEngine;

namespace EmbodiedUI.PlayerTest
{
    /// <summary>
    /// Starts the scene's bridge on the port given with -embodiedPort and
    /// quits the player when the client's session ends.
    /// </summary>
    public sealed class PlayerBootstrap : MonoBehaviour
    {
        public EmbodiedBridge bridge;
        bool started;

        void Awake()
        {
            // A windowed player must keep running without focus.
            Application.runInBackground = true;
            var args = Environment.GetCommandLineArgs();
            for (int i = 0; i + 1 < args.Length; i++)
                if (args[i] == "-embodiedPort")
                    bridge.port = int.Parse(args[i + 1], CultureInfo.InvariantCulture);
            bridge.gameObject.SetActive(true);
        }

        void Update()
        {
            if (bridge.InSession) started = true;
            else if (started) Application.Quit();
        }
    }
}
