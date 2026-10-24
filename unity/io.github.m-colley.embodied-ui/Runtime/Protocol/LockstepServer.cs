// A single-client TCP server for the lockstep protocol, without Unity types,
// so the same code runs in Unity and in .NET tests.

using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using Newtonsoft.Json.Linq;

namespace EmbodiedUI.Protocol
{
    public sealed class LockstepServer : IDisposable
    {
        readonly TcpListener listener;
        TcpClient client;
        NetworkStream stream;

        /// <param name="host">Listening address; 127.0.0.1 keeps the session local.</param>
        /// <param name="port">Port; 0 picks a free one.</param>
        public LockstepServer(string host = "127.0.0.1", int port = Wire.DefaultPort)
        {
            listener = new TcpListener(IPAddress.Parse(host), port);
            listener.Start();
            Port = ((IPEndPoint)listener.LocalEndpoint).Port;
        }

        public int Port { get; }

        public bool HasClient => client != null && client.Connected;

        /// <summary>Wait up to <paramref name="timeoutMs"/> (negative: forever) for a client.</summary>
        public bool Accept(int timeoutMs)
        {
            if (HasClient) return true;
            if (timeoutMs >= 0)
            {
                var deadline = DateTime.UtcNow.AddMilliseconds(timeoutMs);
                while (!listener.Pending())
                {
                    if (DateTime.UtcNow >= deadline) return false;
                    System.Threading.Thread.Sleep(1);
                }
            }
            client = listener.AcceptTcpClient();
            client.NoDelay = true;
            stream = client.GetStream();
            return true;
        }

        /// <summary>
        /// Receive and parse the next request. Malformed requests are answered
        /// with an error reply and skipped. Returns null if the client left or
        /// nothing arrived within <paramref name="timeoutMs"/> (negative: forever).
        /// </summary>
        public Request Next(int timeoutMs)
        {
            while (HasClient)
            {
                Message message;
                try
                {
                    stream.ReadTimeout = timeoutMs < 0 ? System.Threading.Timeout.Infinite : Math.Max(1, timeoutMs);
                    message = Wire.Receive(stream);
                }
                catch (IOException)
                {
                    Drop();
                    return null;
                }
                catch (ProtocolException error)
                {
                    ReplyError(error.Message);
                    Drop();
                    return null;
                }
                try
                {
                    return Request.Parse(message);
                }
                catch (Exception error) when (error is ProtocolException || error is FormatException || error is InvalidCastException)
                {
                    ReplyError(error.Message);
                }
            }
            return null;
        }

        public void Reply(JObject header, IList<Attachment> attachments = null)
        {
            if (!HasClient) return;
            try
            {
                Wire.Send(stream, header, attachments);
            }
            catch (IOException)
            {
                Drop();
            }
        }

        public void ReplyError(string message)
        {
            Reply(Replies.Error(message));
        }

        public void Drop()
        {
            stream?.Dispose();
            client?.Close();
            stream = null;
            client = null;
        }

        public void Dispose()
        {
            Drop();
            listener.Stop();
        }
    }
}
