// Wire format of the embodied-ui/1 lockstep protocol (see protocol.py).
// Every message is a frame: a 4-byte little-endian length and that many
// bytes. The first frame is a UTF-8 JSON header; binary attachments follow
// as one frame each, described by the header's "attachments" list.
// This assembly has no UnityEngine dependency so it also builds with .NET.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace EmbodiedUI.Protocol
{
    /// <summary>A peer sent something the protocol does not allow.</summary>
    public sealed class ProtocolException : Exception
    {
        public ProtocolException(string message) : base(message) { }
    }

    /// <summary>A named binary array: images, depth maps, meshes.</summary>
    public sealed class Attachment
    {
        public string Name;
        public string Dtype;
        public int[] Shape;
        public byte[] Data;

        public Attachment(string name, string dtype, int[] shape, byte[] data)
        {
            Name = name;
            Dtype = dtype;
            Shape = shape;
            Data = data;
        }

        /// <summary>An H x W x C uint8 image, top row first.</summary>
        public static Attachment Image(string name, int height, int width, int channels, byte[] data)
        {
            return new Attachment(name, "uint8", new[] { height, width, channels }, data);
        }

        public long ExpectedBytes()
        {
            long count = 1;
            foreach (var n in Shape) count *= n;
            return count * Wire.ItemSize(Dtype);
        }
    }

    /// <summary>One received message.</summary>
    public sealed class Message
    {
        public JObject Header;
        public List<Attachment> Attachments = new List<Attachment>();

        public string Type => (string)Header["type"];
    }

    public static class Wire
    {
        public const string Protocol = "embodied-ui/1";
        public const int DefaultPort = 47474;
        public const int MaxFrame = 256 * 1024 * 1024;

        public static int ItemSize(string dtype)
        {
            switch (dtype)
            {
                case "uint8": return 1;
                case "uint16": return 2;
                case "int32": return 4;
                case "float32": return 4;
                case "float64": return 8;
                default: throw new ProtocolException("unsupported attachment dtype " + dtype);
            }
        }

        public static void WriteFrame(Stream stream, byte[] payload)
        {
            if (payload.Length > MaxFrame)
                throw new ProtocolException("frame of " + payload.Length + " bytes exceeds the limit");
            var length = new byte[4];
            uint n = (uint)payload.Length;
            length[0] = (byte)n;
            length[1] = (byte)(n >> 8);
            length[2] = (byte)(n >> 16);
            length[3] = (byte)(n >> 24);
            stream.Write(length, 0, 4);
            stream.Write(payload, 0, payload.Length);
        }

        public static byte[] ReadFrame(Stream stream)
        {
            var length = ReadExact(stream, 4);
            uint n = (uint)(length[0] | (length[1] << 8) | (length[2] << 16) | (length[3] << 24));
            if (n > MaxFrame)
                throw new ProtocolException("peer announced a frame of " + n + " bytes");
            return ReadExact(stream, (int)n);
        }

        static byte[] ReadExact(Stream stream, int count)
        {
            var buffer = new byte[count];
            int read = 0;
            while (read < count)
            {
                int got = stream.Read(buffer, read, count - read);
                if (got == 0) throw new EndOfStreamException("peer closed the connection");
                read += got;
            }
            return buffer;
        }

        public static void Send(Stream stream, JObject header, IList<Attachment> attachments = null)
        {
            if (attachments != null && attachments.Count > 0)
            {
                var descriptors = new JArray();
                foreach (var a in attachments)
                {
                    if (a.Data.LongLength != a.ExpectedBytes())
                        throw new ProtocolException("attachment " + a.Name + " has the wrong size");
                    descriptors.Add(new JObject
                    {
                        ["name"] = a.Name,
                        ["dtype"] = a.Dtype,
                        ["shape"] = new JArray(a.Shape),
                    });
                }
                header["attachments"] = descriptors;
            }
            var json = header.ToString(Formatting.None);
            WriteFrame(stream, Encoding.UTF8.GetBytes(json));
            if (attachments != null)
                foreach (var a in attachments) WriteFrame(stream, a.Data);
            stream.Flush();
        }

        public static Message Receive(Stream stream)
        {
            var bytes = ReadFrame(stream);
            JObject header;
            try
            {
                var settings = new JsonSerializerSettings { FloatParseHandling = FloatParseHandling.Double };
                header = JsonConvert.DeserializeObject<JObject>(Encoding.UTF8.GetString(bytes), settings);
            }
            catch (JsonException error)
            {
                throw new ProtocolException("invalid JSON header: " + error.Message);
            }
            if (header == null || header["type"] == null || header["type"].Type != JTokenType.String)
                throw new ProtocolException("message headers must be objects with a string 'type'");
            var message = new Message { Header = header };
            if (header["attachments"] is JArray descriptors)
            {
                foreach (var d in descriptors)
                {
                    var shape = d["shape"].ToObject<int[]>();
                    var attachment = new Attachment((string)d["name"], (string)d["dtype"], shape, ReadFrame(stream));
                    if (attachment.Data.LongLength != attachment.ExpectedBytes())
                        throw new ProtocolException("attachment " + attachment.Name + " has the wrong size");
                    message.Attachments.Add(attachment);
                }
                header.Remove("attachments");
            }
            return message;
        }

        /// <summary>JSON number, or null for NaN and infinities (not valid JSON).</summary>
        public static JToken Number(double value)
        {
            return double.IsNaN(value) || double.IsInfinity(value) ? JValue.CreateNull() : new JValue(value);
        }

        /// <summary>Read a JSON number; null becomes NaN.</summary>
        public static double ToDouble(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null) return double.NaN;
            return token.Value<double>();
        }

        /// <summary>Accept peers with the same major protocol version.</summary>
        public static void CheckProtocol(JToken value)
        {
            var text = value?.Type == JTokenType.String ? (string)value : null;
            if (text == null || text.Split('.')[0] != Protocol)
                throw new ProtocolException("peer speaks " + (text ?? "nothing") + "; this side speaks " + Protocol);
        }
    }
}
