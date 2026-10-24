// Typed requests from the MuJoCo side and the replies an engine sends.

using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;

namespace EmbodiedUI.Protocol
{
    /// <summary>A discrete device event: press, toggle, detent, touch_begin...</summary>
    public sealed class DeviceEvent
    {
        public double Time;
        public string Source;
        public string Kind;
        public double Value;
        public Dictionary<string, double> Data = new Dictionary<string, double>();

        public static DeviceEvent FromJson(JToken token)
        {
            var e = new DeviceEvent
            {
                Time = Wire.ToDouble(token["time"]),
                Source = (string)token["source"],
                Kind = (string)token["kind"],
                Value = Wire.ToDouble(token["value"]),
            };
            if (token["data"] is JObject data)
                foreach (var pair in data) e.Data[pair.Key] = Wire.ToDouble(pair.Value);
            return e;
        }

        public double Get(string key, double fallback = double.NaN)
        {
            return Data.TryGetValue(key, out var value) ? value : fallback;
        }
    }

    public enum RequestKind { Hello, Describe, Step, Reset, Close }

    /// <summary>A parsed client request.</summary>
    public sealed class Request
    {
        public RequestKind Kind;
        public double Dt;
        public List<string> Controls = new List<string>();
        public List<string> Outputs = new List<string>();
        public JArray Manifest = new JArray();
        public long Frame;
        public double Time;
        public Dictionary<string, double> Values = new Dictionary<string, double>();
        public List<DeviceEvent> Events = new List<DeviceEvent>();
        public int? Seed;
        public JObject Options = new JObject();

        public static Request Parse(Message message)
        {
            var h = message.Header;
            switch (message.Type)
            {
                case "hello":
                    Wire.CheckProtocol(h["protocol"]);
                    var hello = new Request { Kind = RequestKind.Hello, Dt = Wire.ToDouble(h["dt"]) };
                    if (!(hello.Dt > 0) || double.IsInfinity(hello.Dt))
                        throw new ProtocolException("hello needs a positive finite dt");
                    if (h["controls"] is JArray controls) foreach (var c in controls) hello.Controls.Add((string)c);
                    if (h["outputs"] is JArray outputs) foreach (var o in outputs) hello.Outputs.Add((string)o);
                    if (h["manifest"] is JArray manifest) hello.Manifest = manifest;
                    return hello;
                case "describe":
                    Wire.CheckProtocol(h["protocol"]);
                    return new Request { Kind = RequestKind.Describe };
                case "step":
                    var step = new Request
                    {
                        Kind = RequestKind.Step,
                        Frame = h["frame"]?.Value<long>() ?? 0,
                        Time = Wire.ToDouble(h["time"]),
                    };
                    if (h["controls"] is JObject values)
                        foreach (var pair in values) step.Values[pair.Key] = Wire.ToDouble(pair.Value);
                    if (h["events"] is JArray events)
                        foreach (var e in events) step.Events.Add(DeviceEvent.FromJson(e));
                    return step;
                case "reset":
                    var reset = new Request { Kind = RequestKind.Reset };
                    if (h["seed"] != null && h["seed"].Type == JTokenType.Integer) reset.Seed = h["seed"].Value<int>();
                    if (h["options"] is JObject options) reset.Options = options;
                    return reset;
                case "close":
                    return new Request { Kind = RequestKind.Close };
                default:
                    throw new ProtocolException("unknown message type " + message.Type);
            }
        }
    }

    /// <summary>Outputs an engine reports after a frame.</summary>
    public sealed class Outputs
    {
        public readonly JObject Values = new JObject();
        public readonly List<Attachment> Attachments = new List<Attachment>();

        public void Set(string name, double value) { Values[name] = Wire.Number(value); }
        public void Set(string name, bool value) { Values[name] = value ? 1.0 : 0.0; }
        public void Set(string name, string value) { Values[name] = value; }

        public void Set(string name, params double[] values)
        {
            var array = new JArray();
            foreach (var v in values) array.Add(Wire.Number(v));
            Values[name] = array;
        }

        public void Image(string name, int height, int width, int channels, byte[] data)
        {
            Attachments.Add(Attachment.Image(name, height, width, channels, data));
        }

        public void Clear()
        {
            Values.RemoveAll();
            Attachments.Clear();
        }
    }

    public static class Replies
    {
        public static JObject Welcome(string engine, double dt, string convention, long frame, double time, Outputs outputs)
        {
            var header = State("welcome", frame, time, outputs);
            header["protocol"] = Wire.Protocol;
            header["engine"] = engine;
            header["dt"] = dt;
            header["convention"] = convention;
            return header;
        }

        public static JObject State(string type, long frame, double time, Outputs outputs)
        {
            return new JObject
            {
                ["type"] = type,
                ["frame"] = frame,
                ["time"] = Wire.Number(time),
                ["outputs"] = outputs != null ? (JObject)outputs.Values.DeepClone() : new JObject(),
            };
        }

        public static JObject Description(string engine, string convention, JObject manifest)
        {
            return new JObject
            {
                ["type"] = "description",
                ["protocol"] = Wire.Protocol,
                ["engine"] = engine,
                ["convention"] = convention,
                ["manifest"] = manifest,
            };
        }

        public static JObject Error(string message)
        {
            return new JObject { ["type"] = "error", ["message"] = message };
        }

        public static JObject Bye()
        {
            return new JObject { ["type"] = "bye" };
        }
    }
}
