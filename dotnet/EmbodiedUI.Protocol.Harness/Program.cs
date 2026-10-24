// Self-tests of the Unity package's protocol layer, and a toy engine that
// serves it over TCP the way the Unity bridge does (one reply per frame).
//
//   dotnet run -- selftest
//   dotnet run -- serve [port]      prints "PORT <n>" once listening

using System;
using System.Collections.Generic;
using System.IO;
using EmbodiedUI.Protocol;
using Newtonsoft.Json.Linq;

static class Program
{
    static int Main(string[] args)
    {
        var mode = args.Length > 0 ? args[0] : "selftest";
        if (mode == "selftest") return SelfTest();
        if (mode == "serve") return Serve(args.Length > 1 ? int.Parse(args[1]) : 0);
        Console.Error.WriteLine("usage: selftest | serve [port]");
        return 2;
    }

    static void Check(bool condition, string what)
    {
        if (!condition) throw new Exception("self-test failed: " + what);
        Console.WriteLine("ok   " + what);
    }

    static int SelfTest()
    {
        // Framing round trip with an attachment and non-finite numbers.
        var buffer = new MemoryStream();
        var outputs = new Outputs();
        outputs.Set("speed", 3.5);
        outputs.Set("u", double.NaN);
        outputs.Set("pose", 1, 2, 3, 1, 0, 0, 0);
        outputs.Image("screen.image", 2, 3, 3, new byte[18]);
        Wire.Send(buffer, Replies.State("state", 7, 0.14, outputs), outputs.Attachments);
        buffer.Position = 0;
        var message = Wire.Receive(buffer);
        Check(message.Type == "state", "message type survives framing");
        Check((long)message.Header["frame"] == 7, "frame number");
        Check(message.Header["outputs"]["u"].Type == JTokenType.Null, "NaN travels as null");
        Check(message.Attachments.Count == 1 && message.Attachments[0].Data.Length == 18, "attachment bytes");
        Check(message.Header["attachments"] == null, "descriptors are consumed");

        // Request parsing.
        var step = Request.Parse(new Message
        {
            Header = JObject.Parse(
                "{\"type\":\"step\",\"frame\":3,\"time\":0.06,\"controls\":{\"wheel.value\":0.25,\"touch.u\":null}," +
                "\"events\":[{\"time\":0.06,\"source\":\"b\",\"kind\":\"press\",\"value\":1,\"data\":{\"u\":0.5}}]}")
        });
        Check(step.Kind == RequestKind.Step && step.Frame == 3, "step request");
        Check(step.Values["wheel.value"] == 0.25 && double.IsNaN(step.Values["touch.u"]), "controls and null");
        Check(step.Events[0].Kind == "press" && step.Events[0].Get("u") == 0.5, "events");

        var threw = false;
        try { Request.Parse(new Message { Header = JObject.Parse("{\"type\":\"hello\",\"protocol\":\"embodied-ui/2\",\"dt\":0.02}") }); }
        catch (ProtocolException) { threw = true; }
        Check(threw, "major version mismatch is rejected");

        threw = false;
        var huge = new MemoryStream(new byte[] { 0, 0, 0, 0x80 });
        try { Wire.ReadFrame(huge); }
        catch (ProtocolException) { threw = true; }
        Check(threw, "oversized frames are rejected");
        Console.WriteLine("all protocol self-tests passed");
        return 0;
    }

    // A toy engine: a 1-D cart pushed by vehicle.throttle (1000 N at 1)
    // and braked by vehicle.brake, a lamp toggled by presses of "b", and an
    // 8 x 4 display whose brightness follows the speed.
    static int Serve(int port)
    {
        using var server = new LockstepServer("127.0.0.1", port);
        Console.WriteLine("PORT " + server.Port);
        Console.Out.Flush();
        double dt = 0, time = 0, position = 0, speed = 0;
        long frame = 0;
        bool lamp = false, ready = false;
        var mass = 1000.0;
        while (true)
        {
            // Serve clients one after another until the process is stopped.
            if (!server.HasClient)
            {
                server.Accept(-1);
                ready = false;
            }
            var request = server.Next(30000);
            if (request == null) continue;
            switch (request.Kind)
            {
                case RequestKind.Describe:
                    server.Reply(Replies.Description("dotnet toy", "unity", ToyManifest()));
                    break;
                case RequestKind.Hello:
                    dt = request.Dt;
                    ready = true;
                    frame = 0;
                    time = 0;
                    server.Reply(Replies.Welcome("dotnet toy", dt, "unity", frame, time, Report(position, speed, lamp, out var a0)), a0);
                    break;
                case RequestKind.Reset:
                    if (!ready) { server.ReplyError("send hello first"); break; }
                    frame = 0; time = 0; position = 0; speed = 0; lamp = false;
                    server.Reply(Replies.State("state", frame, time, Report(position, speed, lamp, out var a1)), a1);
                    break;
                case RequestKind.Step:
                    if (!ready) { server.ReplyError("send hello first"); break; }
                    if (request.Frame != frame) { server.ReplyError("expected frame " + frame); break; }
                    request.Values.TryGetValue("vehicle.throttle", out var throttle);
                    request.Values.TryGetValue("vehicle.brake", out var brake);
                    foreach (var e in request.Events)
                        if (e.Source == "b" && e.Kind == "press") lamp = !lamp;
                    var force = 1000.0 * Clamp(throttle) - 5000.0 * Clamp(brake) * Math.Sign(speed);
                    speed += force / mass * dt;
                    position += speed * dt;
                    frame += 1;
                    time = frame * dt;
                    server.Reply(Replies.State("state", frame, time, Report(position, speed, lamp, out var a2)), a2);
                    break;
                case RequestKind.Close:
                    server.Reply(Replies.Bye());
                    server.Drop();
                    break;
            }
        }
    }

    static double Clamp(double value)
    {
        return double.IsNaN(value) ? 0 : Math.Max(0, Math.Min(1, value));
    }

    static Outputs Report(double position, double speed, bool lamp, out List<Attachment> attachments)
    {
        var outputs = new Outputs();
        outputs.Set("vehicle.speed", speed);
        // Unity convention: forward is +z.
        outputs.Set("vehicle.pose", 0, 0, position, 1, 0, 0, 0);
        outputs.Set("lamp.on", lamp);
        var pixels = new byte[4 * 8 * 3];
        var level = (byte)Math.Min(255, Math.Abs(speed) * 25);
        for (int i = 0; i < pixels.Length; i++) pixels[i] = level;
        outputs.Image("screen.image", 4, 8, 3, pixels);
        attachments = outputs.Attachments;
        return outputs;
    }

    static JObject ToyManifest()
    {
        // One push button 0.4 m ahead of the reference, in Unity axes; a
        // Unity control is pressed along its local +z, i.e. away from the user.
        return JObject.Parse(
            "{\"format\":\"embodied-ui/manifest\",\"version\":1,\"frame\":\"unity\",\"name\":\"toy\"," +
            "\"devices\":[{\"type\":\"PushButton\",\"name\":\"b\",\"pose\":{\"type\":\"Pose\",\"pos\":[0,0,0.4],\"quat\":[1,0,0,0]}}]}");
    }
}
