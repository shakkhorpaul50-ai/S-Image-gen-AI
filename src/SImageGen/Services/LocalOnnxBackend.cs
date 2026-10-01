using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace SImageGen.Services;

/// <summary>Runs the custom 0.15B model in-process via ONNX Runtime (INT8).
/// Single-flight (one generation at a time) for the 0.1 CPU host; the queue
/// serialises the rest. Models download once from ONNX_BASE_URL into a cache dir.</summary>
public class LocalOnnxBackend : IImageBackend
{
    private const int Latent = 32, LatentCh = 4, ImgSize = 256, MaxLen = 77, TeDim = 448;
    private const int Steps = 24;
    private const float Cfg = 5.0f;

    private static readonly string[] Files = ["dit150m_s8.onnx", "te150m_s8.onnx", "vae150m_enc_s8.onnx", "vae150m_dec_s8.onnx"];

    private readonly SemaphoreSlim _slot = new(1, 1);
    private readonly object _initLock = new();
    private volatile bool _ready;
    private InferenceSession? _dit, _te, _vaeEnc, _vaeDec;
    private string[] _vocab = [];
    private double _scale = 1.0;
    private readonly ILogger<LocalOnnxBackend> _log;

    public LocalOnnxBackend(ILogger<LocalOnnxBackend> log) => _log = log;

    public string DefaultModel => "microdiffusion-0.15b-q3";
    public string DefaultEditModel => "microdiffusion-0.15b-q3";
    public string DefaultSize => "256x256";

    private static string BaseUrl => (Environment.GetEnvironmentVariable("ONNX_BASE_URL")
        ?? "https://github.com/shakkhorpaul50-ai/S-Image-gen-AI/releases/download/model-150m-v1")
        .TrimEnd('/');

    private static string CacheDir =>
        Environment.GetEnvironmentVariable("ONNX_CACHE_DIR") ?? "/tmp/simagegen-onnx";

    public async Task<byte[]> GenerateAsync(string prompt, string model, int w, int h, int seed, CancellationToken ct)
    {
        EnsureReady(ct);
        await _slot.WaitAsync(ct);
        try
        {
            return await Task.Run(() =>
            {
                var ids = Encode(prompt);
                var txt = RunTe(ids);
                var mask = MaskFrom(ids);
                var x = RandnLatent(seed == 0 ? Random.Shared.Next(1, 1_000_000_000) : seed, ref seed);
                float dt = 1f / Steps;
                for (var i = 0; i < Steps; i++)
                {
                    ct.ThrowIfCancellationRequested();
                    var t = 1f - i * dt;
                    var vc = RunDit(x, t, txt, mask);
                    var vu = RunDit(x, t, new float[1 * 77 * TeDim], new bool[1 * 77]);
                    for (var j = 0; j < x.Length; j++)
                        x[j] -= (vu[j] + Cfg * (vc[j] - vu[j])) * dt;
                }
                var img = RunVaeDec(Div(x, (float)_scale));
                return ToPng(img);
            }, ct);
        }
        finally
        {
            _slot.Release();
        }
    }

    public async Task<byte[]> EditAsync(byte[] image, string fileName, string prompt, string model, string size, CancellationToken ct)
    {
        EnsureReady(ct);
        await _slot.WaitAsync(ct);
        try
        {
            return await Task.Run(() =>
            {
                const float strength = 0.6f;
                var x0 = LoadImage(image);
                var mu = RunVaeEnc(x0);
                var z0 = Mul(mu, (float)_scale);
                var rng = new Random();
                var eps = Randn(4 * 32 * 32, rng);
                var x = new float[z0.Length];
                for (var j = 0; j < x.Length; j++)
                    x[j] = (1 - strength) * z0[j] + strength * eps[j];
                var ids = Encode(prompt);
                var txt = RunTe(ids);
                var mask = MaskFrom(ids);
                var n = Math.Max(1, (int)(Steps * strength));
                var dt = strength / n;
                for (var i = 0; i < n; i++)
                {
                    ct.ThrowIfCancellationRequested();
                    var t = strength - i * dt;
                    var vc = RunDit(x, t, txt, mask);
                    var vu = RunDit(x, t, new float[1 * 77 * TeDim], new bool[1 * 77]);
                    for (var j = 0; j < x.Length; j++)
                        x[j] -= (vu[j] + Cfg * (vc[j] - vu[j])) * dt;
                }
                var img = RunVaeDec(Div(x, (float)_scale));
                return ToPng(img);
            }, ct);
        }
        finally
        {
            _slot.Release();
        }
    }

    public async Task<byte[]> DownloadAsync(string url, CancellationToken ct)
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
        var bytes = await http.GetByteArrayAsync(url, ct);
        if (bytes.Length == 0) throw new InvalidOperationException("Empty download: " + url);
        return bytes;
    }

    // ---- model loading -------------------------------------------------

    private void EnsureReady(CancellationToken ct)
    {
        if (_ready) return;
        lock (_initLock)
        {
            if (_ready) return;
            Directory.CreateDirectory(CacheDir);
            foreach (var f in Files)
                DownloadFile($"{BaseUrl}/{f}", Path.Combine(CacheDir, f), ct);
            DownloadFile($"{BaseUrl}/tokenizer.json", Path.Combine(CacheDir, "tokenizer.json"), ct);
            LoadTokenizer(Path.Combine(CacheDir, "tokenizer.json"));
            var opts = new Microsoft.ML.OnnxRuntime.SessionOptions
            {
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
                InterOpNumThreads = 1,
                IntraOpNumThreads = 1,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
            };
            _dit = new InferenceSession(Path.Combine(CacheDir, Files[0]), opts);
            _te = new InferenceSession(Path.Combine(CacheDir, Files[1]), opts);
            _vaeEnc = new InferenceSession(Path.Combine(CacheDir, Files[2]), opts);
            _vaeDec = new InferenceSession(Path.Combine(CacheDir, Files[3]), opts);
            _ready = true;
            _log.LogInformation("Local ONNX backend ready ({Dir})", CacheDir);
        }
    }

    private static void DownloadFile(string url, string dest, CancellationToken ct)
    {
        if (File.Exists(dest)) return;
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        using var resp = http.GetAsync(url, ct).GetAwaiter().GetResult();
        resp.EnsureSuccessStatusCode();
        using var fs = File.OpenWrite(dest);
        resp.Content.CopyToAsync(fs, ct).GetAwaiter().GetResult();
    }

    private void LoadTokenizer(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        _vocab = doc.RootElement.GetProperty("vocab").EnumerateArray()
            .Select(e => e.GetString() ?? "").ToArray();
        _scale = doc.RootElement.GetProperty("scale").GetDouble();
    }

    // ---- tokenizer port (must match training encode_text exactly) -----

    private int[] Encode(string cap)
    {
        // stoi: 0=<pad> 1=<bos> 2=<eos> 3=<unk>, then vocab order
        var stoi = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < _vocab.Length; i++) stoi[_vocab[i]] = i;
        var words = cap.ToLowerInvariant().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var ids = new List<int> { 1 };  // BOS
        foreach (var w in words.Take(MaxLen - 2))
            ids.Add(stoi.TryGetValue(w, out var id) ? id : 3);
        ids.Add(2);  // EOS
        while (ids.Count < MaxLen) ids.Add(0);
        return [.. ids.Take(MaxLen)];
    }

    private static bool[] MaskFrom(int[] ids)
    {
        var m = new bool[ids.Length];
        for (var i = 0; i < ids.Length; i++) m[i] = ids[i] == 0;
        return m;
    }

    // ---- ORT runs (batch 1, fixed shapes matching the export) ---------

    private float[] RunTe(int[] ids)
    {
        var input = new DenseTensor<long>(ids.Select(i => (long)i).ToArray(), [1, MaxLen]);
        using var results = _te!.Run([NamedOnnxValue.CreateFromTensor("ids", input)]);
        return results.First().AsEnumerable<float>().ToArray();
    }

    private float[] RunDit(float[] x, float t, float[] txt, bool[] mask)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("x", new DenseTensor<float>(x, [1, LatentCh, Latent, Latent])),
            NamedOnnxValue.CreateFromTensor("t", new DenseTensor<float>(new[] { t }, new[] { 1 })),
            NamedOnnxValue.CreateFromTensor("txt", new DenseTensor<float>(txt, [1, MaxLen, TeDim])),
            NamedOnnxValue.CreateFromTensor("tmask", new DenseTensor<bool>(mask, [1, MaxLen])),
        };
        using var results = _dit!.Run(inputs);
        return results.First().AsEnumerable<float>().ToArray();
    }

    private float[] RunVaeEnc(float[] x)
    {
        using var results = _vaeEnc!.Run([NamedOnnxValue.CreateFromTensor("x", new DenseTensor<float>(x, [1, 3, ImgSize, ImgSize]))]);
        return results.First().AsEnumerable<float>().ToArray();
    }

    private float[] RunVaeDec(float[] z)
    {
        using var results = _vaeDec!.Run([NamedOnnxValue.CreateFromTensor("z", new DenseTensor<float>(z, [1, LatentCh, Latent, Latent]))]);
        return results.First().AsEnumerable<float>().ToArray();
    }

    // ---- tensor helpers ------------------------------------------------

    private static float[] RandnLatent(int seed, ref int usedSeed)
    {
        usedSeed = seed;
        return Randn(1 * LatentCh * Latent * Latent, new Random(seed));
    }

    private static float[] Randn(int n, Random rng)
    {
        // Box-Muller
        var out_ = new float[n];
        for (var i = 0; i < n; i += 2)
        {
            var u1 = 1.0 - rng.NextDouble();
            var u2 = 1.0 - rng.NextDouble();
            var r = Math.Sqrt(-2.0 * Math.Log(u1));
            out_[i] = (float)(r * Math.Cos(2.0 * Math.PI * u2));
            if (i + 1 < n) out_[i + 1] = (float)(r * Math.Sin(2.0 * Math.PI * u2));
        }
        return out_;
    }

    private static float[] Mul(float[] a, float s)
    {
        var r = new float[a.Length];
        for (var i = 0; i < a.Length; i++) r[i] = a[i] * s;
        return r;
    }

    private static float[] Div(float[] a, float s) => Mul(a, 1f / s);

    private static float[] LoadImage(byte[] bytes)
    {
        using var img = Image.Load<Rgb24>(bytes);
        img.Mutate(x => x.Resize(ImgSize, ImgSize));
        var out_ = new float[1 * 3 * ImgSize * ImgSize];
        for (var y = 0; y < ImgSize; y++)
            for (var x = 0; x < ImgSize; x++)
            {
                var px = img[x, y];
                var o = y * ImgSize + x;
                out_[o] = px.R / 127.5f - 1f;
                out_[ImgSize * ImgSize + o] = px.G / 127.5f - 1f;
                out_[2 * ImgSize * ImgSize + o] = px.B / 127.5f - 1f;
            }
        return out_;
    }

    private static byte[] ToPng(float[] img)
    {
        using var out_ = new Image<Rgb24>(ImgSize, ImgSize);
        for (var y = 0; y < ImgSize; y++)
            for (var x = 0; x < ImgSize; x++)
                for (var c = 0; c < 3; c++)
                {
                    var v = img[c * ImgSize * ImgSize + y * ImgSize + x];
                    var b = (byte)Math.Clamp((int)MathF.Round((v * 0.5f + 0.5f) * 255f), 0, 255);
                    var px = out_[x, y];
                    if (c == 0) px.R = b; else if (c == 1) px.G = b; else px.B = b;
                    out_[x, y] = px;
                }
        using var ms = new MemoryStream();
        out_.SaveAsPng(ms);
        return ms.ToArray();
    }
}
