using System.Text;
using System.Text.Json;

namespace SImageGen.Services;

/// <summary>Calls a self-hosted MicroDiffusion inference endpoint
/// (see inference/). Used only in IMAGE_BACKEND=space mode.</summary>
public class CustomModelClient : IImageBackend
{
    private readonly HttpClient _http;

    public string DefaultModel => "microdiffusion-0.15b-q3";
    public string DefaultEditModel => "microdiffusion-0.15b-q3";
    public string DefaultSize => "256x256";

    public CustomModelClient(HttpClient http) => _http = http;

    // Read per call (not in ctor) so a missing INFERENCE_URL never crashes startup;
    // it fails only the request that needs it, with a clear message.
    private static string BaseUrl => (Environment.GetEnvironmentVariable("INFERENCE_URL")
        ?? throw new InvalidOperationException(
            "INFERENCE_URL is not set. Point it at the inference endpoint, e.g. https://<space>.hf.space"))
        .TrimEnd('/');

    private static string? ApiKey => Environment.GetEnvironmentVariable("INFERENCE_KEY");

    public async Task<byte[]> GenerateAsync(string prompt, string model, int w, int h, int seed, CancellationToken ct)
    {
        var body = JsonSerializer.Serialize(new { prompt, seed, steps = 24, cfg = 5.0 });
        using var req = new HttpRequestMessage(HttpMethod.Post, BaseUrl + "/generate")
        {
            Content = new StringContent(body, Encoding.UTF8, "application/json")
        };
        Auth(req);
        using var resp = await _http.SendAsync(req, ct);
        var text = await resp.Content.ReadAsStringAsync(ct);
        if (!resp.IsSuccessStatusCode)
            throw new InvalidOperationException($"Inference {(int)resp.StatusCode}: {Snip(text)}");
        using var doc = JsonDocument.Parse(text);
        var b64 = doc.RootElement.GetProperty("image_b64").GetString()
            ?? throw new InvalidOperationException("Inference: missing image_b64");
        return Convert.FromBase64String(b64);
    }

    public async Task<byte[]> EditAsync(byte[] image, string fileName, string prompt, string model, string size, CancellationToken ct)
    {
        var body = JsonSerializer.Serialize(new
        {
            image_b64 = Convert.ToBase64String(image),
            prompt,
            strength = 0.6,
            steps = 24,
            cfg = 5.0,
            seed = 0
        });
        using var req = new HttpRequestMessage(HttpMethod.Post, BaseUrl + "/edit")
        {
            Content = new StringContent(body, Encoding.UTF8, "application/json")
        };
        Auth(req);
        using var resp = await _http.SendAsync(req, ct);
        var text = await resp.Content.ReadAsStringAsync(ct);
        if (!resp.IsSuccessStatusCode)
            throw new InvalidOperationException($"Inference {(int)resp.StatusCode}: {Snip(text)}");
        using var doc = JsonDocument.Parse(text);
        var b64 = doc.RootElement.GetProperty("image_b64").GetString()
            ?? throw new InvalidOperationException("Inference: missing image_b64");
        return Convert.FromBase64String(b64);
    }

    public async Task<byte[]> DownloadAsync(string url, CancellationToken ct)
    {
        var bytes = await _http.GetByteArrayAsync(url, ct);
        if (bytes.Length == 0) throw new InvalidOperationException("Empty download: " + url);
        return bytes;
    }

    private static void Auth(HttpRequestMessage req)
    {
        if (!string.IsNullOrWhiteSpace(ApiKey))
            req.Headers.Add("X-Api-Key", ApiKey);
    }

    private static string Snip(string s) => s.Length > 300 ? s[..300] : s;
}
