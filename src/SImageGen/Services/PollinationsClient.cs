using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace SImageGen.Services;

/// <summary>Image backend via the Pollinations gateway (their GPUs = seconds per image).
/// Fallback when the local model is unreachable; selectable via IMAGE_BACKEND=pollinations.</summary>
public class PollinationsClient : IImageBackend
{
    private readonly HttpClient _http;

    public string DefaultModel => Environment.GetEnvironmentVariable("IMAGE_MODEL") ?? "zimage";
    public string DefaultEditModel => Environment.GetEnvironmentVariable("IMAGE_EDIT_MODEL") ?? "gptimage";
    public string DefaultSize => Environment.GetEnvironmentVariable("IMAGE_SIZE") ?? "512x512";

    public PollinationsClient(HttpClient http)
    {
        _http = http;
        var key = Environment.GetEnvironmentVariable("POLLINATIONS_API_KEY");
        if (!string.IsNullOrWhiteSpace(key))
            _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", key);
    }

    public static (int W, int H) ParseSize(string size) => IImageBackend.ParseSize(size);

    /// <summary>Text-to-image via GET /image/{prompt}. Supports seed.</summary>
    public async Task<byte[]> GenerateAsync(string prompt, string model, int w, int h, int seed, CancellationToken ct)
    {
        var url = $"https://gen.pollinations.ai/image/{Uri.EscapeDataString(prompt)}"
                + $"?model={Uri.EscapeDataString(model)}&width={w}&height={h}&seed={seed}&nologo=true";
        using var resp = await _http.GetAsync(url, ct);
        var bytes = await resp.Content.ReadAsByteArrayAsync(ct);
        if (!resp.IsSuccessStatusCode)
            throw new InvalidOperationException($"Pollinations {(int)resp.StatusCode}: {Snip(bytes)}");
        if (!LooksLikeImage(resp, bytes))
            throw new InvalidOperationException("Pollinations did not return an image: " + Snip(bytes));
        return bytes;
    }

    /// <summary>Image-to-image via OpenAI-compatible POST /v1/images/edits (best effort).</summary>
    public async Task<byte[]> EditAsync(byte[] image, string fileName, string prompt, string model, string size, CancellationToken ct)
    {
        using var form = new MultipartFormDataContent();
        form.Add(new ByteArrayContent(image), "image", fileName);
        form.Add(new StringContent(prompt), "prompt");
        form.Add(new StringContent(model), "model");
        form.Add(new StringContent(size), "size");
        form.Add(new StringContent("b64_json"), "response_format");
        using var resp = await _http.PostAsync("https://gen.pollinations.ai/v1/images/edits", form, ct);
        var text = await resp.Content.ReadAsStringAsync(ct);
        if (!resp.IsSuccessStatusCode)
            throw new InvalidOperationException($"Pollinations edits {(int)resp.StatusCode}: {Snip(text)}");
        using var doc = JsonDocument.Parse(text);
        var b64 = doc.RootElement.GetProperty("data")[0].GetProperty("b64_json").GetString()
            ?? throw new InvalidOperationException("Pollinations edits: missing b64_json");
        return Convert.FromBase64String(b64);
    }

    public async Task<byte[]> DownloadAsync(string url, CancellationToken ct)
    {
        var bytes = await _http.GetByteArrayAsync(url, ct);
        if (bytes.Length == 0) throw new InvalidOperationException("Empty download: " + url);
        return bytes;
    }

    private static bool LooksLikeImage(HttpResponseMessage resp, byte[] b)
    {
        var mt = resp.Content.Headers.ContentType?.MediaType ?? "";
        if (mt.StartsWith("image/", StringComparison.OrdinalIgnoreCase)) return true;
        if (b.Length < 16) return false;
        var jpeg = b[0] == 0xFF && b[1] == 0xD8;                       // FF D8
        var png = b[0] == 0x89 && b[1] == 0x50 && b[2] == 0x4E && b[3] == 0x47; // .PNG
        var webp = b[8] == (byte)'W' && b[9] == (byte)'E' && b[10] == (byte)'B' && b[11] == (byte)'P';
        return jpeg || png || webp;
    }

    private static string Snip(byte[] b) => Snip(Encoding.UTF8.GetString(b, 0, Math.Min(b.Length, 300)));
    private static string Snip(string s) => s.Length > 300 ? s[..300] : s;
}
