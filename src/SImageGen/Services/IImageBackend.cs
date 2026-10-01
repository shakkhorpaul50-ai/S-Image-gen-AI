namespace SImageGen.Services;

/// <summary>Image generation backend. Implementations: CustomModelClient (own 0.15B
/// model via the inference endpoint) and PollinationsClient (gateway fallback).</summary>
public interface IImageBackend
{
    string DefaultModel { get; }
    string DefaultEditModel { get; }
    string DefaultSize { get; }

    Task<byte[]> GenerateAsync(string prompt, string model, int w, int h, int seed, CancellationToken ct);
    Task<byte[]> EditAsync(byte[] image, string fileName, string prompt, string model, string size, CancellationToken ct);
    Task<byte[]> DownloadAsync(string url, CancellationToken ct);

    static (int W, int H) ParseSize(string size)
    {
        var p = (size ?? "").Split('x');
        if (p.Length == 2 && int.TryParse(p[0], out var w) && int.TryParse(p[1], out var h) && w > 0 && h > 0)
            return (Math.Min(w, 1024), Math.Min(h, 1024));
        return (512, 512);
    }
}
