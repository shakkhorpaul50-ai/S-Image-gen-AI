using CloudinaryDotNet;
using CloudinaryDotNet.Actions;

namespace SImageGen.Services;

/// <summary>Permanent PNG storage on Cloudinary free tier (Render disk is ephemeral).</summary>
public class ImageStore
{
    private readonly Cloudinary? _cloud;
    private readonly string _cloudName;
    private readonly ILogger<ImageStore> _log;
    public bool Enabled => _cloud is not null;

    public ImageStore(ILogger<ImageStore> log)
    {
        _log = log;
        try
        {
            // CLOUDINARY_URL format: cloudinary://API_KEY:API_SECRET@CLOUD_NAME
            var url = Environment.GetEnvironmentVariable("CLOUDINARY_URL")
                ?? throw new InvalidOperationException("CLOUDINARY_URL is not set.");
            var u = new Uri(url.Replace("cloudinary://", "https://"));
            var parts = u.UserInfo.Split(':');
            _cloudName = u.Host;
            _cloud = new Cloudinary(new Account(_cloudName, parts[0], parts[1]));
        }
        catch (Exception ex)
        {
            _log.LogWarning(ex, "CLOUDINARY_URL missing/invalid — uploads fall back to data URLs (dev only).");
            _cloud = null;
            _cloudName = "";
        }
    }

    public async Task<string> UploadPngAsync(byte[] png, string publicId, CancellationToken ct = default)
    {
        if (_cloud is null)
        {
            _log.LogWarning("Cloudinary not configured; returning data URL (not for production).");
            return "data:image/png;base64," + Convert.ToBase64String(png);
        }
        using var ms = new MemoryStream(png);
        var result = await _cloud.UploadAsync(new ImageUploadParams
        {
            File = new FileDescription(publicId + ".png", ms),
            PublicId = "simagegen/" + publicId,
            Overwrite = false,
            UniqueFilename = false
        });
        if (result.Error != null)
        {
            if (result.Error.Message.Contains("already exists", StringComparison.OrdinalIgnoreCase))
                return DeliveryUrl(publicId);
            throw new InvalidOperationException("Cloudinary: " + result.Error.Message);
        }
        return result.SecureUrl.ToString();
    }

    private string DeliveryUrl(string publicId)
        => $"https://res.cloudinary.com/{_cloudName}/image/upload/simagegen/{publicId}.png";
}
