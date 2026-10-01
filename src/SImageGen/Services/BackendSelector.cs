namespace SImageGen.Services;

/// <summary>Selects the generation backend.
/// auto (default): in-process ONNX model, with silent failover to the Pollinations
/// gateway on timeout/failure. local: ONNX only. space: remote custom endpoint
/// (+ gateway fallback). pollinations: gateway only.</summary>
public class BackendSelector
{
    private readonly IServiceProvider _sp;
    public BackendSelector(IServiceProvider sp) => _sp = sp;

    public static string Mode =>
        (Environment.GetEnvironmentVariable("IMAGE_BACKEND") ?? "auto").ToLowerInvariant();

    public static bool CustomConfigured =>
        !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("INFERENCE_URL"));

    public IImageBackend Primary => Mode switch
    {
        "pollinations" => Pollinations,
        "local" => Local,
        "space" => Custom,
        _ => Local,
    };

    public IImageBackend? Fallback => Mode switch
    {
        "pollinations" => null,
        "local" => null,
        "space" => CustomConfigured ? Pollinations : null,
        _ => Pollinations,   // auto
    };

    /// <summary>Failures worth retrying on the fallback backend: timeouts,
    /// connection-level problems and bad-gateway responses. Never 4xx
    /// (user error) or post-generation errors.</summary>
    public static bool IsTransientFailure(Exception ex) =>
        ex is HttpRequestException || ex is TaskCanceledException || ex is TimeoutException ||
        (ex is InvalidOperationException ioe &&
            (ioe.Message.Contains(" 502") || ioe.Message.Contains(" 503") || ioe.Message.Contains(" 504")));

    private IImageBackend Pollinations => _sp.GetRequiredService<PollinationsClient>();
    private IImageBackend Custom => _sp.GetRequiredService<CustomModelClient>();
    private IImageBackend Local => _sp.GetRequiredService<LocalOnnxBackend>();
}
