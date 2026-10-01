namespace SImageGen.Services;

/// <summary>Selects the generation backend.
/// auto (default): own custom model when INFERENCE_URL is set, otherwise the
/// Pollinations gateway; transient custom-model failures fall back to gateway.
/// local: custom model only (failures surface as Failed). pollinations: gateway only.</summary>
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
        "local" => Custom,
        _ => CustomConfigured ? Custom : Pollinations,
    };

    public IImageBackend? Fallback =>
        Mode == "auto" && CustomConfigured ? Pollinations : null;

    /// <summary>Failures worth retrying on the fallback backend: connection-level
    /// problems and bad-gateway responses. Never 4xx (user error) or post-generation errors.</summary>
    public static bool IsTransientFailure(Exception ex) =>
        ex is HttpRequestException || ex is TaskCanceledException ||
        (ex is InvalidOperationException ioe &&
            (ioe.Message.Contains(" 502") || ioe.Message.Contains(" 503") || ioe.Message.Contains(" 504")));

    private IImageBackend Pollinations => _sp.GetRequiredService<PollinationsClient>();
    private IImageBackend Custom => _sp.GetRequiredService<CustomModelClient>();
}
