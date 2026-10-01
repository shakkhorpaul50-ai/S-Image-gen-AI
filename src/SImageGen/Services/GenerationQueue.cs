using System.Threading.Channels;
using Microsoft.EntityFrameworkCore;
using SImageGen.Data;

namespace SImageGen.Services;

/// <summary>In-process async job queue (no Redis on free tier). Browser polls status; worker generates.</summary>
public class GenerationQueue : BackgroundService
{
    private readonly Channel<Guid> _q = Channel.CreateUnbounded<Guid>();
    private readonly IServiceScopeFactory _scopes;
    private readonly ILogger<GenerationQueue> _log;

    public GenerationQueue(IServiceScopeFactory scopes, ILogger<GenerationQueue> log)
    {
        _scopes = scopes;
        _log = log;
    }

    public void Enqueue(Guid id) => _q.Writer.TryWrite(id);

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        await foreach (var id in _q.Reader.ReadAllAsync(ct))
        {
            try
            {
                await ProcessAsync(id, ct);
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "Queue fatal for {Id}", id);
            }
        }
    }

    private async Task ProcessAsync(Guid id, CancellationToken ct)
    {
        using var scope = _scopes.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
        var selector = scope.ServiceProvider.GetRequiredService<BackendSelector>();
        var store = scope.ServiceProvider.GetRequiredService<ImageStore>();
        var primary = selector.Primary;

        var g = await db.Generations.FirstOrDefaultAsync(x => x.Id == id, ct);
        if (g is null || g.Status != "Queued") return;
        g.Status = "Running";
        await db.SaveChangesAsync(ct);

        try
        {
            var (w, h) = IImageBackend.ParseSize(g.Size);
            byte[]? src = g.Mode == "i2i"
                ? await primary.DownloadAsync(g.InputImageUrl!, ct)
                : null;
            byte[] bytes;
            try
            {
                bytes = g.Mode == "i2i"
                    ? await primary.EditAsync(src!, "input.png", g.Prompt, g.Model, g.Size, ct)
                    : await primary.GenerateAsync(g.Prompt, g.Model, w, h, g.Seed, ct);
            }
            catch (Exception ex) when (selector.Fallback is not null && BackendSelector.IsTransientFailure(ex))
            {
                // Silent auto-failover: custom host unreachable -> gateway.
                // The row records the model that actually served it.
                _log.LogWarning(ex, "Primary backend failed for {Id}; failing over to gateway", id);
                var fb = selector.Fallback;
                g.Model = g.Mode == "i2i" ? fb.DefaultEditModel : fb.DefaultModel;
                bytes = g.Mode == "i2i"
                    ? await fb.EditAsync(src!, "input.png", g.Prompt, g.Model, g.Size, ct)
                    : await fb.GenerateAsync(g.Prompt, g.Model, w, h, g.Seed, ct);
            }
            g.ImageUrl = await store.UploadPngAsync(bytes, g.PromptHash, ct);
            g.Status = "Done";
        }
        catch (Exception ex)
        {
            g.Status = "Failed";
            g.Error = ex.Message.Length > 500 ? ex.Message[..500] : ex.Message;
        }
        await db.SaveChangesAsync(ct);
    }
}
