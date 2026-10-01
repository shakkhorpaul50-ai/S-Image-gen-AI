using System.Security.Claims;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using SImageGen.Data;
using SImageGen.Models;
using SImageGen.Services;

namespace SImageGen.Controllers;

[Authorize]
public class GenerateController : Controller
{
    private readonly AppDbContext _db;
    private readonly QuotaService _quota;
    private readonly GenerationQueue _queue;
    private readonly ImageStore _store;
    private readonly BackendSelector _backends;

    public GenerateController(AppDbContext db, QuotaService quota, GenerationQueue queue, ImageStore store, BackendSelector backends)
    {
        _db = db;
        _quota = quota;
        _queue = queue;
        _store = store;
        _backends = backends;
    }

    private int UserId => int.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!);

    [HttpGet]
    public async Task<IActionResult> Index()
    {
        var (used, quota) = await _quota.GetUsageAsync(UserId);
        ViewBag.Used = used;
        ViewBag.Quota = quota;
        return View(new GenerateVm());
    }

    [HttpPost, ValidateAntiForgeryToken]
    [RequestSizeLimit(6_000_000)]
    public async Task<IActionResult> Create(GenerateVm vm)
    {
        if (string.IsNullOrWhiteSpace(vm.Prompt) || vm.Prompt.Length > 500)
            ModelState.AddModelError(nameof(vm.Prompt), "Prompt is required (3-500 chars).");
        if (vm.Mode != "i2i") vm.Mode = "t2i";

        byte[]? srcBytes = null;
        if (vm.Mode == "i2i")
        {
            if (vm.Image is null || vm.Image.Length == 0)
                ModelState.AddModelError(nameof(vm.Image), "Upload a source image.");
            else if (vm.Image.Length > 5_000_000)
                ModelState.AddModelError(nameof(vm.Image), "Max 5 MB.");
            else if (!vm.Image.ContentType.StartsWith("image/"))
                ModelState.AddModelError(nameof(vm.Image), "File must be an image.");
            else
            {
                using var ms = new MemoryStream();
                await vm.Image.CopyToAsync(ms);
                srcBytes = ms.ToArray();
            }
        }
        if (!ModelState.IsValid)
        {
            var (u, q) = await _quota.GetUsageAsync(UserId);
            ViewBag.Used = u;
            ViewBag.Quota = q;
            return View("Index", vm);
        }

        var model = vm.Mode == "i2i" ? _backends.Primary.DefaultEditModel : _backends.Primary.DefaultModel;
        var size = _backends.Primary.DefaultSize;
        var seed = vm.Seed ?? Random.Shared.Next(1, 1_000_000_000);
        var hash = vm.Mode == "i2i"
            ? PromptHash.ForImage(srcBytes!, vm.Prompt, size, model)
            : PromptHash.ForText(vm.Prompt, seed, size, model);

        // Speed path: identical request already generated -> serve instantly, no quota consumed.
        var cached = await _db.Generations
            .FirstOrDefaultAsync(g => g.PromptHash == hash && g.Status == "Done" && g.ImageUrl != null);
        if (cached is not null)
            return RedirectToAction("Result", new { id = cached.Id });

        var (ok, used, quota) = await _quota.TryConsumeAsync(UserId);
        if (!ok) return View("Quota", new QuotaVm { Used = used, Quota = quota });

        var g = new Generation
        {
            UserId = UserId,
            Mode = vm.Mode,
            Prompt = vm.Prompt.Trim(),
            Seed = seed,
            Size = size,
            Model = model,
            PromptHash = hash,
            Status = "Queued"
        };
        if (vm.Mode == "i2i")
            g.InputImageUrl = await _store.UploadPngAsync(srcBytes!, "src-" + hash);
        _db.Generations.Add(g);
        await _db.SaveChangesAsync();
        _queue.Enqueue(g.Id);
        return RedirectToAction("Status", new { id = g.Id });
    }

    [HttpGet]
    public async Task<IActionResult> Status(Guid id)
    {
        var g = await OwnAsync(id);
        if (g is null) return NotFound();
        return View(id);
    }

    [HttpGet]
    public async Task<IActionResult> StatusJson(Guid id)
    {
        var g = await OwnAsync(id);
        if (g is null) return NotFound();
        return Json(new { status = g.Status, imageUrl = g.ImageUrl, error = g.Error });
    }

    [HttpGet]
    public async Task<IActionResult> Result(Guid id)
    {
        var g = await OwnAsync(id);
        if (g is null) return NotFound();
        return View(g);
    }

    private Task<Generation?> OwnAsync(Guid id)
        => _db.Generations.FirstOrDefaultAsync(g => g.Id == id && g.UserId == UserId);
}
