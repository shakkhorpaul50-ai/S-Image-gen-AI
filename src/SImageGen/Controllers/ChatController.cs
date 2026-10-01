using System.Security.Claims;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using SImageGen.Data;
using SImageGen.Models;
using SImageGen.Services;

namespace SImageGen.Controllers;

/// <summary>ChatGPT-style image threads. Every message generates an image
/// (text prompt, or attached photo = img2img restyle).</summary>
[Authorize]
public class ChatController : Controller
{
    private readonly AppDbContext _db;
    private readonly QuotaService _quota;
    private readonly GenerationQueue _queue;
    private readonly ImageStore _store;
    private readonly BackendSelector _backends;

    public ChatController(AppDbContext db, QuotaService quota, GenerationQueue queue,
        ImageStore store, BackendSelector backends)
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
        var convos = await SidebarAsync();
        var latest = convos.FirstOrDefault();
        if (latest is null)
        {
            var (u0, q0) = await _quota.GetUsageAsync(UserId);
            ViewBag.Used = u0;
            ViewBag.Quota = q0;
            ViewBag.Convos = convos;
            ViewBag.IsNew = true;
            ViewBag.ConvoId = null;
            ViewBag.Title = "New chat";
            return View("Thread", new List<Message>());
        }
        return RedirectToAction("Thread", new { id = latest.Id });
    }

    [HttpGet]
    public async Task<IActionResult> Thread(Guid id)
    {
        var convo = await OwnConvoAsync(id);
        if (convo is null) return NotFound();
        ViewBag.Convos = await SidebarAsync();
        var (used, quota) = await _quota.GetUsageAsync(UserId);
        ViewBag.Used = used;
        ViewBag.Quota = quota;
        ViewBag.Title = convo.Title;
        ViewBag.IsNew = false;
        ViewBag.ConvoId = id;
        var messages = await _db.Messages
            .Where(m => m.ConversationId == id)
            .OrderBy(m => m.CreatedAtUtc)
            .ToListAsync();
        return View(messages);
    }

    [HttpPost, ValidateAntiForgeryToken]
    [RequestSizeLimit(6_000_000)]
    public async Task<IActionResult> Send(Guid? conversationId, string prompt, IFormFile? image, int? seed)
    {
        if (string.IsNullOrWhiteSpace(prompt) || prompt.Length > 500)
            return BadRequest("Prompt is required (3-500 chars).");
        prompt = prompt.Trim();
        var mode = "t2i";

        byte[]? srcBytes = null;
        if (image is not null && image.Length > 0)
        {
            if (image.Length > 5_000_000) return BadRequest("Max 5 MB.");
            if (!image.ContentType.StartsWith("image/")) return BadRequest("File must be an image.");
            using var ms = new MemoryStream();
            await image.CopyToAsync(ms);
            srcBytes = ms.ToArray();
            mode = "i2i";
        }

        Guid convoId;
        if (conversationId is { } cid && await OwnConvoAsync(cid) is not null)
        {
            convoId = cid;
        }
        else
        {
            var convo = new Conversation
            {
                UserId = UserId,
                Title = prompt.Length > 40 ? prompt[..40] : prompt
            };
            _db.Conversations.Add(convo);
            await _db.SaveChangesAsync();
            convoId = convo.Id;
        }

        var model = mode == "i2i" ? _backends.Primary.DefaultEditModel : _backends.Primary.DefaultModel;
        var size = _backends.Primary.DefaultSize;
        var seedVal = seed ?? Random.Shared.Next(1, 1_000_000_000);
        var hash = mode == "i2i"
            ? PromptHash.ForImage(srcBytes!, prompt, size, model)
            : PromptHash.ForText(prompt, seedVal, size, model);

        _db.Messages.Add(new Message
        {
            ConversationId = convoId,
            Role = "user",
            TextContent = prompt,
            Status = "Done"
        });

        // Speed path: identical request already generated -> instant assistant message.
        var cached = await _db.Generations
            .FirstOrDefaultAsync(g => g.PromptHash == hash && g.Status == "Done" && g.ImageUrl != null);
        if (cached is not null)
        {
            _db.Messages.Add(new Message
            {
                ConversationId = convoId,
                Role = "assistant",
                TextContent = prompt,
                ImageUrl = cached.ImageUrl,
                GenerationId = cached.Id,
                Status = "Done"
            });
            await _db.SaveChangesAsync();
            return RedirectToAction("Thread", new { id = convoId });
        }

        var (ok, used, quota) = await _quota.TryConsumeAsync(UserId);
        if (!ok)
        {
            await _db.SaveChangesAsync();
            return RedirectToAction("Thread", new { id = convoId });
        }

        var g = new Generation
        {
            UserId = UserId,
            ConversationId = convoId,
            Mode = mode,
            Prompt = prompt,
            Seed = seedVal,
            Size = size,
            Model = model,
            PromptHash = hash,
            Status = "Queued"
        };
        if (mode == "i2i")
            g.InputImageUrl = await _store.UploadPngAsync(srcBytes!, "src-" + hash);
        _db.Generations.Add(g);
        await _db.SaveChangesAsync();
        _db.Messages.Add(new Message
        {
            ConversationId = convoId,
            Role = "assistant",
            TextContent = prompt,
            GenerationId = g.Id,
            Status = "Running"
        });
        await _db.SaveChangesAsync();
        _queue.Enqueue(g.Id);
        return RedirectToAction("Thread", new { id = convoId });
    }

    [HttpGet]
    public async Task<IActionResult> MessagesJson(Guid id, DateTime? after)
    {
        if (await OwnConvoAsync(id) is null) return NotFound();
        var q = _db.Messages.Where(m => m.ConversationId == id);
        if (after is not null)
            q = q.Where(m => m.CreatedAtUtc > after);
        var msgs = await q.OrderBy(m => m.CreatedAtUtc).ToListAsync();
        return Json(msgs.Select(m => new
        {
            id = m.Id,
            role = m.Role,
            text = m.TextContent,
            imageUrl = m.ImageUrl,
            status = m.Status,
            error = m.Error,
            at = m.CreatedAtUtc
        }));
    }

    [HttpPost, ValidateAntiForgeryToken]
    public async Task<IActionResult> Delete(Guid id)
    {
        var convo = await OwnConvoAsync(id);
        if (convo is null) return NotFound();
        _db.Conversations.Remove(convo);
        await _db.SaveChangesAsync();
        return RedirectToAction("Index");
    }

    private Task<Conversation?> OwnConvoAsync(Guid id)
        => _db.Conversations.FirstOrDefaultAsync(c => c.Id == id && c.UserId == UserId);

    private async Task<List<Conversation>> SidebarAsync() =>
        await _db.Conversations
            .Where(c => c.UserId == UserId)
            .OrderByDescending(c => c.CreatedAtUtc)
            .Take(50)
            .ToListAsync();
}
