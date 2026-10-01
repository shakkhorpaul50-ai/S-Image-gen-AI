using System.Security.Claims;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using SImageGen.Data;

namespace SImageGen.Controllers;

[Authorize]
public class GalleryController : Controller
{
    private readonly AppDbContext _db;
    public GalleryController(AppDbContext db) => _db = db;

    [HttpGet]
    public async Task<IActionResult> Index()
    {
        var uid = int.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!);
        var items = await _db.Generations
            .Where(g => g.UserId == uid)
            .OrderByDescending(g => g.CreatedAtUtc)
            .Take(60)
            .ToListAsync();
        return View(items);
    }
}
