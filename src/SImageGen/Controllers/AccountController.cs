using System.Security.Claims;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.RateLimiting;
using Microsoft.EntityFrameworkCore;
using SImageGen.Data;
using SImageGen.Models;

namespace SImageGen.Controllers;

public class AccountController : Controller
{
    private readonly AppDbContext _db;
    public AccountController(AppDbContext db) => _db = db;

    [HttpGet]
    public IActionResult Register() => View(new RegisterVm());

    [HttpGet]
    public IActionResult Login(string? returnUrl = null) => View(new LoginVm { ReturnUrl = returnUrl });

    [HttpPost, ValidateAntiForgeryToken]
    [EnableRateLimiting("auth-register")]
    public async Task<IActionResult> Register(RegisterVm vm)
    {
        if (!ModelState.IsValid) return View(vm);
        var name = vm.Username.Trim();
        if (await _db.Users.AnyAsync(u => u.Username == name))
        {
            ModelState.AddModelError(nameof(vm.Username), "Username is taken.");
            return View(vm);
        }
        var user = new User { Username = name, PasswordHash = BCrypt.Net.BCrypt.HashPassword(vm.Password) };
        _db.Users.Add(user);
        await _db.SaveChangesAsync();
        await SignInAsync(user);
        return RedirectToAction("Index", "Generate");
    }

    [HttpPost, ValidateAntiForgeryToken]
    [EnableRateLimiting("auth-login")]
    public async Task<IActionResult> Login(LoginVm vm)
    {
        if (!ModelState.IsValid) return View(vm);
        var user = await _db.Users.FirstOrDefaultAsync(u => u.Username == vm.Username.Trim());
        if (user is null || !user.IsActive || !BCrypt.Net.BCrypt.Verify(vm.Password, user.PasswordHash))
        {
            ModelState.AddModelError("", "Invalid username or password.");
            return View(vm);
        }
        await SignInAsync(user);
        if (!string.IsNullOrEmpty(vm.ReturnUrl) && Url.IsLocalUrl(vm.ReturnUrl)) return Redirect(vm.ReturnUrl);
        return RedirectToAction("Index", "Generate");
    }

    [HttpPost, ValidateAntiForgeryToken]
    public async Task<IActionResult> Logout()
    {
        await HttpContext.SignOutAsync();
        return RedirectToAction("Index", "Home");
    }

    private Task SignInAsync(User user)
    {
        var claims = new[]
        {
            new Claim(ClaimTypes.NameIdentifier, user.Id.ToString()),
            new Claim(ClaimTypes.Name, user.Username)
        };
        return HttpContext.SignInAsync(new ClaimsPrincipal(
            new ClaimsIdentity(claims, CookieAuthenticationDefaults.AuthenticationScheme)));
    }
}
