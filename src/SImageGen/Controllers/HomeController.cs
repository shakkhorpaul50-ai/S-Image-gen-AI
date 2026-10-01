using Microsoft.AspNetCore.Mvc;

namespace SImageGen.Controllers;

public class HomeController : Controller
{
    [HttpGet]
    public IActionResult Index() => View();

    [HttpGet("/Home/Error")]
    public IActionResult Error() => View();
}
