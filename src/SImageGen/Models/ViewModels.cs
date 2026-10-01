using System.ComponentModel.DataAnnotations;

namespace SImageGen.Models;

public class RegisterVm
{
    [Required, StringLength(32, MinimumLength = 3)]
    [RegularExpression("^[a-zA-Z0-9_]+$", ErrorMessage = "Letters, digits and underscore only.")]
    public string Username { get; set; } = "";

    [Required, StringLength(100, MinimumLength = 6)]
    [DataType(DataType.Password)]
    public string Password { get; set; } = "";

    [DataType(DataType.Password)]
    [Compare(nameof(Password), ErrorMessage = "Passwords do not match.")]
    public string ConfirmPassword { get; set; } = "";
}

public class LoginVm
{
    [Required]
    public string Username { get; set; } = "";

    [Required, DataType(DataType.Password)]
    public string Password { get; set; } = "";

    public string? ReturnUrl { get; set; }
}

public class QuotaVm
{
    public int Used { get; set; }
    public int Quota { get; set; }
}
