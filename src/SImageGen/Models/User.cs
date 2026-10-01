using System.ComponentModel.DataAnnotations;

namespace SImageGen.Models;

public class User
{
    public int Id { get; set; }

    [MaxLength(32)]
    public string Username { get; set; } = "";

    public string PasswordHash { get; set; } = "";

    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;

    public int DailyQuota { get; set; } = 50;

    public bool IsActive { get; set; } = true;
}
