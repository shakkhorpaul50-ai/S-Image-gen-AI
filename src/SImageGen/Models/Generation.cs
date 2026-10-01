using System.ComponentModel.DataAnnotations;

namespace SImageGen.Models;

public class Generation
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public int UserId { get; set; }
    public Guid? ConversationId { get; set; }

    /// <summary>"t2i" or "i2i".</summary>
    [MaxLength(8)]
    public string Mode { get; set; } = "t2i";

    [MaxLength(500)]
    public string Prompt { get; set; } = "";

    public int Seed { get; set; }

    [MaxLength(16)]
    public string Size { get; set; } = "256x256";

    [MaxLength(64)]
    public string Model { get; set; } = "microdiffusion-0.15b-q3";

    /// <summary>SHA256(mode|prompt|seed|size|model[|image]) — repeat requests hit this instead of regenerating.</summary>
    [MaxLength(64)]
    public string PromptHash { get; set; } = "";

    public string? ImageUrl { get; set; }
    public string? InputImageUrl { get; set; }

    /// <summary>Queued | Running | Done | Failed.</summary>
    [MaxLength(16)]
    public string Status { get; set; } = "Queued";

    public string? Error { get; set; }
    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;
}
