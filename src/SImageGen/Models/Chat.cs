using System.ComponentModel.DataAnnotations;

namespace SImageGen.Models;

/// <summary>A chat thread: an ordered list of user prompts and assistant images.</summary>
public class Conversation
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public int UserId { get; set; }

    [MaxLength(120)]
    public string Title { get; set; } = "New chat";

    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;
}

/// <summary>One bubble in a thread. Role is "user" (prompt text) or "assistant"
/// (generated image, linked to its Generation row for seed/model details).</summary>
public class Message
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public Guid ConversationId { get; set; }

    [MaxLength(16)]
    public string Role { get; set; } = "user";

    public string? TextContent { get; set; }
    public string? ImageUrl { get; set; }
    public Guid? GenerationId { get; set; }

    /// <summary>Done | Running | Failed (assistant image messages; user messages are Done).</summary>
    [MaxLength(16)]
    public string Status { get; set; } = "Done";

    public string? Error { get; set; }
    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;
}
