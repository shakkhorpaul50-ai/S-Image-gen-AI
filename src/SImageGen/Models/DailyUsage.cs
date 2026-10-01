namespace SImageGen.Models;

/// <summary>One row per user per UTC day. PK(UserId, Day) makes quota consume atomic.</summary>
public class DailyUsage
{
    public int UserId { get; set; }
    public DateOnly Day { get; set; }
    public int Count { get; set; }
}
