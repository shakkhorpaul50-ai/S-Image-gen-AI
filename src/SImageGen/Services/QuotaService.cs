using Microsoft.EntityFrameworkCore;
using SImageGen.Data;
using SImageGen.Models;

namespace SImageGen.Services;

/// <summary>Enforces N generations per user per UTC day.</summary>
public class QuotaService
{
    private readonly AppDbContext _db;
    public QuotaService(AppDbContext db) => _db = db;

    public async Task<(bool Allowed, int Used, int Quota)> TryConsumeAsync(int userId, CancellationToken ct = default)
    {
        var user = await _db.Users.FirstOrDefaultAsync(u => u.Id == userId, ct);
        if (user is null || !user.IsActive) return (false, 0, 0);
        var today = DateOnly.FromDateTime(DateTime.UtcNow);
        using var tx = await _db.Database.BeginTransactionAsync(ct);
        var row = await _db.DailyUsages.FirstOrDefaultAsync(d => d.UserId == userId && d.Day == today, ct);
        if (row is null)
        {
            row = new DailyUsage { UserId = userId, Day = today, Count = 0 };
            _db.DailyUsages.Add(row);
        }
        if (row.Count >= user.DailyQuota)
        {
            await tx.RollbackAsync(ct);
            return (false, row.Count, user.DailyQuota);
        }
        row.Count++;
        await _db.SaveChangesAsync(ct);
        await tx.CommitAsync(ct);
        return (true, row.Count, user.DailyQuota);
    }

    public async Task<(int Used, int Quota)> GetUsageAsync(int userId, CancellationToken ct = default)
    {
        var user = await _db.Users.FirstOrDefaultAsync(u => u.Id == userId, ct);
        if (user is null) return (0, 0);
        var today = DateOnly.FromDateTime(DateTime.UtcNow);
        var row = await _db.DailyUsages.FirstOrDefaultAsync(d => d.UserId == userId && d.Day == today, ct);
        return (row?.Count ?? 0, user.DailyQuota);
    }
}
