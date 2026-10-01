using Microsoft.EntityFrameworkCore;
using SImageGen.Models;

namespace SImageGen.Data;

public class AppDbContext : DbContext
{
    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

    public DbSet<User> Users => Set<User>();
    public DbSet<DailyUsage> DailyUsages => Set<DailyUsage>();
    public DbSet<Generation> Generations => Set<Generation>();
    public DbSet<Conversation> Conversations => Set<Conversation>();
    public DbSet<Message> Messages => Set<Message>();

    protected override void OnModelCreating(ModelBuilder b)
    {
        b.Entity<User>().HasIndex(u => u.Username).IsUnique();
        b.Entity<DailyUsage>().HasKey(d => new { d.UserId, d.Day });
        b.Entity<Generation>().HasIndex(g => g.PromptHash);
        b.Entity<Generation>().HasIndex(g => new { g.UserId, g.CreatedAtUtc });
        b.Entity<Generation>()
            .HasOne<User>()
            .WithMany()
            .HasForeignKey(g => g.UserId)
            .OnDelete(DeleteBehavior.Cascade);
        b.Entity<Conversation>().HasIndex(c => new { c.UserId, c.CreatedAtUtc });
        b.Entity<Conversation>()
            .HasOne<User>()
            .WithMany()
            .HasForeignKey(c => c.UserId)
            .OnDelete(DeleteBehavior.Cascade);
        b.Entity<Message>().HasIndex(m => new { m.ConversationId, m.CreatedAtUtc });
        b.Entity<Message>()
            .HasOne<Conversation>()
            .WithMany()
            .HasForeignKey(m => m.ConversationId)
            .OnDelete(DeleteBehavior.Cascade);
    }
}
