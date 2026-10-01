using System.Security.Cryptography;
using System.Text;

namespace SImageGen.Services;

/// <summary>Deterministic cache key so identical requests are served instantly.</summary>
public static class PromptHash
{
    public static string ForText(string prompt, int seed, string size, string model)
        => Hex($"t2i|{prompt.Trim()}|{seed}|{size}|{model}");

    public static string ForImage(byte[] image, string prompt, string size, string model)
        => Hex($"i2i|{prompt.Trim()}|{size}|{model}|{Convert.ToHexString(SHA256.HashData(image))}");

    private static string Hex(string s)
        => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(s))).ToLowerInvariant()[..32];
}
