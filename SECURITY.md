# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **Do NOT open a public GitHub issue** for security vulnerabilities
2. Email the maintainers directly or use GitHub's private vulnerability reporting feature
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Resolution Timeline**: Depends on severity
  - Critical: 24-72 hours
  - High: 1-2 weeks
  - Medium: 2-4 weeks
  - Low: Next release

### Safe Harbor

We consider security research conducted in good faith to be authorized and will not pursue legal action against researchers who:

- Make a good faith effort to avoid privacy violations, data destruction, and service interruption
- Only interact with accounts you own or have explicit permission to access
- Do not exploit a security issue beyond what is necessary to demonstrate the vulnerability
- Report vulnerabilities promptly and allow reasonable time for remediation before disclosure

## Security Measures

This project implements the following security practices:

### Automated Scanning

- **Bandit**: Static security analysis on every PR
- **Safety**: Dependency vulnerability scanning
- **Gitleaks**: Secret detection in pre-commit hooks

### Code Review

- All changes require PR review
- Security-sensitive changes receive additional scrutiny

### Dependencies

- Dependencies are pinned to specific versions
- Regular dependency updates via Dependabot (when enabled)
- Vulnerability alerts monitored

## Best Practices for Users

### API Tokens

- Never commit API tokens (Hugging Face, etc.) to version control
- Use environment variables: `export HUGGINGFACE_TOKEN=your_token`
- Use the `--hf-token` flag only in secure environments

### Audio Files

- Be cautious with audio files from untrusted sources
- The tool processes audio locally; no data is sent to external services

### Model Downloads

- Models are downloaded from official sources (Hugging Face Hub, OpenAI)
- Verify model checksums when possible
