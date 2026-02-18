# Security Hardening Guide for Adhan

## Overview

This document outlines security measures implemented and additional hardening recommendations.

---

## ✅ Implemented Security Measures

### 1. Input Validation

**scraper_minimal.py**:
- ✅ URL whitelist - Only allowed domains can be scraped
- ✅ Filename validation - Regex check for safe filenames
- ✅ Size limits - Text truncated to 1000 chars max
- ✅ Type checking - Validates data structures before use

**train_minimal.py**:
- ✅ Config file size limit (1MB max)
- ✅ File extension validation (only .json, .yaml, .yml)
- ✅ Symlink detection and rejection
- ✅ Parameter bounds checking (epochs, batch size, etc.)
- ✅ Dataset size limits (max 10,000 samples)

### 2. Resource Limits

```python
# Training limits
MAX_LENGTH = 512          # Prevent memory exhaustion
MAX_EPOCHS = 10           # Prevent runaway training
MAX_BATCH_SIZE = 32       # Prevent OOM
MAX_WORKERS = 4           # Limit concurrent processes
MAX_CHECKPOINTS = 2       # Limit disk usage

# Data limits
MAX_TEXT_SIZE = 1000      # Per record
MAX_DATASET_SIZE = 10000  # Total records
MAX_CONFIG_SIZE = 1MB     # Config file
```

### 3. Network Security

**HTTPS Only**:
- All requests use HTTPS (no HTTP fallback)
- SSL verification enabled by default

**Retry Logic**:
```python
Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
```

**Timeouts**:
- Default: 10 seconds
- Prevents hanging requests

**Domain Whitelist**:
```python
ALLOWED_DOMAINS = {'ta.wikipedia.org', 'api.wikimedia.org'}
```

### 4. Data Security

**No Secrets in Code**:
- ✅ No API keys or credentials in source
- ✅ No hardcoded passwords
- ✅ Use environment variables if needed

**Safe File Operations**:
- ✅ Path validation with `pathlib`
- ✅ No arbitrary path access
- ✅ Secure file permissions (0644)

**Logging Security**:
- ✅ No sensitive data in logs
- ✅ Error messages sanitized
- ✅ PII not logged

---

## 🔒 Additional Hardening Recommendations

### 1. Environment Security

```bash
# Run in isolated environment
python3 -m venv .venv
source .venv/bin/activate

# Use requirements.txt with exact versions
pip install -r requirements.txt --no-cache-dir

# Set restrictive permissions
chmod 700 .venv
chmod 644 src/**/*.py
```

### 2. Runtime Security

**Use Sandboxing**:
```bash
# Run with limited permissions (Linux)
python3 -B -S -s scraper_minimal.py  # -B: no .pyc, -S: no site, -s: no user site

# Or use containers
docker run --read-only --tmpfs /tmp ...
```

**Resource Limits** (Linux):
```bash
# Limit memory
ulimit -m 4000000  # 4GB

# Limit CPU time
ulimit -t 3600  # 1 hour

# Limit processes
ulimit -u 50
```

### 3. Dependency Security

**Pin Dependencies**:
```txt
# requirements.txt with exact versions
transformers==4.35.2
torch==2.1.2
datasets==2.14.6
requests==2.31.0
```

**Audit Dependencies**:
```bash
# Check for vulnerabilities
pip install safety
safety check

# Or use GitHub Dependabot (automatic)
```

### 4. Code Security

**Static Analysis**:
```bash
# Run bandit for security issues
pip install bandit
bandit -r src/

# Run pylint for code quality
pip install pylint
pylint src/
```

**Type Checking**:
```bash
# Run mypy for type safety
pip install mypy
mypy src/
```

---

## 🛡️ Security Best Practices

### DO ✅

1. **Always validate input**
   - Check file sizes before reading
   - Validate URLs against whitelist
   - Sanitize user input

2. **Use secure defaults**
   - HTTPS only
   - Timeouts on all network calls
   - Limited retries

3. **Implement rate limiting**
   - Prevent abuse
   - Respect server resources

4. **Log security events**
   - Failed validations
   - Rejected requests
   - Resource limit hits

5. **Keep dependencies updated**
   - Regular security patches
   - Use Dependabot or similar

### DON'T ❌

1. **Never trust user input**
   - Always validate
   - Always sanitize
   - Always limit

2. **Never hardcode secrets**
   - Use environment variables
   - Use secret management tools
   - Never commit secrets

3. **Never run as root**
   - Use least privilege
   - Drop privileges early

4. **Never ignore errors**
   - Log all errors
   - Handle gracefully
   - Don't expose internals

5. **Never use HTTP**
   - Always HTTPS
   - Always verify SSL

---

## 🔍 Security Checklist

Use this checklist before deployment:

- [ ] All dependencies pinned to exact versions
- [ ] `safety check` passes with 0 vulnerabilities
- [ ] `bandit` scan passes with 0 high/medium issues
- [ ] CodeQL analysis passes
- [ ] No secrets in code or logs
- [ ] All file paths validated
- [ ] All network requests have timeouts
- [ ] Resource limits configured
- [ ] Rate limiting enabled
- [ ] Error handling comprehensive
- [ ] Logging configured (no PII)
- [ ] Running in virtualenv/container
- [ ] File permissions set correctly (644/755)
- [ ] HTTPS only (no HTTP fallback)
- [ ] Input validation on all user data

---

## 🚨 Incident Response

If a security issue is discovered:

1. **Isolate**: Stop affected systems
2. **Assess**: Determine impact and scope
3. **Contain**: Prevent further damage
4. **Eradicate**: Remove vulnerability
5. **Recover**: Restore normal operations
6. **Document**: Record findings and fixes

---

## 📚 Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)
- [Bandit Security Scanner](https://bandit.readthedocs.io/)
- [Safety DB](https://pyup.io/safety/)
- [GitHub Security Advisories](https://github.com/advisories)

---

## 🔐 Security Contact

For security issues:
- **DO NOT** open public GitHub issues
- Email security team privately
- Use responsible disclosure practices

---

**Last Updated**: 2026-02-18  
**Security Level**: HARDENED  
**Status**: PRODUCTION-READY
