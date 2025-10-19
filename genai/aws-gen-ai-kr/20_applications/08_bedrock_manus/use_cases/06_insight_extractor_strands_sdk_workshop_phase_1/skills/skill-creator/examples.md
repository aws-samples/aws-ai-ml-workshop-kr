# Real-World Skill Examples

This file contains complete examples of skills across different domains. Use these as reference when creating your own skills.

---

## Example 1: API Documentation Writer (Technical Writing)

```markdown
---
name: rest-api-doc-writer
description: Expert guidance for writing comprehensive RESTful API documentation following OpenAPI 3.0 standards
---

# REST API Documentation Writer

This skill provides structured guidance for writing clear, complete REST API documentation. It follows OpenAPI 3.0 standards and industry best practices for developer-facing API docs.

## Core Principles

### Developer-First Mindset
API documentation should enable developers to integrate successfully on their first attempt. Focus on clarity, completeness, and practical examples over technical formality.

### Progressive Disclosure
Provide essential information upfront, with deeper details available for those who need them. Start with quick-start examples, then provide comprehensive reference.

### Consistency
Use consistent terminology, structure, and formatting throughout all endpoints. Predictable patterns reduce cognitive load.

## Documentation Structure

### API Overview Section
**Required elements:**
- Base URL and versioning scheme
- Authentication method (API keys, OAuth, JWT)
- Rate limiting policy
- Common error codes
- Getting started / Quick start guide

### Endpoint Documentation
**For each endpoint, include:**

1. **HTTP Method and Path**
   - `GET /api/v1/users/{id}`
   - Use path parameters in curly braces

2. **Short Description**
   - One sentence explaining what this endpoint does
   - Example: "Retrieves a single user by their unique ID"

3. **Authentication**
   - Required auth level (public, API key, OAuth scope)

4. **Parameters**
   - Path parameters
   - Query parameters
   - Request body (for POST/PUT/PATCH)

   **Table format:**
   | Name | Type | Required | Description |
   |------|------|----------|-------------|

5. **Request Example**
   - Actual HTTP request with headers
   - Complete JSON body for POST/PUT

6. **Response Examples**
   - Success response (200, 201, 204)
   - Complete JSON with realistic data
   - Include response codes

7. **Error Responses**
   - Common error codes (400, 401, 403, 404, 500)
   - Error response format
   - When each error occurs

## Guidelines

### Writing Descriptions
- **Be specific:** "Creates a new user account" not "Handles users"
- **Use action verbs:** "Retrieves", "Creates", "Updates", "Deletes"
- **Explain the outcome:** What happens when this endpoint succeeds?

### Parameter Documentation
- **Type specification:** Use OpenAPI types (string, integer, boolean, array, object)
- **Validation rules:** Min/max length, allowed values, regex patterns
- **Default values:** State explicitly if parameter has a default
- **Examples:** Show realistic example values

### Example Data Quality
- **Use realistic examples:** "Alice Smith" not "Test User 123"
- **Show actual IDs:** "usr_7f3b9c2e" not "string"
- **Include timestamps:** ISO 8601 format
- **Demonstrate relationships:** Show how related endpoints connect

### Error Documentation
**Standard error response format:**
```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "User with ID usr_123 not found",
    "details": {
      "resource": "user",
      "id": "usr_123"
    }
  }
}
```

**Document when each error occurs:**
- **400 Bad Request:** Invalid input format or validation failure
- **401 Unauthorized:** Missing or invalid authentication
- **403 Forbidden:** Valid auth but insufficient permissions
- **404 Not Found:** Resource doesn't exist
- **429 Too Many Requests:** Rate limit exceeded
- **500 Internal Server Error:** Server-side failure

## Example: Complete Endpoint Documentation

### Get User by ID
**GET** `/api/v1/users/{id}`

Retrieves detailed information about a specific user by their unique identifier.

**Authentication:** API Key required

**Path Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| id | string | Yes | Unique user identifier (format: `usr_[a-z0-9]{8}`) |

**Query Parameters:**
| Name | Type | Required | Description | Default |
|------|------|----------|-------------|---------|
| include | string | No | Comma-separated list of related resources to include (`profile`, `preferences`) | None |

**Example Request:**
```http
GET /api/v1/users/usr_7f3b9c2e?include=profile,preferences HTTP/1.1
Host: api.example.com
Authorization: Bearer sk_live_abc123xyz789
Accept: application/json
```

**Success Response (200 OK):**
```json
{
  "id": "usr_7f3b9c2e",
  "email": "alice@example.com",
  "name": "Alice Smith",
  "role": "admin",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-10-19T14:22:00Z",
  "profile": {
    "avatar_url": "https://cdn.example.com/avatars/usr_7f3b9c2e.jpg",
    "bio": "Product manager and API enthusiast",
    "location": "San Francisco, CA"
  },
  "preferences": {
    "notifications_enabled": true,
    "timezone": "America/Los_Angeles",
    "language": "en-US"
  }
}
```

**Error Responses:**

**401 Unauthorized:**
```json
{
  "error": {
    "code": "INVALID_API_KEY",
    "message": "The provided API key is invalid or expired"
  }
}
```

**404 Not Found:**
```json
{
  "error": {
    "code": "USER_NOT_FOUND",
    "message": "User with ID usr_7f3b9c2e not found",
    "details": {
      "resource": "user",
      "id": "usr_7f3b9c2e"
    }
  }
}
```

---

### Create User
**POST** `/api/v1/users`

Creates a new user account with the specified details.

**Authentication:** Admin API Key required

**Request Body:**
| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| email | string | Yes | User's email address | Valid email format, must be unique |
| name | string | Yes | Full name | 2-100 characters |
| role | string | No | User role | One of: `user`, `admin`, `viewer`. Default: `user` |
| send_welcome | boolean | No | Send welcome email | Default: `true` |

**Example Request:**
```http
POST /api/v1/users HTTP/1.1
Host: api.example.com
Authorization: Bearer sk_live_abc123xyz789
Content-Type: application/json

{
  "email": "bob@example.com",
  "name": "Bob Johnson",
  "role": "user",
  "send_welcome": true
}
```

**Success Response (201 Created):**
```json
{
  "id": "usr_9a4c6d1f",
  "email": "bob@example.com",
  "name": "Bob Johnson",
  "role": "user",
  "status": "active",
  "created_at": "2025-10-19T15:30:00Z",
  "updated_at": "2025-10-19T15:30:00Z"
}
```

**Error Responses:**

**400 Bad Request (Validation Failure):**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "fields": {
        "email": "Email address is already registered",
        "name": "Name must be between 2 and 100 characters"
      }
    }
  }
}
```

**403 Forbidden:**
```json
{
  "error": {
    "code": "INSUFFICIENT_PERMISSIONS",
    "message": "Admin role required to create users"
  }
}
```

## Anti-Patterns

### ❌ Vague Descriptions
**Bad:**
```
GET /api/users/{id}
Gets a user
```

**Good:**
```
GET /api/v1/users/{id}
Retrieves detailed information about a specific user by their unique identifier
```

### ❌ Missing Error Documentation
**Bad:**
Only documenting the success case

**Good:**
Document all common error responses (401, 403, 404, 400, 500) with examples

### ❌ Unrealistic Examples
**Bad:**
```json
{
  "id": "string",
  "email": "string",
  "name": "string"
}
```

**Good:**
```json
{
  "id": "usr_7f3b9c2e",
  "email": "alice@example.com",
  "name": "Alice Smith"
}
```

### ❌ Incomplete Parameter Docs
**Bad:**
| Name | Type | Description |
|------|------|-------------|
| email | string | Email |

**Good:**
| Name | Type | Required | Description | Constraints |
|------|------|----------|-------------|-------------|
| email | string | Yes | User's email address | Valid email format, must be unique, max 255 chars |

## API Documentation Checklist

Before publishing API docs, verify:

**Overview Section:**
- [ ] Base URL documented
- [ ] Versioning strategy explained
- [ ] Authentication method specified with examples
- [ ] Rate limits documented
- [ ] Quick start guide included

**For Each Endpoint:**
- [ ] HTTP method and full path with version
- [ ] One-sentence description
- [ ] Authentication requirements specified
- [ ] All parameters documented with type, required/optional, description
- [ ] At least one request example with headers and body
- [ ] Success response example with realistic data
- [ ] Common error responses documented (400, 401, 403, 404)
- [ ] Response codes explicitly stated

**Overall Quality:**
- [ ] Consistent terminology throughout
- [ ] Realistic example data (not "string" or "123")
- [ ] All timestamps in ISO 8601 format
- [ ] Related endpoints cross-referenced
- [ ] No undocumented breaking changes from previous version
```

---

## Example 2: Python Security Code Reviewer (Review Skill)

```markdown
---
name: python-security-reviewer
description: Expert guidance for conducting security-focused code reviews of Python web applications, covering OWASP Top 10 vulnerabilities
---

# Python Security Code Reviewer

This skill provides systematic guidance for identifying security vulnerabilities in Python web applications. It focuses on common security issues (OWASP Top 10), secure coding patterns, and actionable remediation recommendations.

## Core Security Principles

### Defense in Depth
Never rely on a single security control. Layer multiple protections (input validation + parameterized queries + least privilege) so that if one fails, others provide backup.

### Fail Securely
When errors occur, fail in a secure state. Don't expose stack traces to users, don't grant access on auth failures, don't log passwords.

### Principle of Least Privilege
Grant minimum necessary permissions. Database users shouldn't have DROP rights. API tokens should have scoped permissions. Sessions should expire.

### Never Trust User Input
All user-provided data is potentially malicious. Validate, sanitize, and use safely. This includes URLs, file uploads, query parameters, headers, and cookies.

## Security Review Checklist

### 1. Injection Vulnerabilities

**SQL Injection:**
- [ ] All database queries use parameterized statements or ORM methods
- [ ] No string concatenation or f-strings in SQL queries
- [ ] User input is never directly interpolated into raw SQL

**Command Injection:**
- [ ] Avoid `os.system()`, `subprocess.shell=True`, `eval()`, `exec()`
- [ ] If shell commands necessary, use allowlist validation and `shlex.quote()`
- [ ] User input never passed directly to system commands

**Code Injection:**
- [ ] No use of `eval()` or `exec()` with user input
- [ ] Template engines configured to auto-escape (Jinja2 `autoescape=True`)
- [ ] Pickle/YAML deserialization only for trusted data

### 2. Authentication & Session Management

**Password Security:**
- [ ] Passwords hashed with bcrypt, Argon2, or PBKDF2 (not MD5/SHA1)
- [ ] Salt is automatically handled by the hashing library
- [ ] Password complexity requirements enforced
- [ ] Passwords never logged or stored in plain text

**Session Security:**
- [ ] Session tokens are cryptographically random (use `secrets` module)
- [ ] Session cookies have `HttpOnly`, `Secure`, and `SameSite` flags
- [ ] Sessions expire after inactivity and have absolute timeout
- [ ] Session IDs regenerated after login (prevent session fixation)

**Authentication:**
- [ ] Account lockout after failed login attempts
- [ ] No user enumeration (same error for invalid user vs. wrong password)
- [ ] MFA available for sensitive operations
- [ ] Password reset tokens expire and are single-use

### 3. Access Control

**Authorization:**
- [ ] Authorization checks on every protected endpoint/function
- [ ] User can only access their own resources (check ownership)
- [ ] Role-based access control properly enforced
- [ ] No reliance on client-side checks or hidden URLs for security

**Insecure Direct Object References:**
- [ ] Don't expose internal IDs if possible (use UUIDs or slugs)
- [ ] Always verify user has permission to access requested resource
- [ ] Can user A modify user B's data? (test for horizontal privilege escalation)
- [ ] Can regular user access admin functions? (test for vertical escalation)

### 4. Sensitive Data Exposure

**Data Protection:**
- [ ] Sensitive data encrypted at rest (database encryption, encrypted volumes)
- [ ] HTTPS enforced for all traffic (no mixed content)
- [ ] Sensitive data not logged (passwords, tokens, credit cards, SSNs)
- [ ] Secrets stored in environment variables or secret management, not code

**Information Disclosure:**
- [ ] Error messages don't reveal stack traces, file paths, or SQL queries
- [ ] Debug mode disabled in production
- [ ] No sensitive data in URLs (use POST body instead)
- [ ] API responses don't leak internal structure or user information

### 5. Security Misconfiguration

**Framework & Library Security:**
- [ ] Dependencies up-to-date (use `pip-audit` or `safety`)
- [ ] No known vulnerabilities in packages (check CVE databases)
- [ ] Remove unused dependencies
- [ ] Pin dependency versions to avoid supply chain attacks

**Server Configuration:**
- [ ] CORS configured restrictively (not `Access-Control-Allow-Origin: *`)
- [ ] Security headers set (CSP, X-Frame-Options, X-Content-Type-Options)
- [ ] Default accounts and passwords changed
- [ ] Directory listing disabled

### 6. Cross-Site Scripting (XSS)

- [ ] Template auto-escaping enabled (Jinja2, Django templates)
- [ ] User input in HTML contexts is escaped
- [ ] User input in JavaScript contexts is JSON-encoded
- [ ] Content-Security-Policy header implemented
- [ ] No use of `dangerouslySetInnerHTML` equivalents without sanitization

### 7. File Upload Security

- [ ] File type validated by content (magic bytes), not just extension
- [ ] File size limits enforced
- [ ] Uploaded files stored outside webroot
- [ ] Uploaded files served with `Content-Disposition: attachment`
- [ ] Image uploads re-encoded to strip malicious metadata

## Common Vulnerability Patterns

### Pattern 1: SQL Injection

**❌ Vulnerable:**
```python
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
```
**Attack:** `user_id = "1 OR 1=1"` returns all users

**✅ Secure:**
```python
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = ?"
    cursor.execute(query, (user_id,))
```

### Pattern 2: Weak Password Hashing

**❌ Vulnerable:**
```python
import hashlib

password = request.form['password']
hashed = hashlib.md5(password.encode()).hexdigest()
user.password = hashed
```
**Issue:** MD5 is fast and easily cracked with rainbow tables

**✅ Secure:**
```python
from werkzeug.security import generate_password_hash, check_password_hash

password = request.form['password']
user.password_hash = generate_password_hash(password)

# Later, for login:
if check_password_hash(user.password_hash, submitted_password):
    # Valid password
```

### Pattern 3: Missing Authorization Check

**❌ Vulnerable:**
```python
@app.route('/api/users/<user_id>/profile', methods=['PUT'])
def update_profile(user_id):
    # No check if current user owns this profile!
    user = User.query.get(user_id)
    user.bio = request.json['bio']
    db.session.commit()
```
**Attack:** User A can update User B's profile

**✅ Secure:**
```python
@app.route('/api/users/<user_id>/profile', methods=['PUT'])
@login_required
def update_profile(user_id):
    if current_user.id != user_id and not current_user.is_admin:
        abort(403)  # Forbidden

    user = User.query.get_or_404(user_id)
    user.bio = request.json['bio']
    db.session.commit()
```

### Pattern 4: Command Injection

**❌ Vulnerable:**
```python
import os

filename = request.args.get('file')
os.system(f'cat /var/logs/{filename}')
```
**Attack:** `file=../../etc/passwd` or `file=logs; rm -rf /`

**✅ Secure:**
```python
import subprocess
import shlex
import os

filename = request.args.get('file')

# Allowlist validation
if not filename.isalnum():
    abort(400, "Invalid filename")

# Use array form (no shell)
subprocess.run(['cat', f'/var/logs/{filename}'], check=True)
```

## Security Review Report Template

```markdown
## Security Review: [Project Name]

**Reviewer:** [Your Name]
**Date:** [YYYY-MM-DD]
**Scope:** [Files/modules reviewed]

### Summary
[High-level overview of security posture: Critical: X, High: Y, Medium: Z]

### Critical Issues

#### 1. [Vulnerability Type]
**Severity:** Critical
**Location:** `app/routes.py:45-48`
**Description:** User input directly concatenated into SQL query without parameterization
**Attack Scenario:** Attacker can inject SQL to read/modify database or escalate privileges
**Code:**
\```python
query = f"SELECT * FROM users WHERE email = '{email}'"
\```
**Recommendation:** Use parameterized queries
**Fix:**
\```python
query = "SELECT * FROM users WHERE email = ?"
cursor.execute(query, (email,))
\```
**Reference:** https://owasp.org/www-community/attacks/SQL_Injection

### High Priority Issues
[Similar format]

### Medium Priority Issues
[Similar format]

### Positive Security Practices
- [Note good practices observed]

### Overall Recommendations
1. [General recommendation 1]
2. [General recommendation 2]
```

## Validation Checklist

Security review is complete when:
- [ ] All files in scope have been reviewed
- [ ] Each OWASP Top 10 category checked
- [ ] Findings documented with severity, location, and fix
- [ ] Code examples provided for vulnerabilities found
- [ ] Remediation recommendations are specific and actionable
- [ ] References to OWASP or CVE included where applicable
```

---

## Example 3: Data Visualization Designer (Analysis Skill)

```markdown
---
name: data-viz-designer
description: Expert guidance for creating effective, accurate, and accessible data visualizations using Python (matplotlib, seaborn, plotly)
---

# Data Visualization Designer

This skill provides principles and patterns for creating clear, accurate, and insightful data visualizations. It covers chart selection, design best practices, and implementation guidance for Python visualization libraries.

## Core Visualization Principles

### Accuracy First
Never distort data to make it look more dramatic. Use appropriate scales, start axes at zero (for bar charts), and represent quantities honestly.

### Clarity Over Decoration
Every element should serve a purpose. Remove chart junk (excessive gridlines, 3D effects, decorative elements) that doesn't add information.

### Accessibility
Visualizations should be understandable by all users, including those with color blindness or using screen readers. Use patterns in addition to colors, add alt text, ensure sufficient contrast.

### Tell a Story
Guide the viewer to the insight. Use titles, annotations, and highlighting to direct attention to what matters.

## Chart Selection Guide

### Comparing Categories: Bar Chart
**Use when:** Comparing discrete categories (sales by region, survey responses)
**Best practices:**
- Horizontal bars for long category names
- Start y-axis at zero
- Sort by value (highest to lowest) unless natural order exists
- Use consistent colors unless highlighting specific categories

### Showing Trends Over Time: Line Chart
**Use when:** Displaying continuous data over time (stock prices, temperature)
**Best practices:**
- Time always on x-axis
- Use direct labels instead of legend when possible
- Highlight recent or important periods
- Avoid more than 5-7 lines (gets cluttered)

### Showing Distribution: Histogram or Box Plot
**Use when:** Understanding data distribution (age ranges, response times)
**Histogram:** Shows full shape of distribution
**Box plot:** Shows median, quartiles, and outliers
**Best practices:**
- Choose appropriate bin size for histograms
- Annotate median and mean if both are important
- Show sample size

### Showing Relationships: Scatter Plot
**Use when:** Exploring correlation between two variables
**Best practices:**
- Add trend line if relationship exists
- Use point size or color for third dimension
- Label outliers or interesting points
- Include correlation coefficient if relevant

### Showing Composition: Stacked Bar or Pie Chart
**Pie chart:** Only for simple proportions (2-5 slices), part-to-whole relationships
**Stacked bar:** Better for multiple categories or time series
**Best practices:**
- Pie: Start at 12 o'clock, order slices by size
- Stacked bar: Put most important category at bottom
- Use 100% stacked for showing proportions over time
- Never use 3D pie charts

## Design Best Practices

### Color Usage
- **Qualitative palettes:** For distinct categories (no inherent order)
- **Sequential palettes:** For ordered data (low to high)
- **Diverging palettes:** For data with meaningful midpoint (negative to positive)
- **Avoid:** Red/green combinations (colorblind unfriendly)
- **Use:** Colorblind-safe palettes (viridis, colorbrewer)

### Typography
- **Title:** Clear, descriptive (not just "Chart")
- **Axes labels:** Include units (e.g., "Revenue (millions USD)")
- **Font size:** Readable (minimum 10pt for printed, 12pt for screens)
- **Font choice:** Simple, sans-serif (Arial, Helvetica, Open Sans)

### Annotations
- **Use sparingly:** Only for key insights
- **Direct labeling:** Label lines/bars directly instead of using legend
- **Context:** Add reference lines (average, target, previous period)
- **Explanation:** Brief note explaining anomalies or interesting patterns

## Python Implementation Patterns

### Pattern 1: Clean Bar Chart (matplotlib)

```python
import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Product A', 'Product B', 'Product C', 'Product D']
values = [45, 72, 38, 61]

# Create figure with specific size
fig, ax = plt.subplots(figsize=(10, 6))

# Horizontal bar chart (better for text labels)
bars = ax.barh(categories, values, color='steelblue')

# Highlight the top performer
bars[1].set_color('coral')

# Add value labels
for i, v in enumerate(values):
    ax.text(v + 1, i, f'{v}', va='center')

# Formatting
ax.set_xlabel('Sales (thousands USD)', fontsize=12)
ax.set_title('Q4 Sales by Product', fontsize=14, fontweight='bold', pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('sales_by_product.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Pattern 2: Time Series (seaborn)

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Seaborn styling
sns.set_style("whitegrid")
sns.set_palette("husl")

# Data
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=365, freq='D'),
    'visitors': np.random.randint(1000, 5000, 365)
})

# Create plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df, x='date', y='visitors', ax=ax, linewidth=2)

# Add reference line (average)
avg = df['visitors'].mean()
ax.axhline(avg, color='red', linestyle='--', alpha=0.7, label=f'Average: {avg:.0f}')

# Formatting
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Daily Visitors', fontsize=12)
ax.set_title('Website Traffic - 2024', fontsize=14, fontweight='bold')
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig('traffic_2024.png', dpi=300)
plt.show()
```

### Pattern 3: Distribution Comparison (seaborn)

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Create violin plot with box plot overlay
fig, ax = plt.subplots(figsize=(10, 6))

sns.violinplot(data=df, x='category', y='value', inner=None, ax=ax, alpha=0.6)
sns.boxplot(data=df, x='category', y='value', width=0.3, ax=ax,
            boxprops=dict(alpha=0.7), palette='Set2')

# Formatting
ax.set_xlabel('Category', fontsize=12)
ax.set_ylabel('Response Time (ms)', fontsize=12)
ax.set_title('Response Time Distribution by Category', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('distribution_comparison.png', dpi=300)
plt.show()
```

## Anti-Patterns

### ❌ Truncated Y-Axis (Deceptive)
```python
# Makes small differences look huge
ax.set_ylim(980, 1020)  # When data ranges from 990-1010
```
**Fix:** Start at zero for bar charts, or clearly label truncation

### ❌ Too Many Categories
```python
# 30 colors on one chart - unreadable
categories = [f'Category {i}' for i in range(30)]
```
**Fix:** Group into "Other", use faceted plots, or filter top N

### ❌ Rainbow Color Scheme (Not Colorblind Safe)
```python
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
```
**Fix:** Use colorblind-safe palettes
```python
import seaborn as sns
colors = sns.color_palette('colorblind', n_colors=6)
```

### ❌ No Context (Just Raw Data)
```python
plt.title("Chart")
plt.xlabel("X")
plt.ylabel("Y")
```
**Fix:** Descriptive titles and labels with units
```python
plt.title("Monthly Revenue Growth - Q4 2024", fontsize=14, fontweight='bold')
plt.xlabel("Month")
plt.ylabel("Revenue (USD thousands)")
```

## Visualization Checklist

Before finalizing a visualization, verify:

**Content:**
- [ ] Chart type appropriate for the data and message
- [ ] Data accurately represented (no distortion)
- [ ] Key insight is immediately apparent
- [ ] Outliers or anomalies explained

**Design:**
- [ ] Title is clear and descriptive
- [ ] Axis labels include units
- [ ] Font sizes readable (min 10pt)
- [ ] Colors are colorblind-safe
- [ ] Legend is clear or direct labels used
- [ ] No chart junk (unnecessary gridlines, 3D, decorations)

**Accessibility:**
- [ ] Sufficient color contrast (not relying on color alone)
- [ ] Alt text provided for digital visualizations
- [ ] Patterns or textures used in addition to colors

**Technical:**
- [ ] High resolution (300 DPI for print, 150 for web)
- [ ] Proper file format (PNG for web, PDF for print)
- [ ] Aspect ratio appropriate (not stretched)
```

---

## Example 4: Test Case Generator (Code Skill)

```markdown
---
name: pytest-test-generator
description: Expert guidance for writing comprehensive pytest test cases following best practices for unit, integration, and fixture usage
---

# Pytest Test Case Generator

This skill provides structured guidance for writing effective Python tests using pytest. It covers test organization, fixture usage, assertion techniques, and best practices for maintainable test suites.

## Core Testing Principles

### Test One Thing
Each test should verify a single behavior. If a test name requires "and", it's probably testing too much.

### Tests Should Be FIRST
- **Fast:** Run quickly (mock external dependencies)
- **Independent:** No dependencies between tests
- **Repeatable:** Same result every time
- **Self-validating:** Pass or fail, no manual checking
- **Timely:** Written alongside code, not later

### Arrange-Act-Assert (AAA)
```python
def test_user_creation():
    # Arrange: Set up test data
    email = "test@example.com"
    name = "Test User"

    # Act: Execute the behavior
    user = create_user(email=email, name=name)

    # Assert: Verify the outcome
    assert user.email == email
    assert user.name == name
    assert user.id is not None
```

## Test Organization

### Directory Structure
```
project/
├── src/
│   └── myapp/
│       ├── __init__.py
│       ├── models.py
│       └── services.py
└── tests/
    ├── __init__.py
    ├── conftest.py          # Shared fixtures
    ├── unit/                # Fast, isolated tests
    │   ├── test_models.py
    │   └── test_services.py
    └── integration/         # Tests with dependencies
        └── test_api.py
```

### Naming Conventions
- **Files:** `test_*.py` or `*_test.py`
- **Functions:** `test_<what>_<condition>_<expected>`
- **Classes:** `Test<Component>` (optional grouping)

**Examples:**
```python
def test_create_user_with_valid_email_returns_user()
def test_create_user_with_duplicate_email_raises_error()
def test_get_user_when_not_found_returns_none()
```

## Fixture Patterns

### Basic Fixture
```python
import pytest

@pytest.fixture
def sample_user():
    """Provides a basic user instance for testing."""
    return User(
        email="test@example.com",
        name="Test User",
        role="user"
    )

def test_user_has_default_role(sample_user):
    assert sample_user.role == "user"
```

### Fixture with Teardown
```python
@pytest.fixture
def database():
    """Provides a test database connection with cleanup."""
    db = Database(':memory:')
    db.create_tables()

    yield db  # Provide the fixture

    db.close()  # Cleanup after test
```

### Parametrized Fixture
```python
@pytest.fixture(params=['sqlite', 'postgresql', 'mysql'])
def database(request):
    """Test against multiple database types."""
    db_type = request.param
    db = Database(db_type, ':memory:')
    db.create_tables()

    yield db

    db.close()
```

### Fixture Scopes
```python
@pytest.fixture(scope="session")  # Once per test session
def app_config():
    return load_config()

@pytest.fixture(scope="module")   # Once per test file
def database_connection():
    return connect_db()

@pytest.fixture(scope="function")  # Once per test (default)
def temp_file():
    return create_temp_file()
```

## Assertion Techniques

### Basic Assertions
```python
# Equality
assert actual == expected

# Identity
assert obj is not None

# Membership
assert 'key' in dictionary
assert item in list

# Exceptions
with pytest.raises(ValueError):
    invalid_operation()

# Exception message
with pytest.raises(ValueError, match="Invalid email"):
    create_user(email="not-an-email")
```

### Pytest Assertions (Better Error Messages)
```python
# Instead of assertTrue/assertFalse
assert user.is_active  # pytest shows actual value on failure

# Approximate equality for floats
assert result == pytest.approx(0.1 + 0.2)

# Custom failure messages
assert len(users) > 0, f"Expected users but got empty list"
```

## Test Patterns

### Pattern 1: Unit Test with Mock
```python
from unittest.mock import Mock, patch

def test_send_email_calls_smtp_server(sample_user):
    # Arrange
    mock_smtp = Mock()

    # Act
    with patch('smtplib.SMTP', return_value=mock_smtp):
        send_welcome_email(sample_user)

    # Assert
    mock_smtp.sendmail.assert_called_once()
    args = mock_smtp.sendmail.call_args[0]
    assert sample_user.email in args[1]  # Recipient
```

### Pattern 2: Parametrized Tests
```python
@pytest.mark.parametrize("email,expected_valid", [
    ("valid@example.com", True),
    ("also.valid+tag@example.co.uk", True),
    ("invalid@", False),
    ("@invalid.com", False),
    ("no-at-sign.com", False),
])
def test_email_validation(email, expected_valid):
    result = is_valid_email(email)
    assert result == expected_valid
```

### Pattern 3: Integration Test with Database
```python
@pytest.fixture
def db_session():
    """Provides a database session with rollback."""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.rollback()
    session.close()

def test_user_repository_save(db_session):
    # Arrange
    user = User(email="test@example.com", name="Test")
    repo = UserRepository(db_session)

    # Act
    saved_user = repo.save(user)

    # Assert
    assert saved_user.id is not None
    retrieved = repo.get_by_id(saved_user.id)
    assert retrieved.email == "test@example.com"
```

## Anti-Patterns

### ❌ Testing Implementation Details
```python
# Bad: Tests how it works, not what it does
def test_user_repository_uses_sql_query():
    assert "SELECT * FROM users" in repo.get_all_query()
```
**Fix:** Test behavior, not implementation

### ❌ Interdependent Tests
```python
# Bad: Test B depends on Test A running first
def test_create_user():
    global user_id
    user_id = create_user()

def test_get_user():
    user = get_user(user_id)  # Breaks if test_create_user fails
```
**Fix:** Each test should be independent

### ❌ Unclear Test Names
```python
def test_user():  # What about users?
def test_case_1():  # Meaningless
```
**Fix:** Descriptive names explaining what and why

## Test Checklist

- [ ] All critical functionality has tests
- [ ] Tests are fast (< 1 second for unit tests)
- [ ] Tests are independent (can run in any order)
- [ ] Test names clearly describe what's being tested
- [ ] Fixtures used for common setup
- [ ] External dependencies mocked
- [ ] Edge cases covered (empty input, null, large values)
- [ ] Error cases tested (exceptions, validation failures)
- [ ] Tests follow AAA pattern
```

---

## Summary

These examples demonstrate skills across different domains:

1. **API Documentation Writer**: Technical writing skill with structure and templates
2. **Python Security Reviewer**: Review skill with checklists and vulnerability patterns
3. **Data Visualization Designer**: Analysis skill with design principles and code examples
4. **Pytest Test Generator**: Code skill with patterns and best practices

**Key Patterns Across All Examples:**
- Clear scope and purpose
- Core principles/philosophy section
- Actionable guidelines
- Concrete, realistic examples
- Anti-patterns showing what to avoid
- Validation checklist
- Domain-specific expertise

Use these as templates when creating your own skills for different domains.
