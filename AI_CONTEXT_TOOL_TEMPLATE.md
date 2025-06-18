# AI Context Tool - Quick Reference Template

## Tool Name
`mcp__zen__context`

## Operations

### 1. ADD - Store Knowledge
```
Tool: mcp__zen__context
Parameters:
{
  "operation": "add",
  "content": "[Knowledge/insight to remember]",
  "model": "[model-name]",
  "metadata": {
    "category": "[architecture|implementation|bug-fix|decision|optimization|etc]",
    "tags": ["tag1", "tag2", "tag3"],
    "importance": 0.8,
    "tool_source": "[optional-source-tool]",
    "related_files": ["file1.py", "file2.js"]
  },
  "project_id": "[optional-project-name]"
}
```

### 2. SEARCH - Find Knowledge
```
Tool: mcp__zen__context
Parameters:
{
  "operation": "search",
  "content": "[search keywords]",
  "model": "[model-name]",
  "limit": 10,
  "project_id": "[optional-project-name]"
}
```

### 3. LIST - Show Recent Entries
```
Tool: mcp__zen__context
Parameters:
{
  "operation": "list",
  "model": "[model-name]",
  "limit": 10,
  "project_id": "[optional-project-name]"
}
```

### 4. EXPORT - Backup Knowledge
```
Tool: mcp__zen__context
Parameters:
{
  "operation": "export",
  "model": "[model-name]",
  "project_id": "[optional-project-name]"
}
```

## Common Use Cases

### After solving a complex bug
```
{
  "operation": "add",
  "content": "Fixed authentication issue where JWT tokens were expiring prematurely. Root cause: timezone mismatch between server and client. Solution: Always use UTC timestamps in auth.js:generateToken()",
  "model": "flash",
  "metadata": {
    "category": "bug-fix",
    "tags": ["auth", "jwt", "timezone", "security"],
    "importance": 0.9,
    "related_files": ["auth.js", "middleware/auth.js"]
  }
}
```

### Architectural decision
```
{
  "operation": "add",
  "content": "Decided to use Redis for session management instead of in-memory storage. Benefits: horizontal scaling, session persistence across restarts, built-in TTL support. Implementation in session.js using redis-client library.",
  "model": "flash",
  "metadata": {
    "category": "architecture",
    "tags": ["redis", "session", "scaling", "infrastructure"],
    "importance": 0.85,
    "related_files": ["session.js", "config/redis.js"]
  }
}
```

### Performance optimization
```
{
  "operation": "add",
  "content": "Reduced API response time by 60% through database query optimization. Changed from N+1 queries to single JOIN query in UserService.getWithRelations(). Added composite index on (user_id, created_at).",
  "model": "flash",
  "metadata": {
    "category": "optimization",
    "tags": ["performance", "database", "sql", "api"],
    "importance": 0.8,
    "related_files": ["services/UserService.js", "migrations/add_user_index.sql"]
  }
}
```

### Project convention
```
{
  "operation": "add",
  "content": "Project uses camelCase for JavaScript variables/functions, PascalCase for React components, and snake_case for database columns. ESLint configured to enforce these conventions.",
  "model": "flash",
  "metadata": {
    "category": "convention",
    "tags": ["coding-standards", "eslint", "naming"],
    "importance": 0.6,
    "related_files": [".eslintrc.js"]
  }
}
```

## Best Practices

1. **Always include WHY**: Don't just document what was done, explain why
2. **Be specific**: Include function names, line numbers, error messages
3. **Use consistent tags**: Maintain a standard set of tags across the project
4. **Set appropriate importance**: Use 0.8-1.0 for critical knowledge, 0.5-0.7 for useful info
5. **Link related files**: Help future searches by connecting knowledge to code

## Tag Suggestions

### By Category
- **Technology**: `redis`, `docker`, `react`, `nodejs`, `postgres`
- **Type**: `bug-fix`, `optimization`, `refactor`, `feature`, `hotfix`
- **Component**: `auth`, `api`, `database`, `frontend`, `backend`
- **Status**: `todo`, `in-progress`, `completed`, `deprecated`

### By Importance
- **Critical (0.9-1.0)**: Security fixes, data loss prevention, breaking changes
- **High (0.7-0.8)**: Performance improvements, architectural decisions
- **Medium (0.5-0.6)**: Conventions, patterns, helpful tips
- **Low (0.3-0.4)**: Nice-to-know information, minor optimizations

## Quick Copy Templates

### Minimal Add
```
mcp__zen__context: {"operation": "add", "content": "", "model": "flash"}
```

### Search
```
mcp__zen__context: {"operation": "search", "content": "", "model": "flash"}
```

### List Recent
```
mcp__zen__context: {"operation": "list", "model": "flash", "limit": 10}
```

### Export All
```
mcp__zen__context: {"operation": "export", "model": "flash"}
```