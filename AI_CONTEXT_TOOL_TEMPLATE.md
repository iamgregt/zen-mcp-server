# AI Context Tool - Quick Reference Template
*Version 2.0 - Semantic Search Enabled*

## Tool Name
`mcp__zen__context`

## 🔍 Search Capabilities
- **Semantic Search**: AI-powered contextual understanding with vector embeddings
- **Hybrid Search**: Combines semantic similarity with keyword matching
- **Natural Language Queries**: Search using descriptions, not just keywords
- **Intelligent Fallback**: Automatically uses keyword search if vector search fails

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

### 2. SEARCH - Find Knowledge (Semantic + Keyword)
```
Tool: mcp__zen__context
Parameters:
{
  "operation": "search",
  "content": "[natural language query or keywords]",
  "model": "[model-name]",
  "limit": 10,
  "project_id": "[optional-project-name]"
}
```

**Search Query Examples**:
- Natural Language: "How to fix memory leaks in React"
- Conceptual: "authentication best practices"
- Technical: "JWT token validation"
- Keywords: "redis cache optimization"

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

## 🚀 Semantic Search Examples

### Natural Language Search
```
{
  "operation": "search",
  "content": "How do I handle user authentication with JWT tokens?",
  "model": "flash",
  "limit": 5
}
```

### Conceptual Search
```
{
  "operation": "search",
  "content": "performance optimization techniques for databases",
  "model": "flash",
  "limit": 8
}
```

### Problem-Based Search
```
{
  "operation": "search",
  "content": "memory usage issues in Node.js applications",
  "model": "flash",
  "limit": 10
}
```

### Multi-Language Search
```
{
  "operation": "search",
  "content": "comment gérer les erreurs de validation",
  "model": "flash",
  "limit": 5
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

## ⚙️ Setup for Semantic Search

### Enable Vector Search
```bash
# Add to .env file
echo "ENABLE_VECTOR_SEARCH=true" >> .env

# Restart server
./run-server.sh
```

### Check if Semantic Search is Active
Look for these log messages:
- "Vector store initialized successfully"
- "Hybrid search returned X results"
- "ChromaProvider Memory Usage"

## Best Practices

### Content Storage
1. **Always include WHY**: Don't just document what was done, explain why
2. **Be specific**: Include function names, line numbers, error messages
3. **Use consistent tags**: Maintain a standard set of tags across the project
4. **Set appropriate importance**: Use 0.8-1.0 for critical knowledge, 0.5-0.7 for useful info
5. **Link related files**: Help future searches by connecting knowledge to code

### Search Optimization
1. **Use natural language**: "How to fix X" works better than just "X fix"
2. **Be descriptive**: Include context about the problem you're trying to solve
3. **Try different phrasings**: Semantic search understands multiple ways to express concepts
4. **Combine approaches**: Use both conceptual and keyword searches for best results
5. **Check hybrid results**: Look for entries that match both semantically and by keywords

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

## 🔧 Troubleshooting

### Vector Search Not Working
```bash
# Check if enabled
docker exec zen-mcp-server printenv | grep ENABLE_VECTOR_SEARCH

# Should return: ENABLE_VECTOR_SEARCH=true
```

### Search Performance Issues
```bash
# Monitor search metrics
docker exec zen-mcp-server grep "Context Tool Metrics" /tmp/mcp_server.log | tail -5

# Check memory usage
docker exec zen-mcp-server grep "ChromaProvider Memory Usage" /tmp/mcp_server.log | tail -3
```

### "Flash provider not available" Error
- Ensure valid API keys are configured for Gemini or OpenAI
- Keyword extraction uses AI providers for intelligent keyword generation
- Vector search will still work with basic keyword extraction

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

## 🔍 Semantic Search Templates

### Natural Language Search
```
mcp__zen__context: {"operation": "search", "content": "How do I...", "model": "flash"}
```

### Problem-Focused Search
```
mcp__zen__context: {"operation": "search", "content": "issues with [technology/concept]", "model": "flash"}
```

### Conceptual Search
```
mcp__zen__context: {"operation": "search", "content": "[concept] best practices", "model": "flash"}
```

### Multi-Language Search
```
mcp__zen__context: {"operation": "search", "content": "[query in any language]", "model": "flash"}
```

---

## 🆕 What's New in Version 2.0

- **🧠 AI-Powered Search**: Vector embeddings understand context and meaning
- **🔀 Hybrid Algorithm**: Combines semantic similarity with keyword matching
- **🌐 Multilingual Support**: Search in multiple languages with same quality
- **📊 Performance Metrics**: Real-time monitoring of search operations
- **🎯 Smart Fallback**: Graceful degradation when vector search is unavailable
- **💾 Persistent Storage**: Moved from `/tmp` to `/data/kb` for better data protection

*Enable with `ENABLE_VECTOR_SEARCH=true` in your .env file*