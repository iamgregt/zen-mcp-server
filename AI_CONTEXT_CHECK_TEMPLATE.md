# AI Context Check-in Template
*Version 2.0 - Semantic Search Enabled*

## Standard Check-in Prompt
```
Please check the context knowledge base for any relevant information about this [task/feature/issue] before proceeding. Use mcp__zen__context with semantic search to find related knowledge and insights.
```

## 🔍 Semantic Search Prompts

### Natural Language Search
```
Search the knowledge base using natural language: "How do I handle [specific problem]?" or "What are the best practices for [topic]?"
```

### Conceptual Search
```
Look in the context KB for concepts related to [topic] - semantic search will find relevant entries even if keywords don't match exactly.
```

### Problem-Based Search
```
Search the KB for any previous solutions to problems involving [technology/concept/issue].
```

## Alternative Prompts

### Before Starting Work
```
First, search the context KB using semantic search for any existing knowledge about [topic] that might help with this task. Try natural language queries like "How to implement [feature]" or "Common issues with [technology]".
```

### After Completing Work
```
Please add any important insights or decisions from this work to the context KB for future reference. Include the context and reasoning behind decisions.
```

### Quick Search (Semantic)
```
Search context KB: "[natural language description of what you're looking for]"
```

### Quick Search (Keywords)
```
Check context KB: [keywords]
```

### Save Knowledge
```
Save to context KB: [what we just learned/decided] including why this approach was chosen
```

## 🧠 Advanced Semantic Prompts

### Cross-Language Search
```
Search the KB in any language for [topic] - the multilingual embeddings will understand your query regardless of language.
```

### Contextual Understanding
```
Look for knowledge that relates to the concept of [abstract concept] even if the exact terms aren't mentioned.
```

### Problem Pattern Recognition
```
Search for any similar patterns or approaches we've used before for problems like [description of current problem].
```

### Architecture Decision Search
```
Find any previous architectural decisions related to [technology/pattern/approach] and the reasoning behind them.
```

## Ultra-Brief Triggers

### Semantic Search Triggers
- `KB semantic: "[natural language query]"`
- `Smart search: "[how do I...]"`
- `Concept search: [abstract concept]`
- `Pattern search: "[describe problem pattern]"`

### Traditional Triggers
- `KB check: [topic]`
- `Context search: [keywords]`
- `KB keyword: [specific terms]`

### Save Triggers
- `Save this to KB`
- `KB add: [insight]`
- `Context save: [what we learned]`

## 🎯 Contextual Prompt Examples

### Starting a Bug Fix
```
Before fixing this [bug type], search the KB for: "How have we handled similar [technology] issues before?" and "What are common causes of [symptom]?"
```

### Architecture Decision
```
Search the context KB for previous architectural decisions about [technology/pattern] and any lessons learned from similar implementations.
```

### Performance Optimization
```
Look in the KB for performance optimization techniques related to [technology/component] and any metrics or benchmarks we've collected.
```

### Code Review
```
Check if we have any coding standards, patterns, or anti-patterns documented for [language/framework] before reviewing this code.
```

### Deployment Issues
```
Search for any previous deployment issues with [technology/environment] and their solutions.
```

## 🔄 Feedback Loop Prompts

### After Problem Solving
```
Now that we've solved this [issue], please add the solution and root cause analysis to the context KB so we can find it quickly next time.
```

### After Learning Something
```
This insight about [topic] seems valuable - please capture it in the context KB with appropriate tags and context.
```

### After Making Decisions
```
Document this decision about [technology/approach] in the KB, including the alternatives considered and why we chose this path.
```

---

## 💡 Pro Tips for Semantic Search

1. **Be descriptive**: "Authentication token validation issues" works better than just "auth"
2. **Ask questions**: "How to optimize database queries" finds more relevant results than "database optimization"  
3. **Include context**: "Memory leaks in React components" is more specific than "memory leaks"
4. **Use natural language**: Write queries as if asking a colleague
5. **Try multiple phrasings**: Semantic search understands concepts expressed different ways