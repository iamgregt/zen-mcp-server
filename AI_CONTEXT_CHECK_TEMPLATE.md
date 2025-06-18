# AI Context Check-in Template

## Standard Check-in Prompt
```
Please check the context knowledge base for any relevant information about this [task/feature/issue] before proceeding. Use mcp__zen__context to search for related knowledge.
```

## Alternative Prompts

### Before Starting Work
```
First, search the context KB for any existing knowledge about [topic] that might help with this task.
```

### After Completing Work
```
Please add any important insights or decisions from this work to the context KB for future reference.
```

### Quick Search
```
Check context KB: [keywords]
```

### Save Knowledge
```
Save to context KB: [what we just learned/decided]
```

## Ultra-Brief Triggers
- `KB check: [topic]`
- `Save this to KB`
- `Context search: [keywords]`
- `KB add: [insight]`