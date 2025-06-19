**Deliverables**:
- Add to `/providers/gemini.py`:
  - Import statements for embedding support
  - Rate limiter initialization in `__init__`
  - Complete `get_embedding()` method with rate limiting

**Success Criteria**:
- Method can generate 3072-dimensional embeddings
- Rate limiting prevents 429 errors
- Handles both RETRIEVAL_QUERY and RETRIEVAL_DOCUMENT types

---

### Task 1.2: ChromaDB Compatibility

#### Task 1.2.1: Test ChromaDB Dimension Flexibility
**Assigned to**: AI Agent 4  
**Status**: ⬜ Not Started

**Objective**: Verify ChromaDB can handle different embedding dimensions.

**Discovery Steps**:
1. Read current ChromaDB initialization in `/utils/chroma_provider.py`
2. Check ChromaDB documentation for dimension handling

**Deliverables**:
- Create `/tests/test_chroma_dimensions.py` that:
  - Creates collection with 1024 dimensions
  - Creates collection with 3072 dimensions
  - Verifies both work correctly

**Success Criteria**:
- Test proves ChromaDB handles both dimensions
- No errors when switching dimensions

---

#### Task 1.2.2: Design Collection Migration Strategy
**Assigned to**: AI Agent 5  
**Status**: ⬜ Not Started

**Objective**: Create safe migration approach for existing collections.

**Discovery Steps**:
1. Read ChromaDB collection management in `/utils/chroma_provider.py`
2. Understand current collection naming and structure
3. Review `/GEMINI_EMBEDDING_MIGRATION_GUIDE.md` section on migration

**Deliverables**:
- Create `/utils/collection_migrator.py` with:
  - Method to backup existing collection
  - Method to create new collection with new dimensions
  - Method to swap collections safely

**Success Criteria**:
- Migration preserves all metadata
- Rollback is possible
- No data loss during migration

---

## Phase 2: Implementation

### Task 2.1: Core Implementation

#### Task 2.1.1: Implement Gemini Embedding in ChromaProvider
**Assigned to**: AI Agent 6  
**Status**: ⬜ Not Started

**Objective**: Modify ChromaProvider to use Gemini embeddings.

**Discovery Steps**:
1. Read `/utils/chroma_provider.py`
2. Read `/providers/google.py` (should have `get_embedding` from Task 1.1.3)
3. Review `/GEMINI_EMBEDDING_MIGRATION_GUIDE.md` for implementation

**Deliverables**:
- Modify `/utils/chroma_provider.py`:
  - Add `use_gemini` parameter to `__init__`
  - Update `_compute_embedding` to use Gemini when enabled
  - Maintain backward compatibility

**Success Criteria**:
- Can switch between Gemini and SentenceTransformer
- Generates correct dimension embeddings
- Handles errors gracefully

---

#### Task 2.1.2: Add Migration Method
**Assigned to**: AI Agent 7  
**Status**: ⬜ Not Started

**Objective**: Implement batch migration with rate limiting.

**Discovery Steps**:
1. Read `/utils/chroma_provider.py`
2. Read `/utils/collection_migrator.py` (from Task 1.2.2)
3. Understand rate limiting requirements (10 RPM, 1000 RPD)

**Deliverables**:
- Add to `/utils/chroma_provider.py`:
  - `migrate_to_gemini_embeddings()` method
  - Progress tracking and logging
  - Rate limit compliance (5 items per batch)

**Success Criteria**:
- Migrates without hitting rate limits
- Shows accurate progress and ETA
- Handles failures gracefully

---

### Task 2.2: Testing

#### Task 2.2.1: Create Unit Tests
**Assigned to**: AI Agent 8  
**Status**: ⬜ Not Started

**Objective**: Comprehensive unit tests for Gemini embedding integration.

**Discovery Steps**:
1. Read existing tests in `/tests/`
2. Understand test patterns used in the project
3. Review all new methods added in previous tasks

**Deliverables**:
- Create `/tests/test_gemini_embeddings.py` with tests for:
  - Embedding generation
  - Rate limiting
  - Dimension verification
  - Error handling
  - Task type selection

**Success Criteria**:
- All tests pass
- >90% code coverage for new code
- Tests are independent and repeatable

---

#### Task 2.2.2: Create Integration Tests
**Assigned to**: AI Agent 9  
**Status**: ⬜ Not Started

**Objective**: End-to-end tests for the complete migration flow.

**Discovery Steps**:
1. Read `/tests/` for integration test patterns
2. Understand Context tool workflow
3. Review migration process

**Deliverables**:
- Create `/tests/test_migration_integration.py` with:
  - Test adding entries with Gemini
  - Test searching with Gemini
  - Test migration of small dataset
  - Test rollback scenario

**Success Criteria**:
- Integration tests prove system works end-to-end
- No regressions in existing functionality

---

## Phase 3: Migration Execution

### Task 3.1: Pre-Migration

#### Task 3.1.1: Create Migration Readiness Checklist
**Assigned to**: AI Agent 10  
**Status**: ⬜ Not Started

**Objective**: Ensure system is ready for migration.

**Discovery Steps**:
1. Read all code changes from Phase 1 & 2
2. Check current data in `/data/kb/`
3. Review rollback procedures

**Deliverables**:
- Create `/docs/MIGRATION_READINESS_CHECKLIST.md` with:
  - Pre-flight checks
  - Backup procedures
  - Verification steps
  - Go/no-go criteria

**Success Criteria**:
- Comprehensive checklist
- Clear rollback procedures
- Risk mitigation identified

---

#### Task 3.1.2: Backup Current Data
**Assigned to**: AI Agent 11  
**Status**: ⬜ Not Started

**Objective**: Create complete backup of current embeddings and data.

**Discovery Steps**:
1. Locate all data directories (`/data/kb/`, `/data/kb/.chroma/`)
2. Understand backup requirements
3. Check available disk space

**Deliverables**:
- Create `/scripts/backup_embeddings.py` that:
  - Backs up all ChromaDB data
  - Backs up all JSON entries
  - Creates restoration script
  - Timestamps backup

**Success Criteria**:
- Complete backup created
- Restoration tested and verified
- Backup location documented

---

### Task 3.2: Migration

#### Task 3.2.1: Migrate Test Dataset
**Assigned to**: AI Agent 12  
**Status**: ⬜ Not Started

**Objective**: Migrate a small test dataset first.

**Discovery Steps**:
1. Run backup script from Task 3.1.2
2. Read migration method implementation
3. Select 10-20 test entries

**Deliverables**:
- Execute migration on test dataset
- Document in `/docs/TEST_MIGRATION_RESULTS.md`:
  - Time taken
  - Any errors encountered
  - Search quality comparison
  - Performance metrics

**Success Criteria**:
- Test migration completes successfully
- Search quality maintained or improved
- No data loss

---

#### Task 3.2.2: Execute Full Migration
**Assigned to**: AI Agent 13  
**Status**: ⬜ Not Started

**Objective**: Migrate all production data to Gemini embeddings.

**Discovery Steps**:
1. Verify test migration success (Task 3.2.1)
2. Count total entries to migrate
3. Calculate time needed (10 per minute limit)

**Deliverables**:
- Execute full migration
- Create `/docs/MIGRATION_LOG.md` with:
  - Start/end times
  - Entries processed
  - Any errors
  - Final verification

**Success Criteria**:
- All entries migrated
- No data loss
- System functional post-migration

---

## Phase 4: Verification & Cleanup

### Task 4.1: Verification

#### Task 4.1.1: Verify Search Quality
**Assigned to**: AI Agent 14  
**Status**: ⬜ Not Started

**Objective**: Ensure search quality meets or exceeds previous implementation.

**Discovery Steps**:
1. Create test queries covering different scenarios
2. Compare results between old and new embeddings
3. Test edge cases

**Deliverables**:
- Create `/tests/search_quality_comparison.py`
- Document results in `/docs/SEARCH_QUALITY_REPORT.md`

**Success Criteria**:
- Search returns relevant results
- Performance is acceptable
- No regression in quality

---

#### Task 4.1.2: Performance Benchmarking
**Assigned to**: AI Agent 15  
**Status**: ⬜ Not Started

**Objective**: Benchmark performance impact of API-based embeddings.

**Discovery Steps**:
1. Design performance test scenarios
2. Measure latency for various operations
3. Compare with previous implementation

**Deliverables**:
- Create `/benchmarks/embedding_performance.py`
- Document in `/docs/PERFORMANCE_ANALYSIS.md`:
  - Latency measurements
  - Throughput limits
  - Cost analysis

**Success Criteria**:
- Clear performance metrics documented
- Acceptable performance for use cases
- Cost within budget

---

### Task 4.2: Cleanup

#### Task 4.2.1: Remove Old Dependencies
**Assigned to**: AI Agent 16  
**Status**: ⬜ Not Started

**Objective**: Clean up SentenceTransformer code if no longer needed.

**Discovery Steps**:
1. Verify Gemini embeddings working in production
2. Check for any remaining uses of SentenceTransformer
3. Review fallback requirements

**Deliverables**:
- Update `/utils/chroma_provider.py` to remove old code
- Update `requirements.txt` if applicable
- Document changes in `/docs/CLEANUP_LOG.md`

**Success Criteria**:
- Old code removed safely
- No broken dependencies
- Clean codebase

---

#### Task 4.2.2: Update Documentation
**Assigned to**: AI Agent 17  
**Status**: ⬜ Not Started

**Objective**: Update all documentation to reflect new embedding system.

**Discovery Steps**:
1. Find all documentation mentioning embeddings
2. Review changes made throughout migration
3. Identify user-facing changes

**Deliverables**:
- Update `/CLAUDE.md` with new embedding details
- Update `/README.md` if needed
- Create `/docs/GEMINI_EMBEDDINGS_USER_GUIDE.md`

**Success Criteria**:
- Documentation accurate and complete
- User guide clear and helpful
- No outdated references

---

## Task Completion Tracking

### Phase 1: Foundation & Discovery
- [ ] Task 1.1.1: Analyze Current Embedding Implementation
- [ ] Task 1.1.2: Verify Google Provider Setup
- [ ] Task 1.1.3: Create Embedding Provider Interface
- [ ] Task 1.2.1: Test ChromaDB Dimension Flexibility
- [ ] Task 1.2.2: Design Collection Migration Strategy

### Phase 2: Implementation
- [ ] Task 2.1.1: Implement Gemini Embedding in ChromaProvider
- [ ] Task 2.1.2: Add Migration Method
- [ ] Task 2.2.1: Create Unit Tests
- [ ] Task 2.2.2: Create Integration Tests

### Phase 3: Migration Execution
- [ ] Task 3.1.1: Create Migration Readiness Checklist
- [ ] Task 3.1.2: Backup Current Data
- [ ] Task 3.2.1: Migrate Test Dataset
- [ ] Task 3.2.2: Execute Full Migration

### Phase 4: Verification & Cleanup
- [ ] Task 4.1.1: Verify Search Quality
- [ ] Task 4.1.2: Performance Benchmarking
- [ ] Task 4.2.1: Remove Old Dependencies
- [ ] Task 4.2.2: Update Documentation

---

## Notes for AI Agents

1. **Independence**: Each task is designed to be completed without any conversation history
2. **Discovery**: Each task includes discovery steps to understand the current state
3. **Deliverables**: Each task produces concrete files or code changes
4. **Verification**: Each task includes success criteria for validation
5. **No Context Needed**: An agent can start any task by reading this file and following the steps

## Migration Statistics

- **Total Tasks**: 17
- **Estimated Hours**: 20-30 (assuming 1-2 hours per task)
- **Agents Needed**: 17 (or fewer if reusing agents after task completion)
- **Migration Time**: Limited by 10 RPM rate limit
  - 100 entries: ~10 minutes
  - 1,000 entries: ~100 minutes
  - >1,000 entries: Multiple days required