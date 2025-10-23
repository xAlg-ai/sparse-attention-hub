# Developer Guide: Implementing New Sparse Attention Baselines

## 1. Purpose and Intention

This guide provides concrete guidelines on the requirements for a Pull Request (PR) to be accepted in the sparse-attention-hub repository. The goal is to ensure that new baseline implementations are:

- **Correct**: Reproducible results that match or closely approximate the original paper/code
- **Modular**: Reuse existing components where possible and only implement novel logic
- **Well-documented**: Clear explanation of the method, experiments, and any differences from the original
- **Standardized**: Following our repository's evaluation and configuration patterns

## 2. What to Implement

### 2.1 Core Focus: Sparse Token Selection Logic

**Implement only the core logic of sparse token selection.** Many methods include additional components (dense layers, sink tokens, local windows, etc.) that may already be implemented in our repository. Also, the overall goal of the paper might be slightly different -- like showing CPU GPU management etc. But in this repository, we only want to compare the tradeoffs of sparsity and quality.

### 2.2 Modular Design Principle

Your implementation should follow this pattern:
```python
config = ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=128),           # Reuse existing
    LocalMaskerConfig(window_size=128),        # Reuse existing  
    ...                                        # more existing modules
    YourNovelMaskerConfig(                     # Implement this
        param1=value1,
        param2=value2
    )
])
```

### 2.3 Before Implementation Checklist

- [ ] Review existing maskers in `sparse_attention_hub/maskers/` 
- [ ] Identify which components of your method are already implemented
- [ ] Focus implementation effort only on the novel algorithmic contribution
- [ ] Ensure your masker integrates with the existing `ResearchAttentionConfig` framework

## 3. Correctness Guarantee Requirements

### 3.1 Reproduction Target

You must be able to reproduce **at least one table/result** from the original paper or author's code by matching exact or close-to-exact settings. This serves as our correctness guarantee. For this you might have to modify the code / settings to match the authors' setting.

### 3.2 Experimental Validation

Your PR must include the documentation of quality of implementation.

## 4. Documentation Requirements

### 4.1 Required Documentation Structure

Create a markdown file `docs/baselines/YourMethod.md` following this template:
See [MagicPig](./MagicPig.md) example for reference.

```markdown
# [Method Name] Sparse Attention Baseline

## 1. [Method Name] Algorithm Description
[Describe the core algorithm, its novelty, and theoretical foundation]

### Core Algorithm
[Technical details of the sparse attention mechanism]

### Example config in sparse-attention-hub
[Code snippet showing how to use your implementation]

## 2. Experiment Chosen for Reproducing
### Experimental Setup
[Dataset, model, and evaluation details]

## 3. Comparative Results
[Table comparing your results with original paper/code]

## 4. Explanations of Differences
[Detailed analysis of any performance gaps or implementation differences]

## References
[Citation to original paper]
```



### 4.2 Explain Differences from Original

You must clearly document:

- **Evaluation Setup Differences**: How your evaluation differs from the original (see [evaluation setup choices](../general/evaluation_setup.md))
- **Implementation Choices**: Any algorithmic modifications or simplifications made
- **Performance Impact**: How these choices affect the results
- **Theoretical Justification**: Why your choices are reasonable or necessary


### 4.3 Performance Sanity

Once you have a reasonable implementation, you should run attention computation profile to ensure that the implementation is not very slow. Check the [performance profile](../../profile/README.md) for details

```
cd sparse-attention-hub/profile
## change the file to have your sample config 
python3 profile_research_attention.py
```



## 5. Post-PR Integration Process

### 5.1 Review and Approval Process

1. **Initial Review**: Maintainers review code quality, correctness, and documentation
2. **Correctness Validation**: Reproduction results are verified
3. **Code Integration**: PR is merged after addressing review feedback

### 5.2 Author Engagement

Once the PR is approved and merged:

1. **Author Notification**: We reach out to original paper authors for feedback
2. **Feedback Incorporation**: Address any concerns or suggestions from authors
3. **Final Validation**: Ensure author satisfaction with the implementation

### 5.3 Leaderboard Integration

After author feedback is incorporated:

1. **Leaderboard Addition**: Method is added to our evaluation pipeline
2. **Continuous Monitoring**: Performance is tracked across benchmark updates
3. **Community Availability**: Implementation becomes available for research use




*This guide ensures that all baseline implementations maintain high quality standards while contributing meaningfully to the sparse attention research community.*
