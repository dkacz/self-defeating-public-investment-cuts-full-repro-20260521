# Reproducible Notebook Best-Practice Sources

This note records the external standards used to frame the 2026-05-21 public
reproducibility audit packet.

## Sources

1. Rule, A. et al. (2019), "Ten simple rules for writing and sharing computational analyses in Jupyter Notebooks", PLOS Computational Biology, https://doi.org/10.1371/journal.pcbi.1007007.
2. Sandve, G. K. et al. (2013), "Ten Simple Rules for Reproducible Computational Research", PLOS Computational Biology, https://doi.org/10.1371/journal.pcbi.1003285.
3. Wilson, G. et al. (2017), "Good enough practices in scientific computing", PLOS Computational Biology, https://doi.org/10.1371/journal.pcbi.1005510.
4. Pimentel, J. F. et al. (2019), "A Large-Scale Study About Quality and Reproducibility of Jupyter Notebooks", ICSE, https://leomurta.github.io/papers/pimentel2019a.pdf.

## Audit Implications

- The notebook must tell the reader what is being recomputed and why, not merely display stored tables.
- The notebook must run from top to bottom in a clean environment.
- Data, code, outputs and validation targets must have explicit provenance.
- Frozen outputs are acceptable only as validation targets; they must not substitute for estimation.
- Parameters, sample windows, units, transformations and dependencies must be visible.
- The public path and notebook filename must be versioned so browser-side workspace state cannot silently reopen a stale notebook.
- Validation should compare recomputed outputs to named files with hashes and tolerances, not to inline magic numbers.
