# Typed Proof Plans

**A typed intermediate representation for creative mathematical reasoning and automated theorem proving.**

---

## Description

Typed Proof Plans represent mathematical reasoning as structured, typed proof plans rather than as unstructured proof traces or isolated theorem-search steps.

A plan consists of typed mathematical statements, method schemas and instances, obligations, substitutions, certificates, provenance, and trust information. These elements form a directed proof graph whose committed support can be checked for structural validity, type consistency, trusted method use, and acyclicity.

The system is designed as an intermediate layer between creative reasoning and formal verification. It can represent incomplete or speculative reasoning while preserving the boundaries between generated ideas, mathematical transformations, external evidence, and kernel-verified proof steps.

The project follows the typed-proof-plan model described in *Typed Proof Plans as the Intermediate Representation for Creative Mathematical Reasoning*, with the goal of supporting reliable, extensible, and auditable mathematical proof planning.