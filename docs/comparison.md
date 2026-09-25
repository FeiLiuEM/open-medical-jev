# Relation to other "Jev" projects

Open Medical Jev is an **independent project**. This page describes, in
neutral terms, how it relates to other projects in the "Jev" ecosystem. It is
not a ranking, and it is not a description of their internals — check each
project's own pages for authoritative details. Facts below were checked on
**2026-09-25**.

## Different constructions

| project | public description | where |
|---|---|---|
| Jev (TypeSafe) | hosted decision service (the reference this project is measured against, v1.13.0) | typesafe.ai |
| OpenJev (openjev) | open-weight release of the Jev-style line: trained (once) for selection consistency and option-order robustness; weights under their published license (check current terms — CC BY-NC on our check date) | HF: openjev/openjev |
| Medical-OpenJev | a medical-domain adaptation distributed by independent researchers | HF: fancc28/vindahi |
| MedJev | community medical project | GitHub: JunMa11/MedJev |
| ClinicalJev | community medical project (in development at check time) | GitHub: xzhou-code/clinical-jev |
| **Open Medical Jev (this repo)** | two **frozen** readers (Qwen3.5-27B + 35B-A3B) + routing layer; nothing trained; code + recipe only; no weights redistributed | GitHub: FeiLiuEM/open-medical-jev |

## Comparison basis

* Accuracy comparisons in [reports/results_summary.md](../reports/results_summary.md)
  use Jev 1.13.0 on identical item sets where possible, with the protocol
  pinned in [protocol.md](protocol.md); gated numbers always carry coverage.
* OpenJev numbers, when quoted, are its own published figures — not
  re-measured by us unless stated.
* Community medical projects are listed for orientation; this repository does
  not benchmark against them.

## Non-affiliation

Open Medical Jev is not affiliated with TypeSafe; "Jev" is their product.
It is not affiliated with the OpenJev project, Medical-OpenJev, MedJev or
ClinicalJev. The name is used descriptively to place this project in the
ecosystem ("Jev-class judgment from frozen models"), and this project does not
claim any trademark rights in "Jev".

## Ecosystem index

The community-maintained index at `hanxiao.io/all-about-jev` lists Jev-derived
work from many authors; it is maintained by a third party and this project is
listed there (or pending listing) like any other community entry.
