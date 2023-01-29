---
title: "Test Page"
author: avishek
usemathjax: true
tags: ["Test Tag"]
draft: false
---

# Test Page

## LaTeX Test

$$
\text{Confidence Interval } = \hat{X} \pm Z.\frac{\sigma}{\sqrt{n} }
$$

## IncludeChart-ChartJS Test

{% include_chart myChart!300px!300px!{
type: 'polarArea',
data: {
labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
datasets: [{
label: '# of Votes',
data: [12, 19, 3, 5, 2, 3],
borderWidth: 1
}]
},
options: {
responsive: false,
maintainAspectRatio: false,
scales: {
y: {
beginAtZero: true
}
}
}
} %}

## MermaidJS Test

{% mermaid %}
graph LR;
debt[Tech Debt]-->principal[Cost of Fixing Debt: Principal];
debt-->interest[Recurring Cost: Interest];
debt-->risk[Risk-Related Cost];
architecture_decision[Architecture Decision]-->resources[Cloud Resources];
microservice-->database[Cloud DB Resources];
microservice-->development_cost[Development Cost];
microservice-->latency[Latency];
microservice-->bugs[Fixing bugs];
microservice-->downtime[Downtime]-->lost_transactions[Lesser Lost Transactions];

style microservice fill:#006f00,stroke:#000,stroke-width:2px,color:#fff
style debt fill:#006fff,stroke:#000,stroke-width:2px,color:#fff
style architecture_decision fill:#8f0f00,stroke:#000,stroke-width:2px,color:#fff
{% endmermaid %}

## IncludeCodeTag Test
```python
{% include_code https://raw.githubusercontent.com/asengupta/transformers-paper-implementation/main/transformer.py!223!242%}
```
