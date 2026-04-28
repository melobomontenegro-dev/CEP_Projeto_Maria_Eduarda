# Avaliação da Qualidade de Processo Agroindustrial com CEP e Machine Learning

**Autora:** Maria Eduarda Lobo Montenegro
**Instituição:** Universidade de Brasília — Departamento de Engenharia de Produção
**Disciplina:** Controle Estatístico de Processos
**Data:** Abril de 2026
**Versão:** 1.0

---

## Resumo

Análise técnica integrada da qualidade de um processo agroindustrial de classificação de grãos de feijão, combinando **Controle Estatístico de Processos (CEP)** e **Machine Learning** para classificação automática multiclasse.

## Problema de Negócio

O processo de beneficiamento de feijão requer classificação de 7 variedades comerciais. A inspeção manual é lenta, subjetiva e custosa. Objetivo: desenvolver um modelo preditivo que classifique automaticamente variedades e avaliar a estabilidade do processo via CEP.

## Dataset

**Fonte:** UCI Machine Learning Repository — [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)

- **Amostras:** 13.611 grãos individuais
- **Features:** 16 atributos morfológicos contínuos (Area, Perimeter, MajorAxisLength, etc.)
- **Classes:** 7 variedades (SEKER, BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SIRA)

## Metodologia

1. **Definição do Problema** — contexto, objetivos, premissas
2. **EDA** — estatísticas descritivas, distribuições, correlações
3. **CEP** — cartas X-barra e R, índices Cp/Cpk/Pp/Ppk
4. **Preparação de Dados** — outliers (Z-score 3σ), padronização, split 70/30 estratificado
5. **Modelagem** — Regressão Logística, Random Forest, SVM (RBF), validação cruzada 5-fold
6. **Otimização** — GridSearchCV sobre Random Forest
7. **Avaliação** — métricas, matriz de confusão, feature importance
8. **Conclusões e Recomendações**

## Estrutura do Projeto

```
CEP_Projeto_Maria_Eduarda/
├── avaliacao_qualidade_graos.ipynb    # Notebook principal
├── README.md                           # Este arquivo
└── requirements.txt                    # Dependências
```

## Como Executar

### Google Colab (recomendado)
1. Faça upload do notebook no Colab
2. Execute a primeira célula (instala dependências automaticamente)
3. Execute as demais em sequência

### Local
```bash
pip install -r requirements.txt
jupyter notebook avaliacao_qualidade_graos.ipynb
```

## Reprodutibilidade

- Seed fixado em `42` em todos os componentes estocásticos
- Dataset carregado via `ucimlrepo` (versão oficial UCI)
- Resultados reproduzíveis com as versões de `requirements.txt`

## Referências

- Montgomery, D. C. (2020). *Introduction to Statistical Quality Control*. Wiley.
- Koklu, M., & Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques.
- Scikit-learn Documentation — https://scikit-learn.org/
